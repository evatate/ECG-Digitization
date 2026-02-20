#   1. Load pretrained ResNet34 U-Net weights
#   2. Fine-tune 1-2 epochs on clean `0001` training images
#   3. Inference: predict heatmap per test image (once per image, all 12 leads)
#   4. Post-process: weighted centroid → gap interpolation → Savitzky-Golay → resample
#   5. Einthoven correction (II ≈ I + III)
#   6. Build submission.parquet aligned to sample_submission

import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, resample

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


DATA_DIR   = Path('/kaggle/input/physionet-ecg-image-digitization')
MODEL_DIR  = Path('/kaggle/input/models/eliasstenhede/open-ecg-digitizer-weights/pytorch/default/1')
WORK_DIR   = Path('/kaggle/working')

TRAIN_CSV  = DATA_DIR / 'train.csv'
TEST_CSV   = DATA_DIR / 'test.csv'
TRAIN_DIR  = DATA_DIR / 'train'
TEST_DIR   = DATA_DIR / 'test'
SAMPLE_SUB = DATA_DIR / 'sample_submission.parquet'

# Image
IMG_W         = 2048
IMG_H         = 1024
HEADER_CROP   = 0.15     # crop top 15% (header/text area)

# Fine-tuning
TRAIN_SEGMENT = '0001'   # clean digital images only
MAX_TRAIN     = 1500     # cap for speed (~45 min on T4)
EPOCHS        = 2
BATCH_SIZE    = 4
LR            = 3e-4
POS_WEIGHT    = 10.0     # upweight sparse waveform pixels

# Post-processing
GAUSS_SIGMA   = 0.5
GAP_THRESHOLD = 0.08     # heatmap confidence below this = gap
SAVGOL_WIN    = 9        # must be odd
SAVGOL_POLY   = 2

# Standard 12-lead ECG grid layout (hardcoded)
# Row 0: I    | aVR | V1 | V4
# Row 1: II   | aVL | V2 | V5
# Row 2: III  | aVF | V3 | V6
# Row 3: Lead II long strip (full width, 10 seconds)
LEAD_GRID = [
    ['I',   'aVR', 'V1', 'V4'],
    ['II',  'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
LONG_LEAD  = 'II'
SHORT_DUR  = 2.5   # seconds
LONG_DUR   = 10.0  # seconds


def build_model():
    return smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )


def load_pretrained(model, model_dir: Path):
    """
    Try common checkpoint filenames in the model dir and load whichever exists.
    Handles plain state_dict or wrapped dicts.
    """
    candidates = ['model.pth', 'weights.pth', 'unet.pth', 'checkpoint.pth',
                  'model.pt', 'weights.pt']
    path = None
    for fname in candidates:
        p = model_dir / fname
        if p.exists():
            path = p
            break
    if path is None:
        # Fall back: take the first .pth/.pt file found
        matches = list(model_dir.glob('*.pth')) + list(model_dir.glob('*.pt'))
        if matches:
            path = matches[0]
        else:
            raise FileNotFoundError(f'No .pth/.pt file found in {model_dir}')

    print(f'Loading weights from: {path}')
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    return model


model = build_model()
model = load_pretrained(model, MODEL_DIR)
model = model.to(DEVICE)
print('Pretrained weights loaded.')


def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Load → grayscale → crop header → resize → normalize → [1, 1, H, W] tensor.
    """
    img = Image.open(img_path).convert('L')
    w, h = img.size
    crop_top = int(h * HEADER_CROP)
    img = img.crop((0, crop_top, w, h))
    img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Invert if dark background (traces should be dark on light)
    if arr.mean() < 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def signal_to_mask_band(signal: np.ndarray, band_h: int, band_w: int) -> np.ndarray:
    """
    Convert a 1D signal array into a sparse binary mask of shape (band_h, band_w).
    Center row = 0 mV; signal occupies ~60% of band height.
    """
    mask = np.zeros((band_h, band_w), dtype=np.float32)
    N = len(signal)
    if N == 0:
        return mask

    sig_range = signal.max() - signal.min()
    if sig_range < 1e-6:
        return mask

    center  = band_h / 2.0
    scale   = (band_h * 0.6) / sig_range
    sig_c   = signal - signal.mean()

    xs = np.linspace(0, band_w - 1, N).astype(int)
    ys = (center - sig_c * scale).astype(int)
    ys = np.clip(ys, 0, band_h - 1)

    for x, y in zip(xs, ys):
        mask[y, x] = 1.0
        if y > 0:        mask[y - 1, x] = 0.5
        if y < band_h-1: mask[y + 1, x] = 0.5

    return mask


def build_full_mask(sig_df: pd.DataFrame) -> np.ndarray:
    """
    Build a full (IMG_H × IMG_W) mask from all 12 leads using the hardcoded grid.
    """
    mask  = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    row_h = IMG_H // 4
    col_w = IMG_W // 4

    for row_idx, row_leads in enumerate(LEAD_GRID):
        y0 = row_idx * row_h
        for col_idx, lead in enumerate(row_leads):
            if lead not in sig_df.columns:
                continue
            full_sig = sig_df[lead].values
            # Each column shows 2.5 s = 1/4 of the 10-second signal
            quarter  = len(full_sig) // 4
            seg      = full_sig[col_idx * quarter: (col_idx + 1) * quarter]
            x0 = col_idx * col_w
            band = signal_to_mask_band(seg, row_h, col_w)
            mask[y0: y0 + row_h, x0: x0 + col_w] = band

    # Long Lead II strip (row 3, full width)
    if LONG_LEAD in sig_df.columns:
        y0   = 3 * row_h
        band = signal_to_mask_band(sig_df[LONG_LEAD].values, IMG_H - y0, IMG_W)
        mask[y0:, :] = band

    return mask


class ECGDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, max_samples: int = None):
        self.samples = []
        ids = train_df['id'].unique()
        if max_samples:
            ids = ids[:max_samples]
        for ecg_id in ids:
            ecg_id = str(ecg_id)
            img_path = TRAIN_DIR / ecg_id / f'{ecg_id}-{TRAIN_SEGMENT}.png'
            sig_path = TRAIN_DIR / ecg_id / f'{ecg_id}.csv'
            if img_path.exists() and sig_path.exists():
                self.samples.append((str(img_path), str(sig_path)))
        print(f'Dataset: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sig_path = self.samples[idx]
        image   = preprocess_image(img_path).squeeze(0)          # [1, H, W]
        sig_df  = pd.read_csv(sig_path)
        mask    = build_full_mask(sig_df)
        mask_t  = torch.from_numpy(mask).unsqueeze(0)            # [1, H, W]
        return image, mask_t


train_meta = pd.read_csv(TRAIN_CSV, dtype={"id": str})
dataset    = ECGDataset(train_meta, max_samples=MAX_TRAIN)
loader     = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2, pin_memory=True)

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR)
criterion  = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
)
scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
            preds = model(images)
            loss  = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}  '
                  f'Step {batch_idx}/{len(loader)}  '
                  f'Loss: {loss.item():.4f}')

    print(f'==> Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}')

print('Fine-tuning complete.')
torch.save(model.state_dict(), WORK_DIR / 'finetuned_unet.pth')

def estimate_px_per_mv(img_band: np.ndarray) -> float:
    """
    Estimate pixels-per-mV from the RAW GRAYSCALE image band (not the heatmap).

    Standard ECG paper: large box = 5mm = 0.5 mV vertically.
    Grid lines appear as periodic dark horizontal stripes in the image.
    We detect their spacing via FFT of the row-mean projection.
    Falls back to H/3.0 if detection fails.
    """
    H, W = img_band.shape
    default = H / 3.0

    # Invert so grid lines are peaks (they are darker than background)
    inv = 1.0 - img_band
    row_proj = inv.mean(axis=1)   # (H,)
    if row_proj.std() < 1e-4:
        return default

    fft_mag = np.abs(np.fft.rfft(row_proj - row_proj.mean()))
    freqs   = np.fft.rfftfreq(H)
    fft_mag[0] = 0
    # Grid lines are spaced ~H/4 to H/2 px apart for large boxes in a 4-row layout
    # Clamp to plausible range: period between H/20 and H/2
    fft_mag[freqs < 2.0/H]  = 0
    fft_mag[freqs > 0.25]   = 0

    if fft_mag.max() < 1e-6:
        return default

    dominant_freq = freqs[np.argmax(fft_mag)]
    if dominant_freq < 1e-6:
        return default

    grid_spacing_px = 1.0 / dominant_freq
    px_per_mv = grid_spacing_px / 0.5   # large box = 0.5 mV

    # Must be physically reasonable
    px_per_mv = float(np.clip(px_per_mv, H / 8.0, H * 2.0))
    return px_per_mv


def heatmap_to_signal(heatmap: np.ndarray, target_len: int, img_band: np.ndarray = None) -> np.ndarray:
    """
    Convert a 2D heatmap band (H x W) for one lead into a 1D signal.

    Steps:
      1. Gaussian smooth
      2. Weighted centroid per column (subpixel y)
      3. Gap detection (low confidence -> NaN) + linear interpolation
      4. Savitzky-Golay smoothing
      5. Fourier resample to target_len
      6. Pixel row -> millivolts using grid-calibrated scale + median baseline
    """
    H, W = heatmap.shape

    # 1. Smooth
    hm = gaussian_filter(heatmap, sigma=GAUSS_SIGMA)

    # 2. Weighted centroid per column
    ys       = np.arange(H, dtype=np.float32)
    col_conf = hm.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_y = np.where(
            col_conf > 0,
            (hm * ys[:, None]).sum(axis=0) / col_conf,
            np.nan
        )

    # 3. Gap detection + interpolation
    max_conf = hm.max(axis=0)
    col_y[max_conf < GAP_THRESHOLD] = np.nan

    valid = np.where(~np.isnan(col_y))[0]
    if len(valid) < 2:
        return np.zeros(target_len, dtype=np.float32)

    fill  = interp1d(valid, col_y[valid], kind='linear',
                     bounds_error=False,
                     fill_value=(col_y[valid[0]], col_y[valid[-1]]))
    col_y = fill(np.arange(W))

    # 4. Savitzky-Golay
    win   = min(SAVGOL_WIN, W if W % 2 == 1 else W - 1)
    col_y = savgol_filter(col_y, win, SAVGOL_POLY)

    # 5. Fourier resample
    signal = resample(col_y, target_len)

    # 6. Pixel -> mV
    calibration_src = img_band if img_band is not None else heatmap
    px_per_mv   = estimate_px_per_mv(calibration_src)

    # Baseline: use the median of col_y as the isoelectric reference.
    # The 10th percentile overshoots downward for leads with large negative deflections.
    baseline_px = np.median(col_y)
    signal      = -(signal - baseline_px) / px_per_mv  # invert: top = positive

    # 7. Isoelectric correction: find the quietest 20% of the signal
    #    (lowest local variance windows = TP segments between beats)
    #    and shift the whole signal so those segments sit at 0 mV.
    if len(signal) > 20:
        win = max(5, len(signal) // 20)   # window ~5% of signal length
        # Compute rolling variance
        n_wins   = len(signal) - win + 1
        roll_var = np.array([signal[j:j+win].var() for j in range(n_wins)])
        # Take the quietest 20% of windows
        thresh   = np.percentile(roll_var, 20)
        quiet    = roll_var <= thresh
        quiet_vals = np.concatenate([signal[j:j+win] for j in range(n_wins) if quiet[j]])
        if len(quiet_vals) > 0:
            iso_level = np.median(quiet_vals)
            signal    = signal - iso_level   # shift so isoelectric = 0 mV

    return signal.astype(np.float32)


def einthoven_correction(leads: dict) -> dict:
    """
    Einthoven's law: Lead II = Lead I + Lead III.
    Lead II is 10s (4× longer than I and III which are 2.5s).
    We correct only over the overlapping first 2.5s, then restore the full II.
    """
    if not all(k in leads for k in ('I', 'II', 'III')):
        return leads
    I, II_full, III = leads['I'], leads['II'], leads['III']
    # Use the shorter length for the overlapping correction window
    n = min(len(I), len(II_full), len(III))
    II_short = II_full[:n]
    residual       = II_short - (I[:n] + III[:n])
    leads['I']     = I   + residual / 3.0
    leads['III']   = III + residual / 3.0
    # Apply correction to the first n samples of Lead II; leave the rest unchanged
    II_corrected        = II_full.copy()
    II_corrected[:n]    = II_short - residual * 2.0 / 3.0
    leads['II']         = II_corrected
    return leads


def get_lead_band(lead: str, H: int, W: int):
    """
    Return (y0, y1, x0, x1) pixel slice for a lead in the hardcoded grid.
    """
    row_h = H // 4
    col_w = W // 4

    if lead == LONG_LEAD:                          # full-width long strip
        return (3 * row_h, H, 0, W)

    for row_idx, row_leads in enumerate(LEAD_GRID):
        if lead in row_leads:
            col_idx = row_leads.index(lead)
            return (row_idx * row_h, (row_idx + 1) * row_h,
                    col_idx * col_w, (col_idx + 1) * col_w)

    raise ValueError(f'Lead {lead} not found in layout')


@torch.no_grad()
def predict_heatmap(img_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    pred    = model(img_tensor.to(DEVICE))
    heatmap = torch.sigmoid(pred).squeeze().cpu().numpy()
    return heatmap


val_id  = train_meta['id'].iloc[0]
val_img = str(TRAIN_DIR / val_id / f'{val_id}-{TRAIN_SEGMENT}.png')
val_sig = pd.read_csv(TRAIN_DIR / val_id / f'{val_id}.csv')
val_fs  = float(train_meta.loc[train_meta['id'] == val_id, 'fs'].iloc[0])

img_tensor = preprocess_image(val_img)
heatmap    = predict_heatmap(img_tensor)
H_hm, W_hm = heatmap.shape

# LEAD_NAMES has 12 leads; grid is 4 rows x 3 cols = 12 subplots, one per lead in order
fig, axes = plt.subplots(4, 3, figsize=(18, 12))
axes_flat = axes.flatten()   # index 0..11 maps directly to LEAD_NAMES order
img_np_val = img_tensor.squeeze().cpu().numpy()

# Normalise column names
val_sig.columns = [c.strip() for c in val_sig.columns]
print(f"Signal columns: {list(val_sig.columns)}")

val_sig_len = train_meta.loc[train_meta['id'] == val_id, 'sig_len'].iloc[0]

for i, lead in enumerate(LEAD_NAMES):
    ax  = axes_flat[i]
    dur = LONG_DUR if lead == LONG_LEAD else SHORT_DUR
    n   = int(np.floor(val_fs * dur))

    y0, y1, x0, x1 = get_lead_band(lead, H_hm, W_hm)
    pred_sig = heatmap_to_signal(heatmap[y0:y1, x0:x1], n, img_np_val[y0:y1, x0:x1])

    # GT: all leads are stored as full 10s signal; take the right 2.5s segment
    # Short leads show columns 0,1,2,3 of the 10s signal in order
    if lead not in val_sig.columns:
        print(f"WARNING: lead {lead} not found in signal CSV, skipping GT")
        gt_sig = np.zeros(n)
    elif lead == LONG_LEAD:
        gt_sig = val_sig[lead].values[:n]
    else:
        # Find which column position this lead occupies in the grid
        col_idx = None
        for row in LEAD_GRID:
            if lead in row:
                col_idx = row.index(lead)
                break
        if col_idx is not None:
            quarter = int(np.floor(val_fs * SHORT_DUR))
            start   = col_idx * quarter
            gt_sig  = val_sig[lead].values[start: start + n]
        else:
            gt_sig = val_sig[lead].values[:n]

    t = np.linspace(0, dur, n)
    ax.plot(t, gt_sig,   'b', lw=0.9, alpha=0.7, label='GT')
    ax.plot(t, pred_sig, 'r', lw=0.9, alpha=0.7, label='Pred')
    ax.set_title(lead, fontsize=10)
    ax.set_xlabel('s'); ax.set_ylabel('mV')
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle(f'Validation sample: {val_id}', fontsize=13)
plt.tight_layout()
plt.savefig(WORK_DIR / 'validation_plot.png', dpi=100)
plt.show()
print('Validation plot saved.')


test_meta  = pd.read_csv(TEST_CSV)
sample_sub = pd.read_parquet(SAMPLE_SUB)

rows = []

for base_id, group in test_meta.groupby('id'):
    img_path = TEST_DIR / f'{base_id}.png'
    if not img_path.exists():
        print(f'WARNING: missing {img_path}')
        continue

    # Run model once per image
    img_tensor   = preprocess_image(str(img_path))
    heatmap      = predict_heatmap(img_tensor)
    H_hm, W_hm   = heatmap.shape
    lead_signals = {}

    for _, row in group.iterrows():
        lead   = row['lead']
        n_rows = int(row['number_of_rows'])
        y0, y1, x0, x1 = get_lead_band(lead, H_hm, W_hm)
        img_np = img_tensor.squeeze().cpu().numpy()
        signal = heatmap_to_signal(heatmap[y0:y1, x0:x1], n_rows, img_np[y0:y1, x0:x1])
        lead_signals[lead] = signal

    # Physics correction
    lead_signals = einthoven_correction(lead_signals)

    # Flatten to submission rows
    for lead, signal in lead_signals.items():
        for row_id, val in enumerate(signal):
            rows.append({
                'id'   : f'{base_id}_{row_id}_{lead}',
                'value': float(val)
            })

    print(f'Done: {base_id}')

sub_df = pd.DataFrame(rows)

# Align to sample submission ordering
sub_df = sample_sub[['id']].merge(sub_df, on='id', how='left')
sub_df['value'] = sub_df['value'].fillna(0.0)

out_path = WORK_DIR / 'submission.parquet'
sub_df.to_parquet(out_path, index=False)
print(f'\nSubmission saved → {out_path}')
print(f'Rows: {len(sub_df)}  |  Nulls: {sub_df["value"].isna().sum()}')
print(f'Value range: {sub_df["value"].min():.4f} → {sub_df["value"].max():.4f}')