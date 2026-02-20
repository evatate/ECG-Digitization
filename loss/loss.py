import torch
from torch import nn
from torch.nn import functional as F

TEXT_CLASS: int = 1
SIGNAL_CLASS: int = 2


def rgb_to_one_hot(rgb_labels: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB labels to one-hot encoded format.

    Args:
        rgb_labels: Tensor of shape (N, 3, H, W) with RGB labels.

    Returns:
        One-hot encoded tensor of shape (N, 4, H, W)
    """
    labels_one_hot = torch.zeros(
        (rgb_labels.shape[0], 4, rgb_labels.shape[2], rgb_labels.shape[3]), device=rgb_labels.device
    )
    rgb_labels_sum = rgb_labels.sum(dim=1, keepdim=True)
    rgb_labels_sum = torch.where(rgb_labels_sum > 1, rgb_labels_sum, torch.ones_like(rgb_labels_sum))
    rgb_labels = rgb_labels / rgb_labels_sum

    labels_one_hot[:, 0, :, :] = rgb_labels[:, 0, :, :]
    labels_one_hot[:, 1, :, :] = rgb_labels[:, 1, :, :]
    labels_one_hot[:, 2, :, :] = rgb_labels[:, 2, :, :]
    labels_one_hot[:, 3, :, :] = (1 - rgb_labels.sum(dim=1)).clamp(0, 1)  # Background class
    return labels_one_hot


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        signal_class: int = SIGNAL_CLASS,
        union_exponent: int = 2,
        gamma: float = 2.0,
        smooth: float = 1e-3,
    ):
        """
        Combined Soft Dice + Focal Loss for multi-class classification.

        Args:
            alpha (float): Weight multiplier for the signal_class in Dice loss.
            signal_class (int): Index of the class to apply extra weight to in Dice loss.
            union_exponent (int): Exponent for the union term in Dice loss (1 or 2).
            gamma (float): Focusing parameter for Focal Loss.
            smooth (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.signal_class = signal_class
        self.union_exponent = union_exponent
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (N, C, ...) with raw, unnormalized scores for each class.
            target: Tensor of shape (N, ...) with class indices (0 <= target < C).

        Returns:
            Combined loss scalar.
        """
        target_one_hot = rgb_to_one_hot(target_rgb)  # Convert RGB labels to one-hot encoding
        probs = F.softmax(logits, dim=1)  # (N, C, ...)

        dims = tuple(range(2, logits.dim()))  # spatial dims to sum over

        intersection = torch.sum(probs * target_one_hot, dim=dims)  # (N, C)
        if self.union_exponent == 1:
            union = torch.sum(probs + target_one_hot, dim=dims)  # (N, C)
        elif self.union_exponent == 2:
            union = torch.sum(probs**2 + target_one_hot**2, dim=dims)  # (N, C)
        else:
            raise ValueError("union_exponent must be 1 or 2")

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        weights = torch.ones_like(dice_score)
        weights[:, self.signal_class] = self.alpha

        dice_loss = 1 - dice_score
        weighted_dice_loss = (dice_loss * weights).mean()

        pt = torch.sum(probs * target_one_hot, dim=1)  # (N, ...)
        focal_loss = -((1 - pt) ** self.gamma) * torch.log(pt + self.smooth)
        focal_loss = focal_loss.mean()

        loss: torch.Tensor = weighted_dice_loss + focal_loss
        return loss