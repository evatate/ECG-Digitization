from typing import Tuple

import numpy as np
import torch


class PixelSizeFinder(torch.nn.Module):
    def __init__(
        self,
        mm_between_grid_lines: int = 5,
        samples: int = 1000,
        min_number_of_grid_lines: int = 15,
        max_number_of_grid_lines: int = 120,
        max_zoom: int = 10,
        zoom_factor: float = 10.0,
        lower_grid_line_factor: float = 0.5,
    ):
        """
        Finds the mm/pixel in x and y direction from an image with a grid and a similarily colored background.
        The grid needs to be orthogonal to the rows and columns. For an axis, the algorithm tries <samples> different
        pixels between grid lines in the interval
        [axis_shape / <max_number_of_grid_lines>, axis_shape / <min_number_of_grid_lines>]. The optimal pixel size is
        selected and the search space is reduced to 1 / <zoom_factor> of the current interval around the current
        optimum. This is repeated <max_zoom> times or until convergence. When selecting the optimum smaller lines
        between the grid lines are weighted <lower_grid_line_factor>.

        Args:
            mm_between_grid_lines (int, optional): The number of mm between grid lines, assumed to be equal in horizontal and vertical direction.
            samples (int, optional): Number of samples of pixels between grid lines at each zoom level.
            min_number_of_grid_lines (int, optional): Minimum number of grid lines for an axis in the image.
            max_number_of_grid_lines (int, optional): Maximum number of grid lines for an axis in the image.
            max_zoom (int, optional): The maximum number of times to zoom in on the search space.
            zoom_factor (float, optional): The reduction factor of the search space for each zoom.
            lower_grid_line_factor (float, optional): The weighing of the smaller grid lines.
        """

        super(PixelSizeFinder, self).__init__()
        self.mm_between_grid_lines = mm_between_grid_lines
        self.samples = samples
        self.min_number_of_grid_lines = min_number_of_grid_lines
        self.max_number_of_grid_lines = max_number_of_grid_lines
        self.max_zoom = max_zoom
        self.zoom_factor = zoom_factor
        self.lower_grid_line_factor = lower_grid_line_factor

    def forward(self, image: torch.Tensor) -> Tuple[float, float]:
        """
        Finds mm/pixel from an image with a grid and an similarily colored background. The grid needs to be orthogonal
        to the rows and columns.

        Args:
            image (torch.Tensor): Image with a grid that is orthogonal to the rows and columns
            name (str): Name of the image to save.

        Returns:
            Tuple[float, float]: mm/pixel in x and y direction, respectively.
        """
        image = image.squeeze()
        assert image.ndim == 2, "Image must be 2D (H, W) tensor."

        estimate_x = self._mm_per_pixel_y(image.swapaxes(0, 1))
        aspect = image.shape[1] / image.shape[0]
        estimate_y = self._mm_per_pixel_y(image, aspect)

        return estimate_x, estimate_y

    def _mm_per_pixel_y(self, image: torch.Tensor, aspect: float = 1) -> float:
        pxls_between_lines = self._find_pxls_between_horizontal_grid_lines(image, aspect)

        return self.mm_between_grid_lines / pxls_between_lines

    def _find_pxls_between_horizontal_grid_lines(self, image: torch.Tensor, aspect: float) -> float:
        col_sum = image.sum(dim=-1)
        col_sum = col_sum - col_sum.mean()
        autocorrelation = np.correlate(col_sum.cpu().numpy(), col_sum.cpu().numpy(), mode="full")

        pxls_between_lines = self._zoom_grid_search_min_distance(torch.tensor(autocorrelation), aspect)

        return pxls_between_lines

    def _zoom_grid_search_min_distance(self, autocorrelation: torch.Tensor, aspect: float) -> float:
        """
        Finds the best pixels between lines in the autocorrelation tensor by doing a grid search over potential pixels
        distances and zooming in on the current optimum until convergence.

        Args:
            autocorrelation (torch.Tensor): autocorrelation tensor

        Returns:
            Tuple[float, Dict[str, Any]]: best pixels between lines
        """
        autocorrelation = autocorrelation.clone()

        expected_min_grid_lines = self.min_number_of_grid_lines / aspect
        expected_max_grid_lines = self.max_number_of_grid_lines / aspect
        autocorrelation = autocorrelation[: autocorrelation.shape[0] // 2].flip(0)
        autocorrelation = autocorrelation - autocorrelation.mean()

        min_dist_grid_lines = autocorrelation.shape[-1] / expected_max_grid_lines
        max_dist_grid_lines = autocorrelation.shape[-1] / expected_min_grid_lines

        pxls_between_lines = self._grid_search_min_distance(
            autocorrelation, self.samples, min_dist_grid_lines, max_dist_grid_lines
        )

        for _ in range(self.max_zoom):
            w = (max_dist_grid_lines - min_dist_grid_lines) / self.zoom_factor
            min_dist_grid_lines = max(pxls_between_lines - w, min_dist_grid_lines)
            max_dist_grid_lines = min(pxls_between_lines + w, max_dist_grid_lines)
            curr_pxls_between_lines = self._grid_search_min_distance(
                autocorrelation, self.samples, min_dist_grid_lines, max_dist_grid_lines
            )

            pxls_between_lines = curr_pxls_between_lines

        return curr_pxls_between_lines

    def _grid_search_min_distance(
        self, autocorrelation: torch.Tensor, samples: int, min_dist_grid_lines: float, max_dist_grid_lines: float
    ) -> float:
        """Grid searches over the pixels between peaks in the interval [min_dist_grid_lines, max_dist_grid_lines] in the autocorrelation tensor.

        Args:
            autocorrelation (torch.Tensor): autocorrelation tensor
            samples (int): number of samples in the grid search
            min_dist_grid_lines (float): minimum pixels between grid lines
            max_dist_grid_lines (float): maximum pixels between grid lines

        Returns:
            float (float): best pixels between peaks
        """

        pixels_between_grid_lines = torch.linspace(min_dist_grid_lines, max_dist_grid_lines, samples)

        best_grid_discretization_score = torch.tensor(-torch.inf)
        best_pixels_between_grid_line = 0

        for pixels_between_grid_line in pixels_between_grid_lines:
            calculated_grid_lines = torch.zeros_like(autocorrelation)
            weighting = np.linspace(1, 0, len(autocorrelation))

            indices = (
                (torch.arange(0, autocorrelation.shape[-1], pixels_between_grid_line / 5, dtype=torch.float32))
                .round()
                .to(torch.int64)
            )
            indices = indices[indices < autocorrelation.shape[-1]]

            calculated_grid_lines[indices] = self.lower_grid_line_factor
            calculated_grid_lines[indices[::5]] = 1
            calculated_grid_lines = calculated_grid_lines * torch.tensor(weighting, dtype=torch.float32)

            grid_discretization_score = (autocorrelation * calculated_grid_lines).sum()

            if best_grid_discretization_score < grid_discretization_score:
                best_grid_discretization_score = grid_discretization_score
                best_pixels_between_grid_line = pixels_between_grid_line

        return best_pixels_between_grid_line