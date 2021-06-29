# Standard modules
from typing import Optional

# Third party modules
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class Metrics:
    def __init__(self, gt: np.darray, pred: np.ndarray):
        self.gt = gt
        self.pred = pred

    def mse(self):
        """Compute Mean Squared Error (MSE)"""
        return np.mean((self.gt - self.pred) ** 2)

    def nmse(self):
        """Compute Normalized Mean Squared Error (NMSE)"""
        return np.linalg.norm(self.gt - self.pred) ** 2 / np.linalg.norm(self.gt) ** 2

    def psnr(self):
        """Compute Peak Signal to Noise Ratio metric (PSNR)"""
        return peak_signal_noise_ratio(self.gt, self.pred, data_range=self.gt.max())

    def ssim(self, maxval: Optional[float] = None):
        """Compute Structural Similarity Index Metric (SSIM)"""
        if not self.gt.ndim == 3:
            raise ValueError("Unexpected number of dimensions in ground truth.")
        if not self.gt.ndim == self.pred.ndim:
            raise ValueError("Ground truth dimensions does not match pred.")

        maxval = self.gt.max() if maxval is None else maxval

        ssim = 0
        for slice_num in range(self.gt.shape[0]):
            ssim = ssim + structural_similarity(
                self.gt[slice_num], self.pred[slice_num], data_range=maxval
            )

        return ssim / self.gt.shape[0]
