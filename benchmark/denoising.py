import numpy as np
from benchmark import BaseBenchmark
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def gradient_2d(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.diff(image, axis=1, prepend=image[:, :1])
    gy = np.diff(image, axis=0, prepend=image[:1, :])
    return gx, gy


def divergence_2d(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    div_x = np.diff(gx, axis=1, append=gx[:, -1:])
    div_y = np.diff(gy, axis=0, append=gy[-1:, :])
    return div_x + div_y


def tv_norm(image: np.ndarray) -> float:
    gx, gy = gradient_2d(image)
    return np.sum(np.sqrt(gx**2 + gy**2 + 1e-10))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


class ROFDenoising(BaseBenchmark):
    def __init__(self, method: str = "gradient_descent"):
        super().__init__(name=f"ROF_{method}", category="图像去噪")
        self.method = method
        self.image = None
        self.noisy = None
        self.result = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)
        noise_std = kwargs.get("noise_std", 0.1)
        self.noisy = self.image + noise_std * np.random.randn(*size)
        self.noisy = np.clip(self.noisy, 0, 1)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        lam = kwargs.get("lambda", 0.1)
        max_iter = kwargs.get("max_iter", 100)

        if self.method == "gradient_descent":
            self.result = self._gradient_descent(self.noisy, lam, max_iter)
        elif self.method == "chambolle":
            self.result = self._chambolle(self.noisy, lam, max_iter)
        elif self.method == "primal_dual":
            self.result = self._primal_dual(self.noisy, lam, max_iter)
        else:
            self.result = self.noisy.copy()

        accuracy = psnr(self.image, self.result)
        return self.result, accuracy

    def _gradient_descent(self, f: np.ndarray, lam: float, max_iter: int) -> np.ndarray:
        u = f.copy()
        dt = 0.1

        for _ in range(max_iter):
            gx, gy = gradient_2d(u)
            norm = np.sqrt(gx**2 + gy**2 + 1e-10)
            gx /= norm
            gy /= norm
            div = divergence_2d(gx, gy)
            u = u + dt * (div + lam * (f - u))

        return np.clip(u, 0, 1)

    def _chambolle(self, f: np.ndarray, lam: float, max_iter: int) -> np.ndarray:
        px = np.zeros_like(f)
        py = np.zeros_like(f)
        tau = 0.25

        for _ in range(max_iter):
            div_p = divergence_2d(px, py)
            u = f - lam * div_p
            gx, gy = gradient_2d(u)

            px_new = (px + tau * gx) / (1 + tau * np.sqrt(gx**2 + gy**2 + 1e-10))
            py_new = (py + tau * gy) / (1 + tau * np.sqrt(gx**2 + gy**2 + 1e-10))

            px, py = px_new, py_new

        return np.clip(f - lam * divergence_2d(px, py), 0, 1)

    def _primal_dual(self, f: np.ndarray, lam: float, max_iter: int) -> np.ndarray:
        u = f.copy()
        ubar = f.copy()
        px = np.zeros_like(f)
        py = np.zeros_like(f)
        tau = 0.1
        sigma = 0.1

        for _ in range(max_iter):
            gx, gy = gradient_2d(ubar)
            px = px + sigma * gx
            py = py + sigma * gy

            norm = np.sqrt(px**2 + py**2 + 1e-10)
            factor = np.maximum(1, norm / lam)
            px /= factor
            py /= factor

            u_new = (u + tau * lam * f + tau * divergence_2d(px, py)) / (1 + tau * lam)
            ubar = 2 * u_new - u
            u = u_new

        return np.clip(u, 0, 1)

    def cleanup(self):
        self.image = None
        self.noisy = None
        self.result = None


class BM3DDenoising(BaseBenchmark):
    def __init__(self):
        super().__init__(name="BM3D", category="图像去噪")
        self.image = None
        self.noisy = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)
        noise_std = kwargs.get("noise_std", 0.1)
        self.noisy = self.image + noise_std * np.random.randn(*size)
        self.noisy = np.clip(self.noisy, 0, 1)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        block_size = kwargs.get("block_size", 8)
        sigma = kwargs.get("sigma", 0.1)

        result = self._simplified_bm3d(self.noisy, block_size, sigma)
        accuracy = psnr(self.image, result)
        return result, accuracy

    def _simplified_bm3d(
        self, image: np.ndarray, block_size: int, sigma: float
    ) -> np.ndarray:
        h, w = image.shape
        result = np.zeros_like(image)
        count = np.zeros_like(image)

        thresh = 2.0 * sigma

        for i in range(0, h - block_size + 1, block_size // 2):
            for j in range(0, w - block_size + 1, block_size // 2):
                block = image[i : i + block_size, j : j + block_size].copy()

                dct_block = self._dct2d(block)
                dct_block[np.abs(dct_block) < thresh] = 0
                denoised = self._idct2d(dct_block)

                result[i : i + block_size, j : j + block_size] += denoised
                count[i : i + block_size, j : j + block_size] += 1

        count[count == 0] = 1
        return result / count

    def _dct2d(self, block: np.ndarray) -> np.ndarray:
        from scipy.fftpack import dct

        return dct(dct(block.T, norm="ortho").T, norm="ortho")

    def _idct2d(self, block: np.ndarray) -> np.ndarray:
        from scipy.fftpack import idct

        return idct(idct(block.T, norm="ortho").T, norm="ortho")

    def cleanup(self):
        self.image = None
        self.noisy = None


class DnCNNBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="DnCNN", category="图像去噪")
        self.image = None
        self.noisy = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)
        noise_std = kwargs.get("noise_std", 0.1)
        self.noisy = self.image + noise_std * np.random.randn(*size)
        self.noisy = np.clip(self.noisy, 0, 1)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        num_layers = kwargs.get("num_layers", 5)

        result = self._simplified_dncnn(self.noisy, num_layers)
        accuracy = psnr(self.image, result)
        return result, accuracy

    def _simplified_dncnn(self, image: np.ndarray, num_layers: int) -> np.ndarray:
        result = image.copy()

        for _ in range(num_layers):
            padded = np.pad(result, 1, mode="reflect")
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i, j] = np.sum(padded[i : i + 3, j : j + 3] * kernel)

        noise = image - result
        denoised = image - 0.5 * noise
        return np.clip(denoised, 0, 1)

    def cleanup(self):
        self.image = None
        self.noisy = None
