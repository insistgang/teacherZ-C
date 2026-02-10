"""
工具函数模块

本模块提供图像处理中常用的工具函数，包括：
- 噪声添加（高斯噪声、椒盐噪声等）
- 图像质量评估（PSNR、SSIM）
- 图像加载和保存
- 图像预处理函数
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage


def add_noise(image, noise_type='gaussian', **kwargs):
    """
    为图像添加噪声
    
    支持多种噪声类型，便于测试去噪算法。
    
    参数:
    -----------
    image : ndarray
        输入图像，像素值范围 [0, 1]
    noise_type : str
        噪声类型：'gaussian'(高斯), 'salt_pepper'(椒盐), 'poisson'(泊松)
    **kwargs : dict
        额外的噪声参数
        - gaussian: sigma (标准差，默认 0.1)
        - salt_pepper: amount (噪声比例，默认 0.05)
        - poisson: 无额外参数
        
    返回:
    -----------
    noisy_image : ndarray
        添加噪声后的图像，像素值裁剪到 [0, 1]
        
    示例:
    -----------
    >>> image = np.random.rand(100, 100)
    >>> # 添加高斯噪声
    >>> noisy_gauss = add_noise(image, 'gaussian', sigma=0.1)
    >>> # 添加椒盐噪声
    >>> noisy_sp = add_noise(image, 'salt_pepper', amount=0.05)
    """
    image = np.asarray(image, dtype=np.float64)
    
    if noise_type == 'gaussian':
        return gaussian_noise(image, **kwargs)
    elif noise_type == 'salt_pepper':
        return salt_pepper_noise(image, **kwargs)
    elif noise_type == 'poisson':
        return poisson_noise(image, **kwargs)
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")


def gaussian_noise(image, sigma=0.1):
    """
    添加高斯噪声
    
    模型: f_noisy = f + n, 其中 n ~ N(0, σ²)
    
    参数:
    -----------
    image : ndarray
        输入图像 [0, 1]
    sigma : float
        噪声标准差
        
    返回:
    -----------
    noisy : ndarray
        添加高斯噪声后的图像
    """
    image = np.asarray(image, dtype=np.float64)
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)


def salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """
    添加椒盐噪声
    
    随机将像素设置为 0（椒）或 1（盐）。
    
    参数:
    -----------
    image : ndarray
        输入图像 [0, 1]
    amount : float
        噪声像素占总像素的比例 [0, 1]
    salt_vs_pepper : float
        盐噪声（白点）与椒噪声（黑点）的比例
        0.5 表示盐和椒各占一半
        
    返回:
    -----------
    noisy : ndarray
        添加椒盐噪声后的图像
    """
    image = np.asarray(image, dtype=np.float64)
    noisy = image.copy()
    
    # 计算盐和椒的数量
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper)))
    
    # 添加盐噪声（白点，值为 1）
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 1.0
    
    # 添加椒噪声（黑点，值为 0）
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0.0
    
    return noisy


def poisson_noise(image):
    """
    添加泊松噪声
    
    模拟光子计数噪声，服从泊松分布。
    适用于模拟低光照条件下的图像噪声。
    
    参数:
    -----------
    image : ndarray
        输入图像 [0, 1]
        
    返回:
    -----------
    noisy : ndarray
        添加泊松噪声后的图像
    """
    image = np.asarray(image, dtype=np.float64)
    
    # 泊松噪声：像素值的期望值是原图像值（缩放到合适范围）
    # 假设最大光子数为 255（8位图像）
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    
    return np.clip(noisy, 0, 1)


def psnr(original, denoised, max_val=1.0):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    
    PSNR = 10 * log10(MAX² / MSE)
    
    其中：
    - MAX 是图像的最大可能像素值
    - MSE 是均方误差
    
    PSNR 越高表示图像质量越好。
    - PSNR > 40dB: 人眼难以察觉差异
    - PSNR 30-40dB: 质量较好
    - PSNR 20-30dB: 质量中等，有可见失真
    - PSNR < 20dB: 质量较差
    
    参数:
    -----------
    original : ndarray
        原始图像
    denoised : ndarray
        去噪后的图像
    max_val : float
        图像的最大像素值（归一化图像为 1.0，8位图像为 255）
        
    返回:
    -----------
    psnr_value : float
        PSNR 值（dB）
        
    示例:
    -----------
    >>> original = np.random.rand(100, 100)
    >>> denoised = original + 0.01 * np.random.randn(100, 100)
    >>> print(f"PSNR: {psnr(original, denoised):.2f} dB")
    """
    original = np.asarray(original, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    
    # 计算均方误差
    mse = np.mean((original - denoised) ** 2)
    
    if mse == 0:
        return float('inf')  # 两图像完全相同
    
    # 计算 PSNR
    psnr_value = 10 * np.log10((max_val ** 2) / mse)
    
    return psnr_value


def ssim(original, denoised, window_size=11, K1=0.01, K2=0.03, max_val=1.0):
    """
    计算结构相似性指数 (Structural Similarity Index)
    
    SSIM 比 PSNR 更符合人眼视觉感知，考虑亮度、对比度和结构三个因素。
    
    SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
    
    其中：
    - l(x,y): 亮度比较
    - c(x,y): 对比度比较
    - s(x,y): 结构比较
    
    SSIM 取值范围 [-1, 1]，通常 [0, 1]:
    - SSIM = 1: 两图像完全相同
    - SSIM > 0.9: 非常相似
    - SSIM 0.8-0.9: 相似
    - SSIM < 0.8: 差异较明显
    
    参数:
    -----------
    original : ndarray
        原始图像
    denoised : ndarray
        去噪后的图像
    window_size : int
        高斯滑动窗口大小，必须为奇数
    K1, K2 : float
        稳定性常数
    max_val : float
        图像最大像素值
        
    返回:
    -----------
    ssim_value : float
        SSIM 值
        
    参考文献:
    -----------
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
    Image quality assessment: from error visibility to structural similarity. 
    IEEE Transactions on Image Processing, 13(4), 600-612.
    """
    original = np.asarray(original, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    
    # 创建高斯窗口
    sigma = 1.5
    gaussian = np.arange(-(window_size//2), window_size//2 + 1)
    gaussian = np.exp(-(gaussian**2) / (2 * sigma**2))
    gaussian = np.outer(gaussian, gaussian)
    window = gaussian / gaussian.sum()
    
    # 常数
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    
    # 计算局部均值
    mu1 = ndimage.convolve(original, window, mode='reflect')
    mu2 = ndimage.convolve(denoised, window, mode='reflect')
    
    # 计算均值平方和乘积
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = ndimage.convolve(original ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = ndimage.convolve(denoised ** 2, window, mode='reflect') - mu2_sq
    sigma12 = ndimage.convolve(original * denoised, window, mode='reflect') - mu1_mu2
    
    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def load_image(path, gray=True, resize=None):
    """
    加载图像文件
    
    支持多种格式：PNG, JPEG, BMP, TIFF 等。
    
    参数:
    -----------
    path : str
        图像文件路径
    gray : bool
        是否转换为灰度图像，默认 True
    resize : tuple, optional
        调整大小 (width, height)
        
    返回:
    -----------
    image : ndarray
        加载的图像，像素值范围 [0, 1]
        
    示例:
    -----------
    >>> # 加载灰度图像
    >>> img = load_image('lena.png', gray=True)
    >>> # 加载彩色图像
    >>> img = load_image('photo.jpg', gray=False)
    """
    try:
        img = Image.open(path)
        
        if gray:
            img = img.convert('L')  # 转换为灰度
        else:
            img = img.convert('RGB')  # 转换为 RGB
        
        if resize is not None:
            img = img.resize(resize, Image.Resampling.LANCZOS)
        
        # 转换为 numpy 数组并归一化到 [0, 1]
        image = np.array(img, dtype=np.float64) / 255.0
        
        return image
    except Exception as e:
        raise IOError(f"无法加载图像 {path}: {e}")


def save_image(image, path, quality=95):
    """
    保存图像到文件
    
    参数:
    -----------
    image : ndarray
        要保存的图像 [0, 1]
    path : str
        保存路径
    quality : int
        JPEG 质量 (1-95)，仅对 JPEG 有效
    """
    image = np.asarray(image)
    
    # 裁剪到有效范围
    image = np.clip(image, 0, 1)
    
    # 转换为 8-bit
    if image.ndim == 2:
        # 灰度图像
        img = Image.fromarray((image * 255).astype(np.uint8), mode='L')
    else:
        # 彩色图像
        img = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
    
    # 保存
    if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        img.save(path, quality=quality)
    else:
        img.save(path)


def normalize_image(image, method='minmax'):
    """
    归一化图像
    
    参数:
    -----------
    image : ndarray
        输入图像
    method : str
        归一化方法：
        - 'minmax': 线性缩放到 [0, 1]
        - 'zscore': 零均值单位方差
        
    返回:
    -----------
    normalized : ndarray
        归一化后的图像
    """
    image = np.asarray(image, dtype=np.float64)
    
    if method == 'minmax':
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image)
    elif method == 'zscore':
        mean = image.mean()
        std = image.std()
        if std > 0:
            return (image - mean) / std
        else:
            return image - mean
    else:
        raise ValueError(f"未知的归一化方法: {method}")


def create_synthetic_image(shape=(256, 256), pattern='checkerboard'):
    """
    创建合成测试图像
    
    参数:
    -----------
    shape : tuple
        图像尺寸 (height, width)
    pattern : str
        图案类型：'checkerboard'(棋盘), 'circles'(同心圆), 
                'gradient'(渐变), 'step'(阶梯)
                
    返回:
    -----------
    image : ndarray
        合成的测试图像
    """
    h, w = shape
    
    if pattern == 'checkerboard':
        # 棋盘图案
        block_size = 32
        image = np.zeros((h, w))
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    image[i:i+block_size, j:j+block_size] = 1.0
    
    elif pattern == 'circles':
        # 同心圆
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        image = (np.sin(dist / 10) + 1) / 2
    
    elif pattern == 'gradient':
        # 线性渐变
        image = np.linspace(0, 1, w).reshape(1, -1).repeat(h, axis=0)
    
    elif pattern == 'step':
        # 阶梯函数
        image = np.zeros((h, w))
        image[:, :w//4] = 0.1
        image[:, w//4:w//2] = 0.4
        image[:, w//2:3*w//4] = 0.7
        image[:, 3*w//4:] = 0.9
    
    else:
        raise ValueError(f"未知的图案类型: {pattern}")
    
    return image


if __name__ == "__main__":
    # 测试工具函数
    print("测试噪声添加...")
    img = create_synthetic_image((100, 100), 'checkerboard')
    
    noisy_g = add_noise(img, 'gaussian', sigma=0.1)
    noisy_sp = add_noise(img, 'salt_pepper', amount=0.05)
    
    print(f"原图范围: [{img.min():.3f}, {img.max():.3f}]")
    print(f"高斯噪声范围: [{noisy_g.min():.3f}, {noisy_g.max():.3f}]")
    print(f"椒盐噪声范围: [{noisy_sp.min():.3f}, {noisy_sp.max():.3f}]")
    
    print("\n测试质量评估...")
    psnr_val = psnr(img, noisy_g)
    ssim_val = ssim(img, noisy_g)
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
