#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROF 模型单元测试

测试内容:
    - 各算法的基本功能
    - 数值稳定性
    - 收敛性
    - 边界情况

运行方式:
    python -m pytest tests/test_rof.py -v
    或: python tests/test_rof.py
"""

import unittest
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    chambolle_rof,
    split_bregman_rof,
    gradient_descent_rof,
    rof_energy,
    compute_gradient,
    compute_divergence,
    create_synthetic_image,
    add_noise,
    psnr
)


class TestROFBasic(unittest.TestCase):
    """ROF 基础功能测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.image = create_synthetic_image((64, 64), 'checkerboard')
        self.noisy = add_noise(self.image, 'gaussian', sigma=0.1)
        
    def test_chambolle_basic(self):
        """测试 Chambolle 方法基本功能"""
        denoised, p = chambolle_rof(self.noisy, lambda_param=0.5, max_iter=50)
        
        # 检查输出形状
        self.assertEqual(denoised.shape, self.noisy.shape)
        self.assertEqual(p.shape, (*self.noisy.shape, 2))
        
        # 检查输出范围
        self.assertTrue(np.all(denoised >= 0))
        self.assertTrue(np.all(denoised <= 1))
        
        # 检查去噪效果（PSNR 应该提高）
        psnr_noisy = psnr(self.image, self.noisy)
        psnr_denoised = psnr(self.image, denoised)
        self.assertGreater(psnr_denoised, psnr_noisy)
        
    def test_split_bregman_basic(self):
        """测试 Split Bregman 方法基本功能"""
        denoised, history = split_bregman_rof(
            self.noisy, lambda_param=0.5, max_iter=10
        )
        
        # 检查输出形状
        self.assertEqual(denoised.shape, self.noisy.shape)
        
        # 检查输出范围
        self.assertTrue(np.all(denoised >= 0))
        self.assertTrue(np.all(denoised <= 1))
        
        # 检查能量历史
        self.assertEqual(len(history), 10)
        
        # 能量应该递减或稳定
        for i in range(1, len(history)):
            self.assertLessEqual(history[i], history[i-1] + 1e-6)
            
    def test_gradient_descent_basic(self):
        """测试梯度下降方法基本功能"""
        denoised, history = gradient_descent_rof(
            self.noisy, lambda_param=0.5, max_iter=100
        )
        
        # 检查输出形状
        self.assertEqual(denoised.shape, self.noisy.shape)
        
        # 检查输出范围
        self.assertTrue(np.all(denoised >= -0.1))  # 允许小的负值
        self.assertTrue(np.all(denoised <= 1.1))   # 允许小的超值
        
    def test_energy_computation(self):
        """测试能量函数计算"""
        energy = rof_energy(self.image, self.noisy, lambda_param=1.0)
        
        # 能量应该是正数
        self.assertGreater(energy, 0)
        
        # 相同图像的能量应该较小
        energy_same = rof_energy(self.image, self.image, lambda_param=1.0)
        energy_diff = rof_energy(self.noisy, self.image, lambda_param=1.0)
        self.assertLess(energy_same, energy_diff)


class TestROFParameters(unittest.TestCase):
    """ROF 参数测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.image = create_synthetic_image((64, 64), 'step')
        self.noisy = add_noise(self.image, 'gaussian', sigma=0.1)
        
    def test_lambda_parameter(self):
        """测试 lambda 参数的影响"""
        # lambda 较小: 更强的平滑
        denoised_small, _ = chambolle_rof(
            self.noisy, lambda_param=0.1, max_iter=30
        )
        
        # lambda 较大: 更接近原图
        denoised_large, _ = chambolle_rof(
            self.noisy, lambda_param=5.0, max_iter=30
        )
        
        # 大 lambda 应该更接近噪声图像（保留更多细节）
        diff_small = np.mean((denoised_small - self.noisy)**2)
        diff_large = np.mean((denoised_large - self.noisy)**2)
        
        # 大 lambda 的差异应该更小
        self.assertLess(diff_large, diff_small)
        
    def test_different_noise_levels(self):
        """测试不同噪声水平"""
        for sigma in [0.05, 0.1, 0.2]:
            noisy = add_noise(self.image, 'gaussian', sigma=sigma)
            denoised, _ = chambolle_rof(noisy, lambda_param=0.5, max_iter=30)
            
            # 去噪后的图像应该在合理范围内
            self.assertTrue(np.all(np.isfinite(denoised)))
            
    def test_convergence_tolerance(self):
        """测试收敛容差"""
        # 宽松容差应该收敛更快
        _, history_loose = split_bregman_rof(
            self.noisy, lambda_param=0.5, max_iter=100, tol=1e-2
        )
        
        # 严格容差可能需要更多迭代
        _, history_strict = split_bregman_rof(
            self.noisy, lambda_param=0.5, max_iter=100, tol=1e-8
        )
        
        # 两个都应该产生结果
        self.assertGreater(len(history_loose), 0)
        self.assertGreater(len(history_strict), 0)


class TestROFEdgeCases(unittest.TestCase):
    """ROF 边界情况测试"""
    
    def test_constant_image(self):
        """测试常数图像"""
        constant = np.ones((32, 32)) * 0.5
        denoised, _ = chambolle_rof(constant, lambda_param=0.5, max_iter=10)
        
        # 常数图像去噪后应该基本保持不变
        np.testing.assert_allclose(denoised, constant, atol=0.01)
        
    def test_small_image(self):
        """测试小图像"""
        small = np.random.rand(8, 8)
        denoised, _ = chambolle_rof(small, lambda_param=0.5, max_iter=10)
        
        self.assertEqual(denoised.shape, (8, 8))
        
    def test_single_channel(self):
        """测试单通道处理"""
        # 确保正确处理灰度图像
        gray = np.random.rand(64, 64)
        denoised, _ = chambolle_rof(gray, lambda_param=0.5, max_iter=10)
        
        # 输出应该是二维
        self.assertEqual(denoised.ndim, 2)
        
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 创建接近零的图像
        small_values = np.random.rand(32, 32) * 1e-6
        denoised, _ = chambolle_rof(small_values, lambda_param=0.5, max_iter=10)
        
        # 结果应该是有限的
        self.assertTrue(np.all(np.isfinite(denoised)))


class TestGradientDivergence(unittest.TestCase):
    """梯度和散度算子测试"""
    
    def test_gradient_shape(self):
        """测试梯度输出形状"""
        u = np.random.rand(32, 32)
        grad_x, grad_y = compute_gradient(u)
        
        self.assertEqual(grad_x.shape, u.shape)
        self.assertEqual(grad_y.shape, u.shape)
        
    def test_divergence_shape(self):
        """测试散度输出形状"""
        px = np.random.rand(32, 32)
        py = np.random.rand(32, 32)
        div = compute_divergence(px, py)
        
        self.assertEqual(div.shape, px.shape)
        
    def test_gradient_constant(self):
        """测试常数函数的梯度"""
        constant = np.ones((32, 32))
        grad_x, grad_y = compute_gradient(constant)
        
        # 常数函数的梯度应该接近零
        np.testing.assert_allclose(grad_x, 0, atol=1e-10)
        np.testing.assert_allclose(grad_y, 0, atol=1e-10)
        
    def test_gradient_linear(self):
        """测试线性函数的梯度"""
        # f(x,y) = x
        x = np.linspace(0, 1, 32).reshape(1, -1)
        linear = np.repeat(x, 32, axis=0)
        grad_x, grad_y = compute_gradient(linear)
        
        # x 方向梯度应该约为 1
        # 注意边界效应，只检查内部
        self.assertTrue(np.mean(grad_x[5:-5, 5:-5]) > 0.5)
        
    def test_div_grad_identity(self):
        """测试 div(grad(u)) ≈ Laplacian(u)"""
        u = np.random.rand(32, 32)
        
        grad_x, grad_y = compute_gradient(u)
        div_grad = compute_divergence(grad_x, grad_y)
        
        # 数值计算拉普拉斯
        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1] = (
            u[:-2, 1:-1] + u[2:, 1:-1] + 
            u[1:-1, :-2] + u[1:-1, 2:] - 
            4 * u[1:-1, 1:-1]
        )
        
        # 应该在数值误差范围内相似
        np.testing.assert_allclose(
            div_grad[1:-1, 1:-1], 
            laplacian[1:-1, 1:-1], 
            atol=1e-10
        )


class TestROFConsistency(unittest.TestCase):
    """ROF 一致性测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.image = create_synthetic_image((48, 48), 'circles')
        self.noisy = add_noise(self.image, 'gaussian', sigma=0.1)
        
    def test_deterministic(self):
        """测试结果的确定性"""
        # 固定随机种子，两次运行应该产生相同结果
        np.random.seed(42)
        denoised1, _ = chambolle_rof(self.noisy, lambda_param=0.5, max_iter=20)
        
        np.random.seed(42)
        denoised2, _ = chambolle_rof(self.noisy, lambda_param=0.5, max_iter=20)
        
        np.testing.assert_allclose(denoised1, denoised2)
        
    def test_method_consistency(self):
        """测试不同方法之间的一致性"""
        # 不同方法应该产生相似但不完全相同的结果
        denoised_cham, _ = chambolle_rof(
            self.noisy, lambda_param=0.5, max_iter=100
        )
        denoised_sb, _ = split_bregman_rof(
            self.noisy, lambda_param=0.5, max_iter=20
        )
        
        # 计算相对差异
        rel_diff = np.mean(np.abs(denoised_cham - denoised_sb))
        
        # 差异应该在合理范围内（不同方法的解可能有微小差异）
        self.assertLess(rel_diff, 0.1)
        
    def test_energy_decrease(self):
        """测试能量递减"""
        _, history = gradient_descent_rof(
            self.noisy, lambda_param=0.5, max_iter=50
        )
        
        # 能量应该总体递减（允许小的数值误差）
        for i in range(1, len(history)):
            self.assertLessEqual(history[i], history[i-1] + 1e-5)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("=" * 70)
    print("ROF 模型单元测试")
    print("=" * 70)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestROFBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestROFParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestROFEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientDivergence))
    suite.addTests(loader.loadTestsFromTestCase(TestROFConsistency))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)
    
    # 返回退出码
    if result.wasSuccessful():
        print("\n所有测试通过！")
        exit(0)
    else:
        print("\n有测试失败或出错！")
        exit(1)
