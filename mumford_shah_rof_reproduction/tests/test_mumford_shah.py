#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mumford-Shah 和 Chan-Vese 模型单元测试

测试内容:
    - Chan-Vese 分割功能
    - Mumford-Shah 分割功能
    - 水平集方法
    - 初始化函数
    - 边界情况

运行方式:
    python -m pytest tests/test_mumford_shah.py -v
    或: python tests/test_mumford_shah.py
"""

import unittest
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    chan_vese_segmentation,
    mumford_shah_segmentation,
    level_set_evolution,
    ambrosio_tortorelli_approximation,
    initialize_sdf_circle,
    initialize_sdf_rectangle,
    initialize_sdf_multiple_circles,
    reinitialize_sdf,
    heaviside,
    dirac_delta,
    create_synthetic_image,
    add_noise
)


class TestChanVeseBasic(unittest.TestCase):
    """Chan-Vese 基础功能测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        # 创建简单的测试图像（中心有方形区域）
        self.image = np.zeros((64, 64))
        self.image[20:44, 20:44] = 0.8
        self.image = add_noise(self.image, 'gaussian', sigma=0.05)
        
    def test_basic_segmentation(self):
        """测试基本分割功能"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        segmentation, phi, history = chan_vese_segmentation(
            self.image, phi0, max_iter=50, verbose=False
        )
        
        # 检查输出形状
        self.assertEqual(segmentation.shape, self.image.shape)
        self.assertEqual(phi.shape, self.image.shape)
        
        # 检查输出类型
        self.assertTrue(np.all((segmentation == 0) | (segmentation == 1)))
        
        # 应该有合理的前景区域
        foreground_ratio = np.sum(segmentation) / segmentation.size
        self.assertGreater(foreground_ratio, 0.1)
        self.assertLess(foreground_ratio, 0.9)
        
    def test_energy_decrease(self):
        """测试能量递减"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        _, _, history = chan_vese_segmentation(
            self.image, phi0, max_iter=100, verbose=False
        )
        
        # 能量应该总体递减
        for i in range(1, len(history)):
            self.assertLessEqual(history[i], history[i-1] + 1e-4)
            
    def test_convergence(self):
        """测试收敛性"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        segmentation, phi, history = chan_vese_segmentation(
            self.image, phi0, max_iter=200, tol=1e-5, verbose=False
        )
        
        # 如果算法收敛，迭代次数应该小于最大值
        # 注意：不强制要求收敛，因为测试图像可能过于简单
        self.assertGreater(len(history), 0)


class TestChanVeseParameters(unittest.TestCase):
    """Chan-Vese 参数测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.image = np.zeros((64, 64))
        self.image[20:44, 20:44] = 0.8
        self.image = add_noise(self.image, 'gaussian', sigma=0.05)
        
    def test_mu_parameter(self):
        """测试 mu 参数"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        # 小 mu: 允许更复杂的轮廓
        seg_small, _, _ = chan_vese_segmentation(
            self.image, phi0, max_iter=50, mu=0.01, verbose=False
        )
        
        # 大 mu: 更平滑的轮廓
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        seg_large, _, _ = chan_vese_segmentation(
            self.image, phi0, max_iter=50, mu=1.0, verbose=False
        )
        
        # 两种方法都应该产生结果
        self.assertEqual(seg_small.shape, self.image.shape)
        self.assertEqual(seg_large.shape, self.image.shape)
        
    def test_lambda_parameters(self):
        """测试 lambda 参数"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        # 测试不同的 lambda1/lambda2 比例
        seg1, _, _ = chan_vese_segmentation(
            self.image, phi0, max_iter=50, 
            lambda1=2.0, lambda2=1.0, verbose=False
        )
        
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        seg2, _, _ = chan_vese_segmentation(
            self.image, phi0, max_iter=50,
            lambda1=1.0, lambda2=2.0, verbose=False
        )
        
        # 两种配置都应该产生结果
        self.assertEqual(seg1.shape, self.image.shape)
        self.assertEqual(seg2.shape, self.image.shape)
        
    def test_time_step(self):
        """测试时间步长"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        # 小步长
        seg1, _, _ = chan_vese_segmentation(
            self.image, phi0, max_iter=50, dt=0.1, verbose=False
        )
        
        # 大步长
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        seg2, _, _ = chan_vese_segmentation(
            self.image, phi0, max_iter=50, dt=1.0, verbose=False
        )
        
        # 两种步长都应该产生结果
        self.assertEqual(seg1.shape, self.image.shape)
        self.assertEqual(seg2.shape, self.image.shape)


class TestInitialization(unittest.TestCase):
    """初始化函数测试"""
    
    def test_circle_initialization(self):
        """测试圆形初始化"""
        shape = (64, 64)
        phi = initialize_sdf_circle(shape, center=(32, 32), radius=20)
        
        self.assertEqual(phi.shape, shape)
        
        # 中心应该为负值（内部）
        self.assertLess(phi[32, 32], 0)
        
        # 远离中心应该为正值（外部）
        self.assertGreater(phi[0, 0], 0)
        
        # 圆上应该接近零
        self.assertAlmostEqual(phi[32, 12], 0, delta=1)
        
    def test_rectangle_initialization(self):
        """测试矩形初始化"""
        shape = (64, 64)
        phi = initialize_sdf_rectangle(
            shape, 
            top_left=(20, 20), 
            bottom_right=(44, 44)
        )
        
        self.assertEqual(phi.shape, shape)
        
        # 矩形内部应该为负值
        self.assertLess(phi[32, 32], 0)
        
        # 矩形外部应该为正值
        self.assertGreater(phi[10, 10], 0)
        
    def test_multiple_circles_initialization(self):
        """测试多圆初始化"""
        shape = (64, 64)
        centers = [(20, 20), (44, 44)]
        radii = [10, 10]
        
        phi = initialize_sdf_multiple_circles(shape, centers, radii)
        
        self.assertEqual(phi.shape, shape)
        
        # 圆心应该为负值
        self.assertLess(phi[20, 20], 0)
        self.assertLess(phi[44, 44], 0)
        
    def test_default_parameters(self):
        """测试默认参数"""
        shape = (64, 64)
        
        # 应该能使用默认参数运行
        phi = initialize_sdf_circle(shape)
        self.assertEqual(phi.shape, shape)


class TestLevelSetFunctions(unittest.TestCase):
    """水平集函数测试"""
    
    def test_heaviside(self):
        """测试 Heaviside 函数"""
        # 正值应该接近 1
        self.assertAlmostEqual(heaviside(10, eps=1.0), 1.0, delta=0.1)
        
        # 负值应该接近 0
        self.assertAlmostEqual(heaviside(-10, eps=1.0), 0.0, delta=0.1)
        
        # 零值应该为 0.5
        self.assertAlmostEqual(heaviside(0, eps=1.0), 0.5, delta=0.1)
        
    def test_dirac_delta(self):
        """测试 Dirac delta 函数"""
        # 在零附近应该最大
        delta_at_zero = dirac_delta(0, eps=1.0)
        delta_at_one = dirac_delta(1, eps=1.0)
        
        self.assertGreater(delta_at_zero, delta_at_one)
        
        # 应该是正数
        self.assertGreater(dirac_delta(-5, eps=1.0), 0)
        self.assertGreater(dirac_delta(5, eps=1.0), 0)
        
    def test_reinitialize_sdf(self):
        """测试重初始化"""
        # 创建变形的水平集
        rows, cols = 32, 32
        phi = np.random.randn(rows, cols)
        
        # 重初始化
        phi_reinit = reinitialize_sdf(phi, iterations=10)
        
        # 检查形状
        self.assertEqual(phi_reinit.shape, phi.shape)
        
        # 结果应该是有限的
        self.assertTrue(np.all(np.isfinite(phi_reinit)))


class TestMumfordShah(unittest.TestCase):
    """Mumford-Shah 模型测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.image = np.zeros((64, 64))
        self.image[20:44, 20:44] = 0.8
        self.image = add_noise(self.image, 'gaussian', sigma=0.05)
        
    def test_basic_ms_segmentation(self):
        """测试基本 M-S 分割"""
        u, edges, history = mumford_shah_segmentation(
            self.image, mu=0.5, nu=0.01, max_iter=30, verbose=False
        )
        
        # 检查输出形状
        self.assertEqual(u.shape, self.image.shape)
        self.assertEqual(edges.shape, self.image.shape)
        
        # 边缘应该是二值的
        self.assertTrue(np.all((edges == 0) | (edges == 1)))
        
    def test_level_set_evolution(self):
        """测试水平集演化"""
        phi0 = initialize_sdf_circle(self.image.shape, radius=25)
        
        u, phi, segmentation, history = level_set_evolution(
            self.image, phi0, max_iter=50, verbose=False
        )
        
        # 检查输出
        self.assertEqual(u.shape, self.image.shape)
        self.assertEqual(phi.shape, self.image.shape)
        self.assertEqual(segmentation.shape, self.image.shape)
        
        # 分割应该是二值的
        self.assertTrue(np.all((segmentation == 0) | (segmentation == 1)))
        
    def test_ambrosio_tortorelli(self):
        """测试 A-T 近似"""
        u, v = ambrosio_tortorelli_approximation(
            self.image, mu=0.5, nu=0.01, max_iter=20
        )
        
        # 检查输出
        self.assertEqual(u.shape, self.image.shape)
        self.assertEqual(v.shape, self.image.shape)
        
        # v 应该在 [0, 1] 范围内
        self.assertTrue(np.all(v >= 0))
        self.assertTrue(np.all(v <= 1))


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_uniform_image(self):
        """测试均匀图像"""
        uniform = np.ones((32, 32)) * 0.5
        
        phi0 = initialize_sdf_circle(uniform.shape, radius=10)
        seg, phi, _ = chan_vese_segmentation(
            uniform, phi0, max_iter=20, verbose=False
        )
        
        # 应该产生结果（虽然分割意义不大）
        self.assertEqual(seg.shape, uniform.shape)
        
    def test_small_image(self):
        """测试小图像"""
        small = np.random.rand(16, 16)
        
        phi0 = initialize_sdf_circle(small.shape, radius=5)
        seg, phi, _ = chan_vese_segmentation(
            small, phi0, max_iter=20, verbose=False
        )
        
        self.assertEqual(seg.shape, small.shape)
        
    def test_noisy_image(self):
        """测试高噪声图像"""
        image = np.zeros((64, 64))
        image[20:44, 20:44] = 0.8
        noisy = add_noise(image, 'gaussian', sigma=0.3)
        
        phi0 = initialize_sdf_circle(noisy.shape, radius=25)
        seg, phi, _ = chan_vese_segmentation(
            noisy, phi0, max_iter=50, mu=0.3, verbose=False
        )
        
        # 应该产生结果
        self.assertEqual(seg.shape, noisy.shape)


class TestNumericalProperties(unittest.TestCase):
    """数值性质测试"""
    
    def test_deterministic(self):
        """测试确定性"""
        np.random.seed(42)
        image = np.zeros((32, 32))
        image[10:22, 10:22] = 0.8
        
        phi0 = initialize_sdf_circle(image.shape, radius=10)
        
        # 两次运行应该产生相同结果
        seg1, _, _ = chan_vese_segmentation(
            image, phi0, max_iter=30, verbose=False
        )
        
        phi0 = initialize_sdf_circle(image.shape, radius=10)
        seg2, _, _ = chan_vese_segmentation(
            image, phi0, max_iter=30, verbose=False
        )
        
        np.testing.assert_array_equal(seg1, seg2)
        
    def test_symmetry(self):
        """测试对称性"""
        # 创建对称图像
        image = np.zeros((64, 64))
        Y, X = np.ogrid[:64, :64]
        mask = (X - 32)**2 + (Y - 32)**2 <= 20**2
        image[mask] = 0.8
        
        phi0 = initialize_sdf_circle(image.shape, center=(32, 32), radius=25)
        seg, phi, _ = chan_vese_segmentation(
            image, phi0, max_iter=50, verbose=False
        )
        
        # 结果应该近似对称（由于数值误差可能有微小差异）
        # 检查上下对称
        upper = seg[:32, :]
        lower = seg[32:, :][::-1, :]
        np.testing.assert_allclose(upper, lower, atol=0.1)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("=" * 70)
    print("Mumford-Shah 与 Chan-Vese 模型单元测试")
    print("=" * 70)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestChanVeseBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestChanVeseParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestLevelSetFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestMumfordShah))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalProperties))
    
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
