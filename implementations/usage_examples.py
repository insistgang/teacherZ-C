"""
论文算法实现汇总使用示例
==============================

基于 D:\Documents\zx 项目中的核心论文实现

包含4个模块:
    1. SLaT 三阶段分割 (slat_segmentation.py)
    2. ROF 迭代阈值分割 (rof_iterative_segmentation.py)
    3. Tucker 分解加速 (tucker_decomposition.py)
    4. Neural Varifold 点云核 (neural_varifold.py)

运行方式:
    python usage_examples.py
"""

import numpy as np


def demo_slat():
    """SLaT三阶段分割演示"""
    print("\n" + "=" * 70)
    print("【1】SLaT 三阶段分割")
    print("=" * 70)

    from slat_segmentation import SLATSegmentation, slat_segment

    np.random.seed(42)
    H, W = 64, 64

    image = np.zeros((H, W, 3), dtype=np.float64)
    image[: H // 2, : W // 2] = [0.8, 0.2, 0.2]
    image[: H // 2, W // 2 :] = [0.2, 0.8, 0.2]
    image[H // 2 :, : W // 2] = [0.2, 0.2, 0.8]
    image[H // 2 :, W // 2 :] = [0.5, 0.5, 0.5]
    image = np.clip(image + np.random.randn(H, W, 3) * 0.08, 0, 1)

    segmenter = SLATSegmentation(lambda_param=1.5)
    result = segmenter.segment(image, K=4, return_intermediate=True)

    print(f"输入图像: {image.shape}")
    print(f"Stage 1 平滑: {result['smoothed'].shape}")
    print(f"Stage 2 升维: {result['lifted'].shape} (RGB→RGB+Lab)")
    print(f"Stage 3 分割: {result['segmentation'].shape}")
    print(f"聚类中心: {result['cluster_centers'].shape}")

    print("\n【参数说明】")
    print("  - lambda_param: 数据保真项权重 (默认1.0)")
    print("  - 值越大，越保持原图细节；值越小，去噪效果越强")

    print("\n【使用示例】")
    print("  segmenter = SLATSegmentation(lambda_param=1.0)")
    print("  result = segmenter.segment(image, K=4)")
    print("  segmentation = result['segmentation']")

    return result


def demo_rof():
    """ROF迭代阈值分割演示"""
    print("\n" + "=" * 70)
    print("【2】ROF 迭代阈值分割")
    print("=" * 70)

    from rof_iterative_segmentation import (
        IterativeROFSegmentation,
        AutomaticThresholdROF,
        ROFDenoiser,
    )

    np.random.seed(42)
    H, W = 80, 80

    gray = np.zeros((H, W), dtype=np.float64)
    gray[: H // 3] = 0.2
    gray[H // 3 : 2 * H // 3] = 0.5
    gray[2 * H // 3 :] = 0.8
    noisy = np.clip(gray + np.random.randn(H, W) * 0.1, 0, 1)

    print("\n--- 迭代ROF多类分割 ---")
    segmenter = IterativeROFSegmentation(lambda_param=0.15, tree_type="balanced")
    seg_multi = segmenter.segment(noisy, K=3)
    print(f"输入: {noisy.shape}")
    print(f"分割结果: {seg_multi.shape}")
    print(f"各类像素: {[np.sum(seg_multi == k) for k in range(3)]}")

    print("\n--- ROF去噪效果 ---")
    denoiser = ROFDenoiser(max_iter=100)
    denoised = denoiser.solve(noisy, lambda_param=0.2)
    psnr_before = 10 * np.log10(1.0 / np.mean((noisy - gray) ** 2))
    psnr_after = 10 * np.log10(1.0 / np.mean((denoised - gray) ** 2))
    print(f"去噪前 PSNR: {psnr_before:.2f} dB")
    print(f"去噪后 PSNR: {psnr_after:.2f} dB")

    print("\n【参数说明】")
    print("  - lambda_param: ROF正则化参数 (默认0.1)")
    print("  - tree_type: 'balanced'(平衡树) 或 'sequential'(顺序树)")

    print("\n【使用示例】")
    print("  segmenter = IterativeROFSegmentation(lambda_param=0.1)")
    print("  segmentation = segmenter.segment(image, K=4)")

    return seg_multi


def demo_tucker():
    """Tucker分解演示"""
    print("\n" + "=" * 70)
    print("【3】Tucker 分解加速 (Sketching + HOOI)")
    print("=" * 70)

    from tucker_decomposition import (
        SketchingTucker,
        HOOIDecomposition,
        TensorOperations,
        reconstruct_tucker,
    )
    import time

    np.random.seed(42)
    I, J, K = 40, 30, 20
    R = [4, 3, 2]

    core_true = np.random.randn(*R)
    tensor = core_true.copy()
    for n, (I_n, R_n) in enumerate([(I, R[0]), (J, R[1]), (K, R[2])]):
        A = np.random.randn(I_n, R_n)
        tensor = TensorOperations.mode_n_product(tensor, A, n)

    print(f"原始张量: {tensor.shape}")
    print(f"目标Tucker秩: {R}")

    print("\n--- 标准HOOI分解 ---")
    hooi = HOOIDecomposition(ranks=R, max_iter=50)
    start = time.time()
    core_hooi, factors_hooi = hooi.fit(tensor)
    time_hooi = time.time() - start

    X_hooi = reconstruct_tucker(core_hooi, factors_hooi)
    error_hooi = np.linalg.norm(tensor - X_hooi) / np.linalg.norm(tensor)
    print(f"时间: {time_hooi:.4f}s, 误差: {error_hooi:.6f}")

    print("\n--- Sketching加速分解 ---")
    sketchy = SketchingTucker(ranks=R, sketch_multipliers=2.0)
    start = time.time()
    core_sketch, factors_sketch = sketchy.fit(tensor)
    time_sketch = time.time() - start

    X_sketch = reconstruct_tucker(core_sketch, factors_sketch)
    error_sketch = np.linalg.norm(tensor - X_sketch) / np.linalg.norm(tensor)
    print(f"时间: {time_sketch:.4f}s, 误差: {error_sketch:.6f}")

    orig_params = np.prod(tensor.shape)
    tucker_params = np.prod(core_hooi.shape) + sum(
        f.shape[0] * f.shape[1] for f in factors_hooi
    )
    print(
        f"\n压缩比: {orig_params}/{tucker_params} = {orig_params / tucker_params:.1f}:1"
    )

    print("\n【参数说明】")
    print("  - ranks: 目标Tucker秩列表 [R1, R2, R3]")
    print("  - sketch_multipliers: Sketch尺寸 = multiplier × rank")

    print("\n【使用示例】")
    print("  sketchy = SketchingTucker(ranks=[5, 4, 3])")
    print("  core, factors = sketchy.fit(tensor)")
    print("  reconstructed = reconstruct_tucker(core, factors)")

    return core_hooi, factors_hooi


def demo_varifold():
    """Neural Varifold演示"""
    print("\n" + "=" * 70)
    print("【4】Neural Varifold 点云核")
    print("=" * 70)

    import torch
    import torch.nn.functional as F
    from neural_varifold import (
        PositionKernel,
        NormalKernel,
        VarifoldRepresentation,
        VarifoldKernel,
        VarifoldDistance,
        NeuralVarifoldNet,
        compute_varifold_norm,
    )

    torch.manual_seed(42)

    B, N = 2, 128
    positions = torch.randn(B, N, 3)
    normals = F.normalize(torch.randn(B, N, 3), dim=-1)

    print(f"点云: {positions.shape}")
    print(f"法向量: {normals.shape}")

    print("\n--- 位置核 ---")
    pos_k = PositionKernel(sigma=0.5)
    K_pos = pos_k(positions[0], positions[0])
    print(f"核矩阵: {K_pos.shape}, 值域: [{K_pos.min():.4f}, {K_pos.max():.4f}]")

    print("\n--- 法向量核 ---")
    norm_k = NormalKernel(exponent=1)
    K_norm = norm_k(normals[0], normals[0])
    print(f"核矩阵: {K_norm.shape}, 值域: [{K_norm.min():.4f}, {K_norm.max():.4f}]")

    print("\n--- Varifold表示 ---")
    encoder = VarifoldRepresentation(feat_dim=64, use_normals=True)
    features, weights, _ = encoder(positions, normals)
    print(f"特征: {features.shape}, 权重: {weights.shape}")

    print("\n--- Varifold距离 ---")
    v_dist = VarifoldDistance(sigma_pos=0.5)
    v1 = (positions[:1], features[:1], weights[:1])
    v2 = (positions[1:], features[1:], weights[1:])
    dist = v_dist(v1, v2)
    print(f"两点云Varifold距离: {dist.item():.6f}")

    print("\n--- 完整分割网络 ---")
    net = NeuralVarifoldNet(num_classes=5, feat_dim=64)
    logits = net(positions, normals)
    print(f"输入: {positions.shape}")
    print(f"输出: {logits.shape}")
    print(f"参数量: {sum(p.numel() for p in net.parameters()):,}")

    print("\n【参数说明】")
    print("  - sigma: 位置核带宽参数")
    print("  - feat_dim: 特征维度")
    print("  - num_classes: 分割类别数")

    print("\n【使用示例】")
    print("  net = NeuralVarifoldNet(num_classes=10)")
    print("  logits = net(positions, normals)")
    print("  pred = logits.argmax(dim=-1)")

    return net


def main():
    """运行所有演示"""
    print("\n" + "=" * 70)
    print("       论文核心算法实现演示")
    print("       D:\\Documents\\zx 项目")
    print("=" * 70)

    print("\n包含模块:")
    print("  1. SLaT 三阶段分割 (slat_segmentation.py)")
    print("  2. ROF 迭代阈值分割 (rof_iterative_segmentation.py)")
    print("  3. Tucker 分解加速 (tucker_decomposition.py)")
    print("  4. Neural Varifold 点云核 (neural_varifold.py)")

    demo_slat()
    demo_rof()
    demo_tucker()
    demo_varifold()

    print("\n" + "=" * 70)
    print("所有演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
