#!/usr/bin/env python3
"""
下载 Xiaohao Cai 老师的论文
基于 DOI/arXiv 链接自动下载
"""

import os
import requests
import time
import re
from urllib.parse import unquote

# 创建下载目录
download_dir = "xiaohao_cai_papers"
os.makedirs(download_dir, exist_ok=True)

# 所有论文列表 (标题, DOI/arXiv ID, 类型)
papers = [
    # Page 1
    ("Revisiting cross-domain problem for LiDAR-based 3D object detection", "10.1007/978-981-96-7036-9_6", "springer"),
    ("Genes shells and AI Using computer vision to detect cryptic morphological divergence", "10.1038/s41598-025-30613-1", "nature"),
    ("EmoPerso enhancing personality detection with self-supervised emotion-aware modelling", "10.1145/3746252.3761247", "acm"),
    ("From instance segmentation to 3D growth trajectory reconstruction in Planktonic foraminifera", "arXiv.2511.02142", "arxiv"),
    ("CNNs RNNs and Transformers in human action recognition a survey and a hybrid model", "10.1007/s10462-025-11388-3", "springer"),
    ("Hippd brain-inspired hierarchical information processing for personality detection", "arXiv.2510.09893", "arxiv"),
    ("tCURLoRA tensor CUR decomposition based low-rank parameter adaptation", "10.1007/978-3-032-05325-1_55", "springer"),
    ("CenSegNet a generalist high-throughput deep learning framework for centrosome phenotyping", "10.1101/2025.09.15.676250", "biorxiv"),
    ("Talk2Radar bridging natural language with 4D mmWave radar", "10.1109/ICRA55743.2025.11128399", "ieee"),
    ("GRASPTrack geometry-reasoned association via segmentation and projection", "arXiv.2508.08117", "arxiv"),

    # Page 2
    ("A computer vision method for finding mislabelled specimens", "10.1002/ece3.71648", "wiley"),
    ("Explainable artificial intelligence advancements and limitations", "10.3390/app15137261", "mdpi"),
    ("Neural varifolds an aggregate representation for quantifying geometry of point clouds", "", "none"),
    ("Mogo residual quantized hierarchical causal transformer for 3D human motion generation", "arXiv.2506.05952", "arxiv"),
    ("Less but better parameter-efficient fine-tuning of large language models", "arXiv.2504.05411", "arxiv"),
    ("CornerPoint3D look at the nearest corner instead of the center", "arXiv.2504.02464", "arxiv"),
    ("LL4G self-supervised dynamic optimization for graph-based personality detection", "arXiv.2504.02146", "arxiv"),
    ("Human action recognition based on CNNs and vision transformers", "", "none"),
    ("Medical image classification by incorporating clinical variables", "10.1098/rsos.241222", "royal"),
    ("GAMED knowledge adaptive multi-experts decoupling for multimodal fake news detection", "10.1145/3701551.3703541", "acm"),

    # Page 3
    ("Concept-based explainable artificial intelligence metrics and benchmarks", "arXiv.2501.19271", "arxiv"),
    ("GAMED knowledge adaptive multi-experts decoupling", "arXiv.2412.12164", "arxiv"),
    ("Non-negative subspace feature representation for few-shot learning", "10.1016/j.imavis.2024.105334", "elsevier"),
    ("Few-shot learning for inference in medical imaging", "10.1371/journal.pone.0309368", "plos"),
    ("3DKMI MATLAB package for shape signatures from Krawtchouk moments", "10.1111/2041-210X.14388", "wiley"),
    ("Detect closer surfaces that can be seen", "10.3233/FAIA240472", "ios"),
    ("Detect closer surfaces new modeling in cross-domain 3D object detection", "", "none"),
    ("Discrepancy-based diffusion models for lesion detection in brain MRI", "10.1016/j.compbiomed.2024.109079", "elsevier"),
    ("An efficient and versatile variational method for high-dimensional data classification", "10.1007/s10915-024-02644-9", "springer"),
    ("Neural varifolds aggregate representation for quantifying geometry of point clouds", "arXiv.2407.04844", "arxiv"),

    # Page 4
    ("RNNs CNNs and transformers in human action recognition survey", "arXiv.2407.06162", "arxiv"),
    ("Automatic identification of satellite features from ISAR images", "10.1109/CONTROL60310.2024.10532050", "ieee"),
    ("Biologically-inspired iterative learning control design", "10.1109/CONTROL60310.2024.10531909", "ieee"),
    ("Multilevel explainable artificial intelligence visual and linguistic explanations", "10.1109/TAI.2023.3308555", "ieee"),
    ("An efficient two-sided sketching method for large-scale tensor decomposition", "arXiv.2404.16580", "arxiv"),
    ("3D orientation field transform", "10.1007/s10044-024-01212-z", "springer"),
    ("Editorial segmentation and classification theories algorithms and applications", "10.3389/fcomp.2024.1363578", "frontiers"),
    ("Proximal nested sampling with data-driven priors", "10.3390/psf2023009013", "mdpi"),
    ("Remote sensing image ship matching utilising line features", "10.3390/s23239479", "mdpi"),
    ("IIHT medical report generation with image-to-indicator hierarchical transformer", "10.1007/978-981-99-8076-5_5", "springer"),

    # Page 5
    ("A bilevel formalism for the peer-reviewing problem", "10.3233/FAIA230263", "ios"),
    ("Using computer vision to identify limpets from their shells", "10.3389/fmars.2023.1167818", "frontiers"),
    ("Semantic segmentation by semantic proportions", "arXiv.2305.15608", "arxiv"),
    ("Practical sketching algorithms for low-rank tucker approximation", "10.1007/s10915-023-02172-y", "springer"),
    ("TransNet transfer learning-based network for human action recognition", "10.1109/ICMLA58977.2023.00277", "ieee"),
    ("Robust Bayesian attention belief network for radar work mode recognition 2023", "10.1016/j.dsp.2022.103874", "elsevier"),
    ("An overview of SaT segmentation methodology", "10.1007/978-3-030-98661-2_75", "springer"),
    ("Data augmentation in classification and segmentation", "10.3390/jimaging9020046", "mdpi"),
    ("Robust Bayesian attention belief network for radar work mode recognition 2022", "10.1016/j.dsp.2022.103874", "elsevier"),
    ("Balanced neural architecture search and optimization", "10.1109/RFID-TA54958.2022.9924146", "ieee"),

    # Page 6
    ("Proximal nested sampling for high-dimensional Bayesian model selection", "10.1007/s11222-022-10152-9", "springer"),
    ("DNCNet deep radar signal denoising and recognition", "10.1109/TAES.2022.3153756", "ieee"),
    ("Electron tomography analysis of prolamellar body", "10.1093/plcell/koac205", "oxford"),
    ("Sparse Bayesian mass mapping with uncertainties hypothesis testing", "10.1093/mnras/stab1983", "oxford"),
    ("Balanced Neural Architecture Search and Its Application in SEI", "10.1109/TSP.2021.3107633", "ieee"),
    ("Deep Rectum Segmentation for Image Guided Radiation Therapy", "10.23919/EUSIPCO54536.2021.9616115", "ieee"),
    ("Offline and online reconstruction for radio interferometric imaging", "10.23919/URSIGASS49373.2020.9232233", "ieee"),
    ("Wavelet-based segmentation on the sphere", "10.1016/j.patcog.2019.107081", "elsevier"),
    ("3D segmentation of trees through flexible multiclass graph cut", "10.1109/TGRS.2019.2940146", "ieee"),
    ("Sparse Bayesian mass mapping with uncertainties local credible intervals", "10.1093/mnras/stz3453", "oxford"),

    # Page 7
    ("Electron tomography analysis of thylakoid assembly in Bienertia", "10.1038/s41598-019-56083-w", "nature"),
    ("Sparse Bayesian mass mapping with uncertainties peak statistics", "10.1093/mnras/stz2373", "oxford"),
    ("Quantifying uncertainty in high dimensional inverse problems", "10.23919/EUSIPCO.2019.8903038", "ieee"),
    ("Online radio interferometric imaging Assimilating and discarding visibilities", "10.1093/mnras/stz704", "oxford"),
    ("A two-stage classification method for high-dimensional data", "", "none"),
    ("Distributed and parallel sparse convex optimization for radio interferometry", "", "none"),
    ("Linkage between piecewise constant Mumford-Shah and ROF model", "10.1137/18M1202980", "siam"),
    ("Uncertainty quantification for radio interferometric imaging I", "10.1093/MNRAS/STY2004", "oxford"),
    ("Uncertainty quantification for radio interferometric imaging II", "10.1093/MNRAS/STY2015", "oxford"),
    ("The synergy between different colour spaces for degraded images", "10.31988/scitrends.10714", "scitrends"),

    # Page 8
    ("A three-stage approach for segmenting degraded color images SLaT", "10.1007/s10915-017-0402-2", "springer"),
    ("A graph cut approach to 3D tree delineation", "", "eprints"),
    ("Variational-based segmentation of bio-pores in tomographic images", "10.1016/j.cageo.2016.09.013", "elsevier"),
    ("VoxTox research programme", "10.23726/cij.2017.457", "cij"),
    ("Individual tree species classification from airborne multisensor imagery", "10.1109/JSTARS.2016.2569408", "ieee"),
    ("Automatic contouring of soft organs for image-guided prostate radiotherapy", "10.1016/S0167-8140(16)33144-9", "elsevier"),
    ("Mapping individual trees from airborne multi-sensor imagery", "10.1109/IGARSS.2015.7327059", "ieee"),
    ("Nonparametric image registration of airborne LiDAR hyperspectral", "10.1109/TGRS.2015.2431692", "ieee"),
    ("Variational image segmentation model coupled with image restoration", "10.1016/j.patcog.2015.01.008", "elsevier"),
    ("Accuracy of manual and automated rectal contours", "10.1200/jco.2015.33.7_suppl.94", "asco"),

    # Page 9
    ("Disparity and optical flow partitioning using extended Potts priors", "10.1093/imaiai/iau010", "oxford"),
    ("Vessel segmentation in medical imaging using tight-frame-based algorithm", "10.1137/110843472", "siam"),
    ("Two-stage image segmentation using convex variant of Mumford-Shah", "10.1137/120867068", "siam"),
    ("Multiclass segmentation by iterated ROF thresholding", "10.1007/978-3-642-40395-8_18", "springer"),
    ("Framelet-based algorithm for segmentation of tubular structures", "10.1007/978-3-642-24785-9_35", "springer"),
]

def sanitize_filename(title):
    """清理文件名"""
    # 移除或替换非法字符
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = title[:100]  # 限制长度
    return title.strip()

def download_arxiv(arxiv_id, filename):
    """下载 arXiv 论文"""
    # 提取 arxiv 编号
    arxiv_num = arxiv_id.replace("arXiv.", "").replace("arXiv:", "").strip()

    # arXiv 下载链接
    urls = [
        f"https://arxiv.org/pdf/{arxiv_num}.pdf",
        f"https://arxiv.org/pdf/{arxiv_num}",
    ]

    for url in urls:
        try:
            print(f"  尝试下载: {url}")
            response = requests.get(url, timeout=30, allow_redirects=True)
            if response.status_code == 200 and len(response.content) > 1000:
                filepath = os.path.join(download_dir, f"{filename}.pdf")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  [OK] 成功下载: {filepath}")
                return True
        except Exception as e:
            print(f"  [FAIL] 失败")
            continue
    return False

def download_from_doi(doi, publisher, filename):
    """尝试从 DOI 下载论文"""
    if not doi:
        return False

    # 构建可能的 PDF 链接
    urls = []

    if publisher == "springer":
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
        urls.append(f"https://link.springer.com/content/pdf/10.1007/{doi.split('/')[-1]}.pdf")
    elif publisher == "ieee":
        urls.append(f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={doi.split('/')[-1]}")
    elif publisher == "nature":
        urls.append(f"https://www.nature.com/articles/{doi.replace('10.1038/', '')}.pdf")
    elif publisher == "elsevier":
        urls.append(f"https://doi.org/{doi}")
    elif publisher == "mdpi":
        urls.append(f"https://www.mdpi.com/{doi}/pdf")
    elif publisher == "arxiv":
        return download_arxiv(doi, filename)
    elif publisher == "acm":
        urls.append(f"https://doi.org/{doi}")
    elif publisher == "oxford":
        urls.append(f"https://academic.oup.com/download.php?filename={doi}")
    elif publisher == "wiley":
        urls.append(f"https://onlinelibrary.wiley.com/doi/pdf/{doi}")
    elif publisher == "siam":
        urls.append(f"https://epubs.siam.org/doi/pdf/{doi}")
    elif publisher == "frontiers":
        urls.append(f"https://www.frontiersin.org/articles/{doi}/pdf")

    # 添加 DOI 解析链接
    urls.insert(0, f"https://doi.org/{doi}")

    for url in urls:
        try:
            print(f"  尝试: {url[:80]}...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=30, allow_redirects=True, headers=headers)

            # 检查是否是 PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type.lower() or response.content[:4] == b'%PDF':
                filepath = os.path.join(download_dir, f"{filename}.pdf")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ 成功下载: {filepath}")
                return True
        except Exception as e:
            continue

    return False

def main():
    print(f"开始下载论文到目录: {download_dir}")
    print(f"共 {len(papers)} 篇论文\n")

    success_count = 0
    failed_papers = []

    for i, (title, doi, publisher) in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {title[:60]}...")

        filename = sanitize_filename(title)

        # 检查是否已存在
        filepath = os.path.join(download_dir, f"{filename}.pdf")
        if os.path.exists(filepath):
            print(f"  [OK] 已存在，跳过")
            success_count += 1
            continue

        # 尝试下载
        downloaded = False

        if publisher == "arxiv" or doi.startswith("arXiv"):
            downloaded = download_arxiv(doi, filename)
        elif doi:
            downloaded = download_from_doi(doi, publisher, filename)

        if downloaded:
            success_count += 1
        else:
            failed_papers.append((title, doi, publisher))
            print(f"  [FAIL] 无法自动下载")

        # 延迟避免被封
        time.sleep(1)
        print()

    # 报告结果
    print("\n" + "="*60)
    print(f"下载完成: {success_count}/{len(papers)} 篇论文")
    print(f"保存位置: {os.path.abspath(download_dir)}")

    if failed_papers:
        print(f"\n无法自动下载的论文 ({len(failed_papers)} 篇):")
        print("-"*60)
        for title, doi, publisher in failed_papers:
            print(f"  - {title[:70]}")
            if doi:
                print(f"    DOI: https://doi.org/{doi}")
            print(f"    出版商: {publisher}")
            print()

        # 保存失败列表
        with open(os.path.join(download_dir, "failed_downloads.txt"), "w", encoding="utf-8") as f:
            f.write("无法自动下载的论文列表:\n\n")
            for title, doi, publisher in failed_papers:
                f.write(f"标题: {title}\n")
                if doi:
                    f.write(f"DOI: https://doi.org/{doi}\n")
                f.write(f"出版商: {publisher}\n")
                f.write("-"*60 + "\n\n")

if __name__ == "__main__":
    main()
