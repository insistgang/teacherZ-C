#!/usr/bin/env python3
"""
尝试用其他方法下载剩余论文
- 搜索 arXiv 预印本
- 搜索作者主页
- 搜索 ResearchGate
- 尝试开放获取版本
"""

import os
import requests
import time
import re
from urllib.parse import quote

# 创建下载目录
download_dir = "xiaohao_cai_papers"
os.makedirs(download_dir, exist_ok=True)

# 需要下载的论文列表 (标题, DOI, 出版商)
papers = [
    ("EmoPerso enhancing personality detection with self-supervised emotion-aware modelling", "10.1145/3746252.3761247", "acm"),
    ("CenSegNet a generalist high-throughput deep learning framework for centrosome phenotyping", "10.1101/2025.09.15.676250", "biorxiv"),
    ("Talk2Radar bridging natural language with 4D mmWave radar", "10.1109/ICRA55743.2025.11128399", "ieee"),
    ("A computer vision method for finding mislabelled specimens", "10.1002/ece3.71648", "wiley"),
    ("Explainable artificial intelligence advancements and limitations", "10.3390/app15137261", "mdpi"),
    ("Neural varifolds an aggregate representation for quantifying geometry of point clouds", "", "none"),
    ("Human action recognition based on CNNs and vision transformers", "", "none"),
    ("Medical image classification by incorporating clinical variables", "10.1098/rsos.241222", "royal"),
    ("GAMED knowledge adaptive multi-experts decoupling for multimodal fake news detection", "10.1145/3701551.3703541", "acm"),
    ("Non-negative subspace feature representation for few-shot learning", "10.1016/j.imavis.2024.105334", "elsevier"),
    ("Few-shot learning for inference in medical imaging", "10.1371/journal.pone.0309368", "plos"),
    ("3DKMI MATLAB package for shape signatures from Krawtchouk moments", "10.1111/2041-210X.14388", "wiley"),
    ("Detect closer surfaces that can be seen", "10.3233/FAIA240472", "ios"),
    ("Detect closer surfaces new modeling in cross-domain 3D object detection", "", "none"),
    ("Discrepancy-based diffusion models for lesion detection in brain MRI", "10.1016/j.compbiomed.2024.109079", "elsevier"),
    ("Automatic identification of satellite features from ISAR images", "10.1109/CONTROL60310.2024.10532050", "ieee"),
    ("Biologically-inspired iterative learning control design", "10.1109/CONTROL60310.2024.10531909", "ieee"),
    ("Multilevel explainable artificial intelligence visual and linguistic explanations", "10.1109/TAI.2023.3308555", "ieee"),
    ("Proximal nested sampling with data-driven priors", "10.3390/psf2023009013", "mdpi"),
    ("Remote sensing image ship matching utilising line features", "10.3390/s23239479", "mdpi"),
    ("A bilevel formalism for the peer-reviewing problem", "10.3233/FAIA230263", "ios"),
    ("Using computer vision to identify limpets from their shells", "10.3389/fmars.2023.1167818", "frontiers"),
    ("TransNet transfer learning-based network for human action recognition", "10.1109/ICMLA58977.2023.00277", "ieee"),
    ("Robust Bayesian attention belief network for radar work mode recognition 2023", "10.1016/j.dsp.2022.103874", "elsevier"),
    ("Data augmentation in classification and segmentation", "10.3390/jimaging9020046", "mdpi"),
    ("Balanced neural architecture search and optimization", "10.1109/RFID-TA54958.2022.9924146", "ieee"),
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
    ("Sparse Bayesian mass mapping with uncertainties peak statistics", "10.1093/mnras/stz2373", "oxford"),
    ("Quantifying uncertainty in high dimensional inverse problems", "10.23919/EUSIPCO.2019.8903038", "ieee"),
    ("Online radio interferometric imaging Assimilating and discarding visibilities", "10.1093/mnras/stz704", "oxford"),
    ("A two-stage classification method for high-dimensional data", "", "none"),
    ("Distributed and parallel sparse convex optimization for radio interferometry", "", "none"),
    ("Linkage between piecewise constant Mumford-Shah and ROF model", "10.1137/18M1202980", "siam"),
    ("Uncertainty quantification for radio interferometric imaging I", "10.1093/MNRAS/STY2004", "oxford"),
    ("Uncertainty quantification for radio interferometric imaging II", "10.1093/MNRAS/STY2015", "oxford"),
    ("The synergy between different colour spaces for degraded images", "10.31988/scitrends.10714", "scitrends"),
    ("A graph cut approach to 3D tree delineation", "", "eprints"),
    ("Variational-based segmentation of bio-pores in tomographic images", "10.1016/j.cageo.2016.09.013", "elsevier"),
    ("VoxTox research programme", "10.23726/cij.2017.457", "cij"),
    ("Individual tree species classification from airborne multisensor imagery", "10.1109/JSTARS.2016.2569408", "ieee"),
    ("Automatic contouring of soft organs for image-guided prostate radiotherapy", "10.1016/S0167-8140(16)33144-9", "elsevier"),
    ("Mapping individual trees from airborne multi-sensor imagery", "10.1109/IGARSS.2015.7327059", "ieee"),
    ("Nonparametric image registration of airborne LiDAR hyperspectral", "10.1109/TGRS.2015.2431692", "ieee"),
    ("Variational image segmentation model coupled with image restoration", "10.1016/j.patcog.2015.01.008", "elsevier"),
    ("Accuracy of manual and automated rectal contours", "10.1200/jco.2015.33.7_suppl.94", "asco"),
    ("Disparity and optical flow partitioning using extended Potts priors", "10.1093/imaiai/iau010", "oxford"),
    ("Vessel segmentation in medical imaging using tight-frame-based algorithm", "10.1137/110843472", "siam"),
    ("Two-stage image segmentation using convex variant of Mumford-Shah", "10.1137/120867068", "siam"),
]

def sanitize_filename(title):
    """清理文件名"""
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = title[:100]
    return title.strip()

def try_download_arxiv(title, filename):
    """尝试从 arXiv 搜索并下载"""
    # 提取关键词（去掉常见词）
    keywords = title.replace("a", "").replace("the", "").replace("and", "").replace("for", "").replace("in", "")
    keywords = keywords.replace("of", "").replace("with", "").replace("to", "").replace("from", "")
    keywords = " ".join(keywords.split()[:5])  # 取前5个词

    # arXiv API 搜索
    try:
        search_query = quote(keywords)
        url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results=5"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # 检查是否有结果
            if "<entry>" in response.text:
                # 提取 arXiv ID
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                entries = root.findall('.//atom:entry', ns)
                for entry in entries:
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None:
                        arxiv_url = id_elem.text
                        arxiv_id = arxiv_url.split('/')[-1]
                        if arxiv_id:
                            # 尝试下载 PDF
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                            pdf_response = requests.get(pdf_url, timeout=30)
                            if pdf_response.status_code == 200 and len(pdf_response.content) > 10000:
                                filepath = os.path.join(download_dir, f"{filename}.pdf")
                                with open(filepath, 'wb') as f:
                                    f.write(pdf_response.content)
                                return True, f"arXiv:{arxiv_id}"
    except Exception as e:
        pass
    return False, ""

def try_semantic_scholar(title, filename):
    """尝试从 Semantic Scholar 获取 PDF"""
    try:
        # Semantic Scholar API
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title[:200],
            "fields": "paperId,title,openAccessPdf",
            "limit": 5
        }
        response = requests.get(search_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                for paper in data["data"]:
                    if paper.get("openAccessPdf") and paper["openAccessPdf"].get("url"):
                        pdf_url = paper["openAccessPdf"]["url"]
                        pdf_response = requests.get(pdf_url, timeout=30, allow_redirects=True)
                        if pdf_response.status_code == 200 and len(pdf_response.content) > 10000:
                            # 检查是否是PDF
                            if pdf_response.content[:4] == b'%PDF' or 'pdf' in pdf_response.headers.get('content-type', '').lower():
                                filepath = os.path.join(download_dir, f"{filename}.pdf")
                                with open(filepath, 'wb') as f:
                                    f.write(pdf_response.content)
                                return True, "Semantic Scholar"
    except Exception as e:
        pass
    return False, ""

def try_unpaywall(doi, filename):
    """尝试通过 Unpaywall API 获取开放获取版本"""
    if not doi:
        return False, ""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email=user@example.com"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # 检查是否有开放获取版本
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("url_for_pdf"):
                pdf_url = best_oa["url_for_pdf"]
                pdf_response = requests.get(pdf_url, timeout=30, allow_redirects=True)
                if pdf_response.status_code == 200 and len(pdf_response.content) > 10000:
                    if pdf_response.content[:4] == b'%PDF':
                        filepath = os.path.join(download_dir, f"{filename}.pdf")
                        with open(filepath, 'wb') as f:
                            f.write(pdf_response.content)
                        return True, "Unpaywall"
    except Exception as e:
        pass
    return False, ""

def try_direct_publisher(doi, publisher, filename):
    """尝试直接从出版商下载开放获取版本"""
    if not doi:
        return False, ""

    urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    if publisher == "mdpi":
        # MDPI 通常是开放获取
        article_id = doi.split('/')[-1].split('.')[-1]
        urls.append(f"https://www.mdpi.com/{article_id}/pdf")
    elif publisher == "frontiers":
        # Frontiers 通常是开放获取
        article_id = doi.split('/')[-1]
        urls.append(f"https://www.frontiersin.org/articles/{doi}/pdf")
    elif publisher == "plos":
        # PLOS 是开放获取
        article_id = doi.split('/')[-1]
        urls.append(f"https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.{article_id}&type=printable")
    elif publisher == "royal":
        # Royal Society Open Science
        urls.append(f"https://royalsocietypublishing.org/doi/pdf/{doi}")
    elif publisher == "oxford":
        # Oxford Academic - 部分开放获取
        urls.append(f"https://academic.oup.com/mnras/article-pdf/{doi}")
    elif publisher == "ieee":
        # IEEE - 尝试直接链接
        article_num = doi.split('/')[-1]
        urls.append(f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?arnumber={article_num}")
    elif publisher == "elsevier":
        # Elsevier - 尝试
        urls.append(f"https://www.sciencedirect.com/science/article/pii/{doi.replace('.', '').replace('/', '')}")
    elif publisher == "springer":
        # Springer - 部分开放获取
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    elif publisher == "wiley":
        # Wiley - 部分开放获取
        urls.append(f"https://onlinelibrary.wiley.com/doi/pdf/{doi}")
    elif publisher == "biorxiv":
        # bioRxiv 预印本
        article_id = doi.replace("10.1101/", "")
        urls.append(f"https://www.biorxiv.org/content/10.1101/{article_id}.full.pdf")

    for url in urls:
        try:
            response = requests.get(url, timeout=30, allow_redirects=True, headers=headers)
            if response.status_code == 200 and len(response.content) > 10000:
                if response.content[:4] == b'%PDF':
                    filepath = os.path.join(download_dir, f"{filename}.pdf")
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    return True, f"Publisher:{publisher}"
        except:
            continue

    return False, ""

def main():
    print(f"尝试下载剩余论文到: {download_dir}")
    print(f"共 {len(papers)} 篇待下载\n")

    success_count = 0
    failed_papers = []

    for i, (title, doi, publisher) in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {title[:60]}...")

        filename = sanitize_filename(title)

        # 检查是否已存在
        filepath = os.path.join(download_dir, f"{filename}.pdf")
        if os.path.exists(filepath):
            print(f"  [OK] 已存在")
            success_count += 1
            continue

        downloaded = False
        source = ""

        # 方法1: 尝试 Unpaywall (开放获取数据库)
        if not downloaded and doi:
            print(f"  尝试 Unpaywall...")
            downloaded, source = try_unpaywall(doi, filename)
            if downloaded:
                print(f"  [OK] 从 {source} 下载成功")

        # 方法2: 尝试 Semantic Scholar
        if not downloaded:
            print(f"  尝试 Semantic Scholar...")
            downloaded, source = try_semantic_scholar(title, filename)
            if downloaded:
                print(f"  [OK] 从 {source} 下载成功")

        # 方法3: 尝试 arXiv
        if not downloaded:
            print(f"  尝试 arXiv...")
            downloaded, source = try_download_arxiv(title, filename)
            if downloaded:
                print(f"  [OK] 从 {source} 下载成功")

        # 方法4: 尝试直接出版商链接
        if not downloaded and doi:
            print(f"  尝试出版商 ({publisher})...")
            downloaded, source = try_direct_publisher(doi, publisher, filename)
            if downloaded:
                print(f"  [OK] 从 {source} 下载成功")

        if downloaded:
            success_count += 1
        else:
            failed_papers.append((title, doi, publisher))
            print(f"  [FAIL] 无法下载")

        time.sleep(2)  # 避免请求过快
        print()

    # 报告结果
    print("\n" + "="*60)
    print(f"本次下载成功: {success_count}/{len(papers)} 篇")
    print(f"总计已下载: {len([f for f in os.listdir(download_dir) if f.endswith('.pdf')])} 篇")

    if failed_papers:
        print(f"\n仍然无法下载的论文 ({len(failed_papers)} 篇):")
        print("-"*60)
        for title, doi, publisher in failed_papers:
            print(f"  - {title[:70]}")
            if doi:
                print(f"    DOI: https://doi.org/{doi}")
            print()

        # 保存失败列表
        with open(os.path.join(download_dir, "still_failed.txt"), "w", encoding="utf-8") as f:
            f.write("仍然无法下载的论文列表:\n\n")
            for title, doi, publisher in failed_papers:
                f.write(f"标题: {title}\n")
                if doi:
                    f.write(f"DOI: https://doi.org/{doi}\n")
                f.write(f"出版商: {publisher}\n")
                f.write("-"*60 + "\n\n")

if __name__ == "__main__":
    main()
