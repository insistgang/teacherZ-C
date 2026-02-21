#!/usr/bin/env python3
"""
检查笔记与PDF的对应关系
"""
import os
import re
from pathlib import Path

# 目录设置
PDF_DIR = Path("web-viewer/00_papers")
NOTE_DIR = Path("xiaohao_cai_ultimate_notes")

# PDF列表
pdf_files = list(PDF_DIR.glob("*.pdf"))
print(f"PDF总数: {len(pdf_files)}")

# 笔记列表
note_files = list(NOTE_DIR.glob("*超精读笔记*.md")) + list(NOTE_DIR.glob("[0-9]*.md")) + list(NOTE_DIR.glob("[A-Z]*.md"))
note_files = [f for f in note_files if f.name not in ["README.md", "00_分析报告汇总.md", "论文精读完成报告_20260220.md"]]
print(f"笔记总数: {len(note_files)}")

# 提取笔记中的元数据
note_metadata = []
for note_file in note_files:
    try:
        content = note_file.read_text(encoding="utf-8")
        # 提取arXiv ID
        arxiv_match = re.search(r"arXiv[：:]\s*([\d.]+v?\d*)", content)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        # 提取标题
        title_match = re.search(r"\*\*标题\*\*\s*\|\s*([^\n|]+)", content)
        title = title_match.group(1).strip() if title_match else None

        # 提取作者
        author_match = re.search(r"\*\*作者\*\*\s*\|\s*([^\n|]+)", content)
        author = author_match.group(1).strip() if author_match else None

        # 检查是否已填充
        is_filled = "已填充" in note_file.name

        note_metadata.append({
            "file": note_file.name,
            "arxiv_id": arxiv_id,
            "title": title,
            "author": author,
            "is_filled": is_filled
        })
    except Exception as e:
        print(f"读取笔记失败: {note_file.name}, {e}")

# PDF-笔记映射表
pdf_note_mapping = {
    "3D方向场变换 3D Orientation Field.pdf": "3D_Orientation_Field_Transform",
    "SLaT三阶段分割 SLaT Segmentation.pdf": "SLaT_Three-stage_Segmentation",
    "HiFi-Mamba MRI重建 HiFi-Mamba MRI Reconstruction.pdf": "HiFi-Mamba_MRI_Reconstruction",
    "3D树木描绘图割 3D Tree Delineation.pdf": "3D_Tree_Delineation_Graph_Cut",
    "3D树木分割图 3D Tree Segmentation.pdf": "3D_Tree_Segmentation_MCGC",
    "3DKMI Krawtchouk矩形状签名 3DKMI.pdf": "3DKMI",
    "2014_1410.0226_LiDAR Hyperspectral Registration.pdf": "LiDAR_Hyperspectral_Registration",
    "Bilevel Peer-Reviewing": "Bilevel_Peer-Reviewing_Problem",
    "CornerPoint3D 3D检测新尺度 CornerPoint3D.pdf": "CornerPoint3D",
    "变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf": "Mumford-Shah_and_ROF_Linkage",
    "两阶段分类 Two-Stage.pdf": "Two_Stage_High_Dimensional_Classification",
    "GAMED虚假新闻检测 GAMED Fake News.pdf": "GAMED",
    "HiFi-MambaV2分层MRI HiFi-MambaV2 Hierarchical MRI.pdf": "HiFi-MambaV2",
    "Equalizing Protected Attributes.pdf": "Equalizing_Protected_Attributes",
    "医学图像小样本学习 Medical Few-Shot.pdf": "Few-shot_Medical_Imaging_Inference",
    "高维逆问题不确定性量化 Uncertainty Quantification.pdf": "High-Dimensional_Inverse_Problems",
    "IIHT Medical Report IIHT.pdf": "IIHT",
    "近端嵌套采样 Proximal Nested Sampling.pdf": "Proximal_Nested_Sampling",
    "无线电干涉不确定性I Radio Interferometric I.pdf": "Radio_Interferometric_Imaging_I",
    "无线电干涉不确定性II Radio Interferometric II.pdf": "Radio_Interferometric_Imaging_II",
    "在线无线电干涉成像 Online Radio Imaging.pdf": "Online_Radio_Interferometric_Imaging",
    "Tucker近似 Tucker Approximation.pdf": "Practical_Sketching_Tucker_Approximation",
    "Tensor_Train_Approximation.pdf": "Tensor_Train_Approximation",
    "双面Sketching张量 Two-Sided Sketching.pdf": "Two_Sided_Sketching",
    "GO-LDA": "GO-LDA",
    "点云神经表示 Neural Varifolds.pdf": "Neural_varifolds",
    "语义比例分割 Semantic Proportions.pdf": "Semantic_Segmentation_by_Proportions",
    "跨域LiDAR检测 Cross-Domain LiDAR.pdf": "Cross-Domain_LiDAR",
    "可见表面检测 Detect Closer Surfaces.pdf": "Detect_Closer_Surfaces",
    "Diffusion Brain MRI.pdf": "Discrepancy-based_Diffusion_MRI",
    "EmoPerso Emotion-Aware.pdf": "EmoPerso",
    "高效PEFT微调 Less but Better PEFT.pdf": "Less_but_Better_PEFT",
    "蛋白质结构网络图LL4G LL4G Graph.pdf": "LL4G",
    "HIPPD Brain-Inspired.pdf": "HIPPD",
    "大模型高效微调CALM CALM Fine-tuning.pdf": "CALM",
    "概念级XAI指标 Concept XAI.pdf": "Concept-Based_XAI_Metrics",
    "TransNet动作识别 TransNet HAR.pdf": "TransNet",
    "深度学习架构综述 CNNs RNNs Transformers.pdf": "CNNs_RNNs_Transformers",
    "多层次可解释AI Multilevel XAI.pdf": "Multilevel_Explainable_AI",
    "框架分割管状结构 Framelet Tubular.pdf": "Framelet",
    "多类分割迭代ROF Iterated ROF.pdf": "多类分割迭代ROF",
    "分割方法论总览 SaT Overview.pdf": "分割方法论总览",
    "高效变分分类 Efficient Variational.pdf": "高效变分分类",
    "GRASPTrack": "GRASPTrack",
    "MotionDuet 3D Motion.pdf": "MotionDuet",
    "MOGO 3D人体运动生成 MOGO Motion.pdf": "MOGO",
    "Talk2Radar": "Talk2Radar",
    "tCURLoRA": "tCURLoRA",
    "CenSegNet中心体": "CenSegNet",
    "DNCNet": "DNCNet",
    "ISAR": "ISAR",
    "PURIFY": "PURIFY",
    "RobustPCA": "RobustPCA",
    "船舶匹配遥感 Ship Matching.pdf": "Ship_Matching",
    "稀疏贝叶斯": "稀疏贝叶斯",
    "数据增强综述": "数据增强",
    "可解释AI综述 XAI Survey.pdf": "可解释AI综述",
    "雷达工作模式识别": "雷达工作模式",
    "平衡神经网络搜索": "平衡神经网络搜索",
    "Biologically-Inspired": "Biologically-Inspired",
    "Federated_Learning": "Federated_Learning",
    "Deep_Learning_Rectum": "Deep_Learning_Rectum",
    "2019_Thylakoid": "2019_Thylakoid",
    "2023_Limpets": "2023_Limpets",
    "2025_Genes_Shells": "2025_Genes_Shells",
    "非负子空间": "Non-negative_Subspace",
    "分割恢复联合模型 Segmentation Restoration.pdf": "Variational_Segmentation-Restoration",
    "球面小波分割 Wavelet Sphere.pdf": "Wavelet_Segmentation_on_Sphere",
    "两阶段图像分割": "Two-Stage_Segmentation",
    "小波框架血管分割 Tight-Frame Vessel.pdf": "Tight-Frame_Vessel",
    "生物孔隙分割 Bio-Pores.pdf": "生物孔隙",
    "3D树木分割图割 3D Tree Graph Cut.pdf": "3D_Tree_Delineation",  # 注意
    "3D_Growth": "3D_Growth_Trajectory",
}

# 检查对应关系
matched = []
unmatched_pdfs = []
mismatched = []

for pdf in pdf_files:
    pdf_name = pdf.name
    matched_note = None

    for pdf_key, note_key in pdf_note_mapping.items():
        if pdf_key.lower() in pdf_name.lower() or pdf_name.lower().replace(' ', '_') in note_key.lower():
            # 检查笔记文件是否存在
            possible_notes = [
                NOTE_DIR / f"{note_key}_超精读笔记_已填充.md",
                NOTE_DIR / f"{note_key}_超精读笔记.md",
                NOTE_DIR / f"{note_key}.md",
            ]
            for note_path in possible_notes:
                if note_path.exists():
                    matched_note = note_path
                    break
            break

    if matched_note:
        matched.append((pdf_name, matched_note.name))
    else:
        unmatched_pdfs.append(pdf_name)

# 检查是否有笔记没有对应PDF
all_matched_notes = set([n for _, n in matched])
unmatched_notes = []
for note in note_files:
    if note.name not in all_matched_notes:
        unmatched_notes.append(note.name)

print("\n" + "="*60)
print("PDF与笔记对应关系检查结果")
print("="*60)
print(f"\n总PDF数: {len(pdf_files)}")
print(f"总笔记数: {len(note_files)}")
print(f"已匹配: {len(matched)}")
print(f"未匹配PDF: {len(unmatched_pdfs)}")
print(f"未匹配笔记: {len(unmatched_notes)}")

print("\n" + "-"*60)
print("已匹配示例 (前15个):")
print("-"*60)
for pdf, note in matched[:15]:
    print(f"  {pdf[:50]:<50} <-> {note[:40]}")

if len(unmatched_pdfs) > 0:
    print("\n" + "-"*60)
    print(f"未匹配的PDF ({len(unmatched_pdfs)}个):")
    print("-"*60)
    for pdf in unmatched_pdfs[:20]:
        print(f"  - {pdf}")

if len(unmatched_notes) > 0:
    print("\n" + "-"*60)
    print(f"未匹配的笔记 ({len(unmatched_notes)}个):")
    print("-"*60)
    for note in unmatched_notes[:20]:
        print(f"  - {note}")

# 检查笔记质量 - 抽样检查已填充笔记
print("\n" + "="*60)
print("笔记质量抽样检查 (已填充笔记)")
print("="*60)

filled_notes = [n for n in note_files if "已填充" in n.name]
print(f"\n已填充笔记数: {len(filled_notes)}")

# 检查5个已填充笔记的内容质量
print("\n抽样检查5个已填充笔记的结构:")
sample_notes = filled_notes[:5]
for note in sample_notes:
    content = note.read_text(encoding="utf-8")
    has_math = "## 🔢 1. 数学家Agent" in content or "### 1.1 核心数学框架" in content
    has_engineer = "## 🔧 2. 工程师Agent" in content or "### 2.1 算法架构" in content
    has_app = "## 💼 3. 应用专家Agent" in content
    has_skeptic = "## 🤨 4. 质疑者Agent" in content
    has_summary = "## 🎯 5. 综合理解" in content
    has_code = "```python" in content
    has_formula = "$$" in content

    quality_score = sum([has_math, has_engineer, has_app, has_skeptic, has_summary, has_code, has_formula])
    status = "✅ 优秀" if quality_score >= 6 else "⚠️ 一般" if quality_score >= 4 else "❌ 差"

    print(f"\n  {note.name[:50]}")
    print(f"    状态: {status} | 质量分: {quality_score}/7")
    print(f"    数学家:{has_math} 工程师:{has_engineer} 应用:{has_app} 质疑:{has_skeptic} 总结:{has_summary} 代码:{has_code} 公式:{has_formula}")
