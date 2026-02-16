import os
import hashlib
import re
from pathlib import Path
from datetime import datetime

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

BASE_DIR = Path(r"D:\Documents\zx")

AUTHOR_PATTERNS = [
    r"Xiaohao\s+Cai",
    r"X\.\s*Cai",
    r"x\.cai@soton",
    r"xcai@soton",
    r"Xiaohao Cai",
]

def extract_text_from_pdf(pdf_path):
    if HAS_FITZ:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text[:5000]
        except Exception as e:
            return None
    
    if HAS_MARKITDOWN:
        try:
            md = MarkItDown()
            result = md.convert(str(pdf_path))
            return result.text_content[:5000]
        except Exception as e:
            return None
    
    return None

def check_author(text):
    if not text:
        return None, "无法提取文本"
    
    text_lower = text.lower()
    found_patterns = []
    
    for pattern in AUTHOR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found_patterns.append(pattern)
    
    if found_patterns:
        return True, found_patterns
    return False, []

def get_file_hash(filepath):
    try:
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except:
        return None

def extract_arxiv_id(filename):
    patterns = [
        r'(\d{4}\.\d{4,5})',
        r'(\d{2}\d{2}\.\d{4,5})',
    ]
    for p in patterns:
        m = re.search(p, filename)
        if m:
            return m.group(1)
    return None

def scan_directory(directory):
    pdfs = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(Path(root) / f)
    return pdfs

def main():
    directories = [
        "xiaohao_cai_papers_final",
        "xiaohao_cai_clean",
    ]
    
    all_pdfs = []
    for d in directories:
        dir_path = BASE_DIR / d
        if dir_path.exists():
            pdfs = scan_directory(dir_path)
            all_pdfs.extend(pdfs)
    
    for pdf_name in ["xiaohao_cai_evolution.pdf", "xiaohao_cai_network.pdf", 
                     "xiaohao_cai_timeline.pdf", "xiaohao_cai_topics.pdf", 
                     "xiaohao_cai_venues.pdf"]:
        pdf_path = BASE_DIR / pdf_name
        if pdf_path.exists():
            all_pdfs.append(pdf_path)
    
    all_pdfs = list(set(all_pdfs))
    
    results = {
        'cai_papers': [],
        'not_cai_papers': [],
        'corrupted': [],
        'errors': [],
    }
    
    hash_map = {}
    arxiv_map = {}
    
    print(f"开始扫描 {len(all_pdfs)} 个PDF文件...")
    print("=" * 60)
    
    for i, pdf_path in enumerate(all_pdfs, 1):
        filename = pdf_path.name
        print(f"[{i}/{len(all_pdfs)}] 检查: {filename[:50]}...")
        
        file_hash = get_file_hash(pdf_path)
        if file_hash:
            if file_hash in hash_map:
                hash_map[file_hash].append(pdf_path)
            else:
                hash_map[file_hash] = [pdf_path]
        
        arxiv_id = extract_arxiv_id(filename)
        if arxiv_id:
            if arxiv_id in arxiv_map:
                arxiv_map[arxiv_id].append(pdf_path)
            else:
                arxiv_map[arxiv_id] = [pdf_path]
        
        text = extract_text_from_pdf(pdf_path)
        
        if text is None:
            results['errors'].append({
                'file': str(pdf_path),
                'reason': '无法读取或提取文本'
            })
            results['corrupted'].append(str(pdf_path))
            continue
        
        is_cai, info = check_author(text)
        
        if is_cai:
            results['cai_papers'].append({
                'file': str(pdf_path),
                'patterns': info
            })
        else:
            if info == "无法提取文本":
                results['corrupted'].append(str(pdf_path))
            else:
                results['not_cai_papers'].append({
                    'file': str(pdf_path),
                    'reason': '未找到作者标识'
                })
    
    duplicates_by_hash = {k: v for k, v in hash_map.items() if len(v) > 1}
    duplicates_by_arxiv = {k: v for k, v in arxiv_map.items() if len(v) > 1}
    
    print("\n" + "=" * 60)
    print("扫描完成!")
    print("=" * 60)
    
    print(f"\n【汇总统计】")
    print(f"  总PDF数量: {len(all_pdfs)}")
    print(f"  是Cai论文: {len(results['cai_papers'])}")
    print(f"  不是Cai论文: {len(results['not_cai_papers'])}")
    print(f"  损坏/无法读取: {len(results['corrupted'])}")
    
    cai_arxiv_ids = set()
    for p in results['cai_papers']:
        aid = extract_arxiv_id(Path(p['file']).name)
        if aid:
            cai_arxiv_ids.add(aid)
    
    print(f"  去重后有效论文(arXiv ID): {len(cai_arxiv_ids)}")
    
    if results['not_cai_papers']:
        print(f"\n【不是Cai的论文】({len(results['not_cai_papers'])}个)")
        for p in results['not_cai_papers']:
            print(f"  - {Path(p['file']).name}")
    
    if results['corrupted']:
        print(f"\n【损坏/无法读取的文件】({len(results['corrupted'])}个)")
        for f in results['corrupted']:
            print(f"  - {Path(f).name}")
    
    if duplicates_by_hash:
        print(f"\n【内容重复(相同MD5)】({len(duplicates_by_hash)}组)")
        for h, files in duplicates_by_hash.items():
            print(f"  MD5: {h[:16]}...")
            for f in files:
                print(f"    - {f.relative_to(BASE_DIR)}")
    
    if duplicates_by_arxiv:
        print(f"\n【相同arXiv ID的文件】({len(duplicates_by_arxiv)}组)")
        for aid, files in duplicates_by_arxiv.items():
            print(f"  arXiv: {aid}")
            for f in files:
                print(f"    - {f.relative_to(BASE_DIR)}")
    
    report = f"""
# Xiaohao Cai论文验证报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 汇总统计
- **总PDF数量**: {len(all_pdfs)}
- **是Cai论文**: {len(results['cai_papers'])}
- **不是Cai论文**: {len(results['not_cai_papers'])}
- **损坏/无法读取**: {len(results['corrupted'])}
- **去重后有效论文**: {len(cai_arxiv_ids)}

## 不是Cai的论文列表
"""
    for p in results['not_cai_papers']:
        report += f"- {Path(p['file']).name}\n"
    
    report += "\n## 损坏文件列表\n"
    for f in results['corrupted']:
        report += f"- {Path(f).name}\n"
    
    report += "\n## 重复文件(按内容)\n"
    for h, files in duplicates_by_hash.items():
        report += f"### MD5: {h}\n"
        for f in files:
            report += f"  - {f.relative_to(BASE_DIR)}\n"
    
    report += "\n## 重复文件(按arXiv ID)\n"
    for aid, files in duplicates_by_arxiv.items():
        report += f"### arXiv: {aid}\n"
        for f in files:
            report += f"  - {f.relative_to(BASE_DIR)}\n"
    
    report_path = BASE_DIR / "paper_verification_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n详细报告已保存到: {report_path}")

if __name__ == "__main__":
    main()
