import os
import json
from pathlib import Path

# Read the data.js file
with open('D:/Documents/zx/visualizer_complete/data.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract papers array
start = content.find('papers: [')
end = content.find('],', start) + 1
papers_str = content[start:end]
papers_str = papers_str.replace('papers: [', '[').replace('],', ']').rstrip(',')

# Parse papers
papers = eval(papers_str)

pdf_dir = Path('D:/Documents/zx/docs/00_papers')
notes_dir = Path('D:/Documents/zx/visualizer_complete/notes')

missing_pdfs = []
missing_notes = []
mismatched_notes = []

for paper in papers:
    pdf_file = paper.get('pdfFile')
    note_file = paper.get('noteFile')
    
    # Check PDF
    if pdf_file:
        pdf_path = pdf_dir / pdf_file
        if not pdf_path.exists():
            missing_pdfs.append({
                'id': paper['id'],
                'title': paper['title'],
                'pdfFile': pdf_file
            })
    
    # Check note
    if note_file:
        note_path = notes_dir / note_file
        if not note_path.exists():
            missing_notes.append({
                'id': paper['id'],
                'title': paper['title'],
                'noteFile': note_file
            })
        else:
            # Check for LaTeX formulas
            try:
                with open(note_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '$$' in content:
                        mismatched_notes.append({
                            'id': paper['id'],
                            'title': paper['title'],
                            'noteFile': note_file,
                            'has_latex': True
                        })
            except:
                pass

print(f"\n=== SUMMARY ===")
print(f"Total papers: {len(papers)}")
print(f"Missing PDFs: {len(missing_pdfs)}")
print(f"Missing notes: {len(missing_notes)}")
print(f"Notes with LaTeX: {len(mismatched_notes)}")

print("\n=== MISSING PDFS ===")
for item in missing_pdfs[:10]:  # Show first 10
    print(f"ID {item['id']}: {item['title']} - {item['pdfFile']}")

if len(missing_pdfs) > 10:
    print(f"... and {len(missing_pdfs) - 10} more")

print("\n=== MISSING NOTES ===")
for item in missing_notes[:10]:  # Show first 10
    print(f"ID {item['id']}: {item['title']} - {item['noteFile']}")

if len(missing_notes) > 10:
    print(f"... and {len(missing_notes) - 10} more")

print("\n=== NOTES WITH LATEX ===")
for item in mismatched_notes[:10]:  # Show first 10
    print(f"ID {item['id']}: {item['title']} - {item['noteFile']}")

if len(mismatched_notes) > 10:
    print(f"... and {len(mismatched_notes) - 10} more")
