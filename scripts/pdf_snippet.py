import sys, os
from typing import List

def main():
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        print("ERROR: PyPDF2 not installed.")
        sys.exit(2)
    if len(sys.argv) < 2:
        print("Usage: python scripts/pdf_snippet.py <pdf_path>")
        sys.exit(1)
    p = sys.argv[1]
    if not os.path.exists(p):
        print("MISSING")
        sys.exit(0)
    reader = PdfReader(p)
    text = "\n".join(page.extract_text() or '' for page in reader.pages)
    keys = ["COCO", "coco", "VOC", "split", "class", "increment", "base"]
    lines = [l for l in text.splitlines() if any(k in l for k in keys)]
    out_path = os.path.join(os.path.dirname(__file__), 'pdf_snippet_out.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')
    # Print ASCII-only preview for terminals with limited encodings
    for l in lines[:80]:
        s = l.encode('ascii', errors='ignore').decode('ascii', errors='ignore')
        print(s)
    print(f"\n[Wrote full UTF-8 lines to {out_path}]")

if __name__ == '__main__':
    main()
