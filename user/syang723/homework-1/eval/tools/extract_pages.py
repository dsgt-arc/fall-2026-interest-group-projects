"""Extract per-page text of every corpus PDF: uv run python extract_pages.py <data_dir> <out.json>"""
import glob, json, logging, os, sys
from pypdf import PdfReader

logging.disable(logging.CRITICAL)
out = {os.path.basename(f)[:-4]: [(p.extract_text() or "") for p in PdfReader(f).pages]
       for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.pdf")))}
json.dump(out, open(sys.argv[2], "w"))
