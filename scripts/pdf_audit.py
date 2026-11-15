# scripts/pdf_audit.py
import re
import sys
import collections
from pypdf import PdfReader

if len(sys.argv) < 2:
    print("Usage: python scripts/pdf_audit.py <path_to_pdf>")
    sys.exit(1)

OP = re.compile(r"1 0 0 1 ([\d.\-]+) ([\d.\-]+) Tm\s*/\w+ [\d.]+ Tf[^\[]*\[<([0-9a-f]+)>\]TJ")
c = collections.Counter()
reader = PdfReader(sys.argv[1])
for i, pg in enumerate(reader.pages):
    contents = pg.get_contents()
    if contents is None:
        continue
    data = contents.get_data().decode("latin-1")
    for x, y, _ in OP.findall(data):
        c[(i, x, y)] += 1

total_draws = sum(c.values())
positions = len(c)
max_draw = max(c.values()) if c else 0
print(f"draws {total_draws} positions {positions} max {max_draw}")
