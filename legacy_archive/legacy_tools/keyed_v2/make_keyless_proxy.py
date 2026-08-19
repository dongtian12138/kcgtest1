#!/usr/bin/env python3
"""Create a keyless diagnostic copy of the insert proxy (strip C2Keys)."""
from pathlib import Path

src = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/isaac/d38999_insert_proxy_v2.usda")
dst = Path("/home/noob/WorkPlace/kcgtest1/artifacts/kcg_connector/isaac/d38999_insert_proxy_v2_keyless_diag.usda")
text = src.read_text(encoding="utf-8")

start_marker = '                def "C2Keys"'
si = text.find(start_marker)
if si < 0:
    raise SystemExit("C2Keys block not found")
# find the matching closing brace at the same indent (16 spaces)
i = si
depth = 0
end = None
for i in range(si, len(text)):
    if text[i] == "{":
        depth += 1
    elif text[i] == "}":
        depth -= 1
        if depth == 0:
            end = i
            break
if end is None:
    raise SystemExit("C2Keys block end not found")
# include the trailing newline
while end < len(text) and text[end] != "\n":
    end += 1
end += 1
stripped = text[:si] + text[end:]
dst.write_text(stripped, encoding="utf-8")
print("wrote", dst, "removed", end - si, "chars")
