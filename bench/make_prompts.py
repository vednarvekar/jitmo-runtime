"""Build the 'real prompt' files used for the memory runs.

Source text: the project's own docs (docs/*.md) followed by llama.cpp's LICENSE-LLVM-OpenMP file
(real, non-repeated English text). Each prompt = first N characters + a closing instruction.
Usage: python -I make_prompts.py <out_dir> <chars_2k> <chars_8k>
"""
import glob
import os
import sys

out, c2, c8 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
parts = []
for p in sorted(glob.glob(os.path.join(root, "docs", "*.md"))):
    parts.append(open(p, encoding="utf-8").read())
parts.append(open(r"V:\jitmo-sandbox\llama\LICENSE-LLVM-OpenMP", encoding="utf-8", errors="replace").read())
src = "\n\n".join(parts)
tail = "\n\n---\nTask: summarize the document above in five short bullet points, then list three open questions it raises.\n"
os.makedirs(out, exist_ok=True)
for name, n in (("prompt_2k.txt", c2), ("prompt_8k.txt", c8)):
    assert n <= len(src), (n, len(src))
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write(src[:n] + tail)
    print(name, n, "chars")
print("source chars", len(src))
