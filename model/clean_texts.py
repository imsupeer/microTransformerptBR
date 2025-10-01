import re, sys, pathlib

src = pathlib.Path("data/raw")
dst = pathlib.Path("data/clean")
dst.mkdir(parents=True, exist_ok=True)
pat_space = re.compile(r"\s+")
for f in src.glob("**/*.txt"):
    txt = f.read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("\r", "")
    txt = pat_space.sub(" ", txt).strip()
    if len(txt) > 0:
        (dst / f.name).write_text(txt, encoding="utf-8")
