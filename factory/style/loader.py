import json
import re
import zipfile
from pathlib import Path

from ..utils import normalize_space


def load_references(paths, max_refs=20):
    items = []
    for raw in paths or []:
        if not raw:
            continue
        text = _read_one(raw)
        if text:
            items.append({"source": str(raw), "text": text})
        if len(items) >= max_refs:
            break
    texts = []
    for item in items:
        refs = _split_reference_texts(item["text"])
        for ref in refs:
            item["refs"] = refs
            texts.append(ref)
            if len(texts) >= max_refs:
                break
        if len(texts) >= max_refs:
            break
    return texts[:max_refs], items


def _read_one(raw):
    text = str(raw).strip()
    if not text:
        return ""
    p = Path(text)
    if p.exists() and p.is_file():
        return _read_file(p)
    if len(text) > 40 or any(ch in text for ch in "，。！？,!?"):
        return normalize_space(text)
    return ""


def _read_file(p):
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".text"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            return "\n\n".join(str(x) for x in data)
        if isinstance(data, dict):
            refs = data.get("references") or data.get("scripts") or data.get("texts") or []
            if refs:
                return "\n\n".join(str(x) for x in refs)
            for key in ("text", "script", "content", "copy"):
                if key in data:
                    return str(data[key])
            return json.dumps(data, ensure_ascii=False)
    if suffix == ".docx":
        return _read_docx(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _read_docx(p):
    with zipfile.ZipFile(str(p)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", "replace")
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>|<w:p/>", xml, re.S)
    lines = []
    for para in paragraphs:
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        if text.strip():
            lines.append(text.strip())
    return "\n".join(lines)


def _split_reference_texts(text):
    text = normalize_space(text)
    chunks = re.split(r"\n{2,}", text)
    chunks = [normalize_space(c) for c in chunks if normalize_space(c)]
    if len(chunks) <= 1:
        chunks = [normalize_space(c) for c in re.split(r"(?<=[。！？!?])\s{2,}|(?<=[。！？!?])", text) if normalize_space(c)]
        chunks = [normalize_space(c) for c in chunks if len(normalize_space(c)) >= 20]
    return chunks[:20]
