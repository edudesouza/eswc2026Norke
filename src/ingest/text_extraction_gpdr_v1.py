import json
import os
import re
import uuid
import warnings

from typing import Dict, List, Optional

from dotenv import load_dotenv
from rich import print

import pymupdf4llm

load_dotenv()
warnings.filterwarnings("ignore")

RE_CAPITULO     = re.compile(r"^(?:##\s+|\*\*)(CHAPTER\s+[IVXLCDM]+\s+.+?)(?:\*\*)?$",re.IGNORECASE,)
RE_ARTIGO       = re.compile(r"^(Article\s+\d+)\s+(.+)$")
RE_ITEM         = re.compile(r"^(\d+)\.\s+(.*)$")
RE_FOOTNOTE     = re.compile(r"^\[\d+")
RE_PAGE_NUMBER  = re.compile(r"^\d+$")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "pdf", "GPDR_full.pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def should_skip_line(line: str) -> bool:
    if not line:
        return True
    if RE_PAGE_NUMBER.match(line):
        return True
    if RE_FOOTNOTE.match(line):
        return True
    return False


def split_normativo(md_text: str) -> List[Dict]:
    print("--> inicio")

    started = False
    capitulo_nome: Optional[str] = None
    artigo_nr: Optional[str] = None
    artigo_titulo: Optional[str] = None
    artigo_id: Optional[str] = None
    item_id: Optional[str] = None
    buffer: List[str] = []
    chunks: List[Dict] = []
    artigo_itens: List[str] = []
    artigo_item_entries: List[Dict] = []
    article_header_text: Optional[str] = None
    chunk_parts: List[str] = []
    chunk_len = 0
    chunk_start_capitulo: Optional[str] = None
    chunk_start_artigo: Optional[str] = None
    chunk_start_titulo: Optional[str] = None

    def build_chunk_text(parts: List[str]) -> str:
        return " ".join(part.strip() for part in parts if part.strip())

    def add_chunk_part(text: str, *, is_item_end: bool) -> None:
        nonlocal chunk_parts, chunk_len
        nonlocal chunk_start_capitulo, chunk_start_artigo, chunk_start_titulo

        clean_text = text.strip()
        if not clean_text:
            return

        if not chunk_parts:
            chunk_start_capitulo = capitulo_nome
            chunk_start_artigo = artigo_nr
            chunk_start_titulo = artigo_titulo

        chunk_parts.append(clean_text)
        chunk_len = len(build_chunk_text(chunk_parts))

        if is_item_end and chunk_len > 900:
            emit_chunk()

    def emit_chunk() -> None:
        nonlocal chunk_parts, chunk_len
        nonlocal chunk_start_capitulo, chunk_start_artigo, chunk_start_titulo

        chunk_text = build_chunk_text(chunk_parts)
        if not chunk_text:
            chunk_parts = []
            chunk_len = 0
            chunk_start_capitulo = None
            chunk_start_artigo = None
            chunk_start_titulo = None
            return

        chunks.append(
            {
                "text": chunk_text,
                "capitulo": chunk_start_capitulo,
                "artigo": chunk_start_artigo or "",
                "titulo_artigo": chunk_start_titulo or "",
                "paragrafo": "",
                "tipo": "chunk",
            }
        )

        chunk_parts = []
        chunk_len = 0
        chunk_start_capitulo = None
        chunk_start_artigo = None
        chunk_start_titulo = None

    def flush_item() -> None:
        nonlocal buffer, item_id, artigo_itens, artigo_item_entries

        if not item_id or not buffer:
            buffer = []
            item_id = None
            return

        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            item_id = None
            return

        artigo_itens.append(text)
        artigo_item_entries.append(
            {
                "text": text,
                "capitulo": capitulo_nome,
                "artigo": artigo_nr,
                "titulo_artigo": artigo_titulo,
                "paragrafo": item_id,
                "tipo": "full_text",
                "parent_id": artigo_id,
            }
        )
        add_chunk_part(text, is_item_end=True)

        buffer = []
        item_id = None

    def flush_article() -> None:
        nonlocal artigo_nr, artigo_titulo, artigo_id, artigo_itens, artigo_item_entries, article_header_text

        flush_item()

        if not artigo_nr or not artigo_id or not article_header_text:
            artigo_nr = None
            artigo_titulo = None
            artigo_id = None
            artigo_itens = []
            article_header_text = None
            return

        article_parts = [article_header_text] + artigo_itens
        article_text = "\n".join(part for part in article_parts if part).strip()

        chunks.append(
            {
                "text": article_text,
                "capitulo": capitulo_nome,
                "artigo": artigo_nr,
                "titulo_artigo": artigo_titulo,
                "paragrafo": "caput",
                "tipo": "full_text",
                "id": artigo_id,
            }
        )
        chunks.extend(artigo_item_entries)

        artigo_nr = None
        artigo_titulo = None
        artigo_id = None
        artigo_itens = []
        artigo_item_entries = []
        article_header_text = None

    lines = md_text.splitlines()

    for line in lines:
        l = line.strip()

        if should_skip_line(l):
            continue

        m = RE_CAPITULO.match(l)
        if m:
            started = True
            flush_article()
            capitulo_nome = m.group(1).strip()
            continue

        if not started:
            continue

        m = RE_ARTIGO.match(l)
        if m:
            flush_article()
            artigo_nr = m.group(1).strip()
            artigo_titulo = m.group(2).strip()
            artigo_id = uuid.uuid4().hex[:8]
            artigo_itens = []
            artigo_item_entries = []
            article_header_text = f"{artigo_nr} {artigo_titulo}"
            add_chunk_part(article_header_text, is_item_end=False)
            continue

        m = RE_ITEM.match(l)
        if m and artigo_nr:
            flush_item()
            item_id = m.group(1).strip()
            buffer = [f"{item_id}. {m.group(2).strip()}"]
            continue

        if item_id:
            if buffer and buffer[-1] != "":
                if re.match(r"^\([a-z]\)", l, re.IGNORECASE):
                    buffer.append(l)
                else:
                    buffer[-1] = buffer[-1].rstrip() + " " + l
            else:
                buffer.append(l)

    flush_article()

    final_chunk = build_chunk_text(chunk_parts)
    if final_chunk:
        emit_chunk()

    return chunks


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_text = pymupdf4llm.to_markdown(PDF_PATH)

    txt_path = os.path.join(OUTPUT_DIR, "normativos_gpdr_v1.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"-> OK: texto extraido salvo em {txt_path}")

    chunks = split_normativo(full_text)

    print("-> fim")

    out_path = os.path.join(OUTPUT_DIR, "chunks_normativos_gpdr_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"-> OK: {len(chunks)} chunks salvos em {out_path}")


if __name__ == "__main__":
    main()
