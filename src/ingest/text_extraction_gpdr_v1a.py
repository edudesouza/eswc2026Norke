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
    item_id: Optional[str] = None
    parent_id: str = ""
    buffer: List[str] = []
    chunks: List[Dict] = []
    chunk_parts: List[str] = []
    overlap_text: str = ""

    def push_chunk(text: str) -> None:
        nonlocal chunk_parts, overlap_text, chunks

        if not text:
            return

        pending_parts = chunk_parts + [text]
        pending_text = "\n".join(pending_parts).strip()

        if len(pending_text) > 1000 and chunk_parts:
            chunk_text = "\n".join(chunk_parts).strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "capitulo": capitulo_nome,
                        "artigo": artigo_nr or "",
                        "titulo_artigo": artigo_titulo or "",
                        "paragrafo": "",
                        "tipo": "chunk",
                        "parent_id": parent_id,
                    }
                )
            chunk_parts = [overlap_text] if overlap_text else []

        if text not in chunk_parts:
            chunk_parts.append(text)
        overlap_text = text

    def flush() -> None:
        nonlocal buffer, item_id, chunks, parent_id

        if not item_id or not buffer:
            buffer = []
            item_id = None
            return

        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            item_id = None
            return

        if item_id == "caput":
            parent_id = uuid.uuid4().hex[:8]
            chunks.append(
                {
                    "text": text,
                    "capitulo": capitulo_nome,
                    "artigo": artigo_nr,
                    "titulo_artigo": artigo_titulo,
                    "paragrafo": item_id,
                    "tipo": "full_text",
                    "id": parent_id,
                }
            )
        else:
            chunks.append(
                {
                    "text": text,
                    "capitulo": capitulo_nome,
                    "artigo": artigo_nr,
                    "titulo_artigo": artigo_titulo,
                    "paragrafo": item_id,
                    "tipo": "full_text",
                    "parent_id": parent_id,
                }
            )
            push_chunk(text)

        buffer = []
        item_id = None

    lines = md_text.splitlines()

    for line in lines:
        l = line.strip()

        if should_skip_line(l):
            continue

        m = RE_CAPITULO.match(l)
        if m:
            started = True
            flush()
            capitulo_nome = m.group(1).strip()
            artigo_nr = None
            artigo_titulo = None
            parent_id = ""
            continue

        if not started:
            continue

        m = RE_ARTIGO.match(l)
        if m:
            flush()
            artigo_nr = m.group(1).strip()
            artigo_titulo = m.group(2).strip()
            item_id = "caput"
            buffer = [f"{artigo_nr} {artigo_titulo}"]
            continue

        m = RE_ITEM.match(l)
        if m and artigo_nr:
            flush()
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

    flush()

    final_chunk = "\n".join(chunk_parts).strip()
    if final_chunk:
        chunks.append(
            {
                "text": final_chunk,
                "capitulo": capitulo_nome,
                "artigo": artigo_nr or "",
                "titulo_artigo": artigo_titulo or "",
                "paragrafo": "",
                "tipo": "chunk",
                "parent_id": parent_id,
            }
        )

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
