import os, re, json, uuid
import warnings


from typing                         import List, Dict, Optional
from dotenv                         import load_dotenv
from rich                           import print

import pymupdf4llm

load_dotenv()

warnings.filterwarnings('ignore')

# Regex para texto plano do PDF LGPD
RE_CAPITULO  = re.compile(r"^(CAPÍTULO|TÍTULO|SEÇÃO|DISPOSIÇÕES|LIVRO)\s+[A-ZIVX]+")
#RE_ARTIGO    = re.compile(r"^(Art\.\s+\d+º?)\s+(.*)")
RE_ARTIGO = re.compile(r"^(Art\.\s+\d+º?)\.?\s+(.*)")
RE_PAR_UNICO = re.compile(r"^(Parágrafo\s+único\.)\s*(.*)", re.IGNORECASE)
RE_PARAG     = re.compile(r"^(§\s*\d+º?)\s+(.*)")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
file_name  = os.path.join(BASE_DIR, 'pdf', 'LGPD.pdf')
full_text  = pymupdf4llm.to_markdown(file_name, header=False, footer=False)

print( full_text )

def split_normativo(md_text: str) -> List[Dict]:

    print( '--> inicio' )

    capitulo_nome:  Optional[str] = None
    artigo_nr:      Optional[str] = None
    paragrafo_id:   Optional[str] = None
    parent_id:      str = ""
    chunk:          str = ""
    overlap:        str = ""
    buffer:         List[str] = []
    chunks:         List[Dict] = []

    def flush():

        nonlocal buffer, paragrafo_id, capitulo_nome, artigo_nr, chunks, parent_id, chunk, overlap

        if paragrafo_id and buffer:

            text = "\n".join(buffer).strip()

            if text:

                if paragrafo_id == "caput":
                    parent_id = uuid.uuid4().hex[:8]
                    chunks.append({
                        "text": text,
                        "capitulo": capitulo_nome,
                        "artigo": artigo_nr,
                        "paragrafo": paragrafo_id,
                        "tipo": "full_text",
                        "id": parent_id
                    })
                else:
                    chunks.append({
                        "text": text,
                        "capitulo": capitulo_nome,
                        "artigo": artigo_nr,
                        "paragrafo": paragrafo_id,
                        "tipo": "full_text",
                        "parent_id": parent_id
                    })

            if len(chunk) < 1000:
                chunk = f'{overlap}{chunk}{text}'

            if len(chunk) > 1000:
                chunks.append({
                    "text": chunk,
                    "capitulo": capitulo_nome,
                    "artigo": "",
                    "paragrafo": "",
                    "tipo": "chunk",
                    "parent_id": parent_id
                })

                overlap = text
                chunk = ''

        buffer = []
        paragrafo_id = None

    lines = md_text.splitlines()
    parent_id = uuid.uuid4().hex[:8]

    for line in lines:
        l = line.strip()
        if not l:
            # Linha vazia: adiciona apenas se já estiver processando um parágrafo
            if paragrafo_id and buffer:
                buffer.append("")
            continue

        # 1) Capítulo / Seção
        m = RE_CAPITULO.match(l)
        if m:
            flush()
            capitulo_nome = m.group(0).strip()
            continue

        # 2) Artigo (o caput vai até o próximo Art. ou até o primeiro §)
        m = RE_ARTIGO.match(l)
        if m:
            flush()
            artigo_nr = m.group(1).strip()  # "Art. 4º"
            rest = m.group(2).strip()
            paragrafo_id = "caput"
            if rest:
                buffer = [f"{artigo_nr} {rest}"]
            else:
                buffer = [artigo_nr]
            continue

        # 3) Parágrafo Único
        m = RE_PAR_UNICO.match(l)
        if m:
            flush()
            paragrafo_id = "Parágrafo único"
            rest = m.group(2).strip()
            buffer = [f"{m.group(1)} {rest}".strip()]
            continue

        # 4) §1º, §2º ...
        m = RE_PARAG.match(l)
        if m:
            flush()
            paragrafo_id = re.sub(r"\s+$", "", m.group(1))
            rest = m.group(2).strip()
            buffer = [f"{paragrafo_id} {rest}".strip()]
            continue

        # 5) Continuação de texto (pertence ao parágrafo atual)
        if paragrafo_id:
            # Cola linhas quebradas como linha contínua, mantém listas separadas
            if buffer and buffer[-1] != "":
                # Se a linha começa com - ou número de inciso, mantém separado
                if l.startswith("-") or l.startswith(("I -", "II -", "III -", "IV -", "V -", "VI -", "VII -", "VIII -", "IX -", "X -", "a)", "b)", "c)", "d)")):
                    buffer.append(l)
                else:
                    buffer[-1] = buffer[-1].rstrip() + " " + l
            else:
                buffer.append(l)
        else:
            # Texto fora de parágrafo: preâmbulo
            if capitulo_nome or not buffer:
                paragrafo_id = "preambulo"
                buffer.append(l)

    flush()
    return chunks

chunks = split_normativo( full_text )

print( '-> fim' )

# Salva JSON
out_path = "output/chunks_normativos_lgpd.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"-> OK: {len(chunks)} chunks salvos em {out_path}")

# Salva texto extraído em TXT para análise e auditoria
txt_path = "output/normativos_lgpd.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"-> OK: texto extraído salvo em {txt_path}")