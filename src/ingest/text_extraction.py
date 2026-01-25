
import os, re, json, uuid
import warnings


from typing                         import List, Dict, Optional
from langchain_text_splitters       import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters.base  import Language
from dotenv                         import load_dotenv
from rich                           import print

import pymupdf.layout
import pymupdf4llm

load_dotenv()

warnings.filterwarnings('ignore')

RE_CAPITULO  = re.compile(r"^\s*##\s*\*\*(.+?)\*\*\s*$")
RE_ARTIGO    = re.compile(r"^\s*\*\*(Artigo\s+\d+º?)\s*[-–—]?\*\*\s*(.*)\s*$", re.IGNORECASE)
RE_PAR_UNICO = re.compile(r"^\s*\*\*(Parágrafo\s+Único\.)\*\*\s*(.*)\s*$", re.IGNORECASE)
RE_PARAG     = re.compile(r"^\s*\*\*(§\s*\d+º?)\*\*\s*[-–—]?\s*(.*)\s*$", re.IGNORECASE)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
file_name  = os.path.join(BASE_DIR, 'pdf', 'Nouveaux_Regulamento_2023.pdf')
#full_text  = pymupdf4llm.to_markdown(file_name, header=False, footer=False, pages=[2,3])
full_text  = pymupdf4llm.to_markdown(file_name, header=False, footer=False)

#print( full_text )

def split_normativo(md_text: str) -> List[Dict]:

    print( '--> inicio' )
    
    capitulo_nome:  Optional[str] = None
    artigo_nr:      Optional[str] = None
    paragrafo_id:   Optional[str] = None
    tipo:           Optional[str] = None
    parent_id:      str = ""
    chunk:          str = "" 
    overlap:        str = "" 
    buffer:         List[str] = []
    chunks:         List[Dict] = []   

    def flush():

        nonlocal buffer, paragrafo_id, capitulo_nome, artigo_nr, chunks, tipo, parent_id, chunk, overlap
        
        if paragrafo_id and buffer:

            text = "\n".join(buffer).strip()            
            if text:
                chunks.append({
                    "text": text,
                    "capitulo": capitulo_nome,
                    "artigo": artigo_nr,
                    "paragrafo": paragrafo_id,
                    "tipo":"full_text",
                    "parent_id":parent_id
                })

            if len(chunk)<1000:
                chunk = f'{overlap}{chunk}{text}'

            if len(chunk)>1000:
                chunks.append({
                    "text": chunk,
                    "capitulo": capitulo_nome,
                    "artigo": "",
                    "paragrafo": "",
                    "tipo":"chunk",
                    "id":parent_id
                })

                parent_id = uuid.uuid4().hex[:8]
                overlap   = text  
                chunk     = ''                     
   
        buffer = []
        paragrafo_id = None

    lines = md_text.splitlines()
    parent_id = uuid.uuid4().hex[:8]

    for line in lines:
        
        l = line.strip()
        if not l:
            # mantém quebra dentro do parágrafo como espaço (evita fragmentar)
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue

        # 1) Capítulo / Seção
        m = RE_CAPITULO.match(l)
        if m:
            flush()
            capitulo_nome = m.group(1).strip()
            continue

        # 2) Artigo
        m = RE_ARTIGO.match(l)
        if m:
            flush()
            artigo_nr = m.group(1).strip()  # "Artigo 4º"
            # Se a mesma linha já tem conteúdo, vira "caput" do artigo.
            rest = m.group(2).strip()
            if rest:
                paragrafo_id = "caput"
                buffer = [f"**{artigo_nr} -** {rest}"]
            continue

        # 3) Parágrafo Único
        m = RE_PAR_UNICO.match(l)
        if m:
            flush()
            paragrafo_id = "Parágrafo Único"
            rest = m.group(2).strip()
            buffer = [f"**{m.group(1)}** {rest}".strip()]
            continue

        # 4) §1º, §2º ...
        m = RE_PARAG.match(l)
        if m:
            flush()
            paragrafo_id = m.group(1).replace(" ", "")  # "§1º"
            rest = m.group(2).strip()
            buffer = [f"**{paragrafo_id}** - {rest}".strip()]
            continue

        # 5) Continuação de texto (pertence ao parágrafo atual)
        if paragrafo_id:
            # cola linhas “quebradas” como uma linha contínua
            if buffer and buffer[-1] != "":
                buffer[-1] = buffer[-1].rstrip() + " " + l
            else:
                buffer.append(l)
        else:
            # Texto fora de parágrafo: pode ser introdução antes do 1º artigo.
            # Decide: ignorar ou guardar como chunk "preâmbulo".
            # Aqui vou guardar como "preâmbulo" dentro do capítulo atual.
            paragrafo_id = "preambulo"
            buffer = [l]
    
    flush()
    return chunks

chunks = split_normativo( full_text )

print( '-> fim' )

out_path = "output/chunks_normativos.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"-> OK: {len(chunks)} chunks salvos em {out_path}")
