import json,re,unicodedata, uuid
from rich import print

import requests
from requests.auth import HTTPBasicAuth

from src_en.config import settings

# py -m src_en.ingest.json_insert_chapter

'''Explixação:

'''

GRAPHDB_BASE_URL   = settings.GRAPHDB_BASE_URL
GRAPHDB_USERNAME   = settings.GRAPHDB_USERNAME
GRAPHDB_PASSWORD   = settings.GRAPHDB_PASSWORD
GRAPHDB_REPOSITORY = settings.repositorio
GRAPHDB_REPO_URL   = f"{GRAPHDB_BASE_URL}/repositories/{GRAPHDB_REPOSITORY}/statements"

def normalize(texto):  
    
    s = (texto or "").strip().lower()
    s = s.replace("º", "").replace("ª", "").replace("°", "")
    s = s.replace("§", "paragrafo_")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s_]", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)

    return s

def no_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def upload(ttl,path,USUARIO):

    try:

        params  = {"context": f"<https://omc.co/graph/{USUARIO}>"}    
        headers = {"Content-Type": "text/turtle"}

        auth = HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD) if (GRAPHDB_USERNAME and GRAPHDB_PASSWORD) else None
        resp = requests.post(GRAPHDB_REPO_URL, params=params, data=ttl, headers=headers, auth=auth, timeout=1200)
        resp.raise_for_status()

        print( resp, path )

    except Exception as erro:
        print( f'ERRO upload: {path} - {erro}' )
    
    return True 

with open("src_en/ingest/output/chunks_normativos_gpdr_v1.json", encoding="utf-8") as f:
    json_chunks = json.loads(f.read())

full_text = [item for item in json_chunks if item.get("tipo") == "full_text"]
ttl_path  = "src_en/ingest/output/normativos_gdpr_chapter.ttl"
ttl_lines = []

for index, item in enumerate(full_text,start=1):     
    
    capitulo  = normalize(item['capitulo'])
    artigo    = normalize(item['artigo'])
    paragrafo = normalize(item['paragrafo'])
    text      = no_spaces(item['text'])
    
    if paragrafo == 'caput':     
             
        tipo            =  'Chapter' 
        id              = item['id'] 
        chapter         = item['capitulo']  
        arr_chapter     = chapter.split(' ')
        chapter_number  = arr_chapter[1] 
        chapter_uri     = f'Chapter_{chapter_number}'
        chapter_title   = ' '.join(arr_chapter[2:])
        path            = f"{id}"

        rdf = f'''<https://omc.co/5511993891773/7492/{chapter_uri}>
        a v:{tipo} ; 
        v:chapterNumber "{chapter_number}" ; 
        v:title  "{chapter_title}"@en ;
        v:is_part_of <https://omc.co/5511993891773/7492> .'''

        upload(rdf,path,'5511993891773')

        ttl_lines.append(rdf)

with open(ttl_path, "w", encoding="utf-8") as f:
    f.write("".join(ttl_lines))

print("OK:", ttl_path)
print("Itens full_text:", len(ttl_lines))

'''
<https://omc.co/5511993891773/749/entidade/Artigo1>
  a v:Artigo ;
  v:descricao "Define obrigatoriedade de cumprimento do Regulamento Interno por condôminos"@pt ;
  v:temEvidenciaEm <https://omc.co/5511993891773/749/c/5ec8059f> .

<https://omc.co/5511993891773/749/entidade/ParagrafoUnico1>
  a v:ParagrafoUnico ;
  v:descricao "Empregados/Prestadores são responsabilidade do condomínio..."@pt ;
  v:temEvidenciaEm <https://omc.co/5511993891773/749/c/5ec8059f> .
  '''

'''{
    "text": "**Artigo 1º -** Todos os proprietários de unidades, promitentes compradores, cessionários e promitentes cessionários, atuais e futuros, ocupantes e locatários, doravante denominados comum e genericamente **CONDÔMINOS/CONDÔMINO** ficam obrigados a cumprir e fazer cumprir as determinações constantes do presente Regulamento Interno.",
    "capitulo": "CAPÍTULO I – DAS DISPOSIÇÕES GERAIS",
    "artigo": "Artigo 1º",
    "paragrafo": "caput",
    "tipo": "full_text",
    "parent_id": "eadea72c"
  },'''