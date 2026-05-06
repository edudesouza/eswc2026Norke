import json,re,unicodedata, uuid
from rich import print

import requests
from requests.auth import HTTPBasicAuth

# py -m src_en.ingest.json_gdpr_paragrafo

'''Explixação:

'''

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
    
    if not s:
        return ""

    # normaliza quebras
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # escapa primeiro
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')

    # agora converte quebra real → \n textual
    s = re.sub(r"\s+", " ", s)

    # limpa espaços horizontais
    s = re.sub(r"[ \t]+", " ", s)

    return s.strip()

def upload(ttl,USUARIO):

    try:
        from src_en.config import settings

        graphdb_repo_url = (
            f"{settings.GRAPHDB_BASE_URL}/repositories/"
            f"{settings.repositorio_v3}/statements"
        )

        params  = {"context": f"<https://omc.co/graph/{USUARIO}>"}    
        headers = {"Content-Type": "text/turtle"}

        auth = HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD) if (settings.GRAPHDB_USERNAME and settings.GRAPHDB_PASSWORD) else None
        resp = requests.post(graphdb_repo_url, params=params, data=ttl, headers=headers, auth=auth, timeout=1200)
        resp.raise_for_status()

        print( resp )

    except Exception as erro:
        print( f'ERRO upload: {erro}' )
        print( ttl )
    
    return True 

with open("src_en/ingest/gdpr.json", encoding="utf-8") as f:
    json_gdpr = json.load(f)

ttl_path  = "src_en/ingest/output/normativos_gdpr_paragrafo.ttl"
ttl_lines = []

for chapter in json_gdpr["chapters"]:

    chapter_uri = 'Chapter_' + chapter["number"]
    
    for item in chapter["contents"]:
        articles = []

        if item["type"] == "article":
            articles.append(item)

        if item["type"] == "section":
            articles.extend(
                content
                for content in item["contents"]
                if content["type"] == "article"
            )

        for article in articles:

            article_uri = 'Article_' + article["number"]
            
            for point_index, point in enumerate(article["contents"], start=1):

                paragraph_number = point.get("number") or f"text_{point_index}"
                paragraph_uri    = "Point_" + str(paragraph_number)

                paragraph_nr = point.get("number")
                
                text = no_spaces(point.get("text"))
                dados = text or ""

                for subpoint in point.get("subpoints") or []:
                    sub_text = no_spaces(subpoint.get("text"))

                    if len(point.get("subpoints", [])) > 0:
                        dados += f" {sub_text}"
                        paragraph_nr = point.get("number")

                tipo      = 'Point' 
                parent_id = f"chapter_{chapter['number'].lower()}_article_{article['number']}"

                rdf = f'''<https://omc.co/5511993891773/7492/{chapter_uri}/{article_uri}/{paragraph_uri}>
                a v:{tipo} ;            
                v:fullText "{dados}"@en ;
                v:breadcrumb        "chapter {chapter['number']}, article {article['number']}, point {point.get('number')}"@en ;
                v:breadcrumbLabel   "chapter {chapter['number']} ({chapter['title']}), article {article['number']} ({article['title']}), point {paragraph_nr}"@en ;
                v:is_part_of <https://omc.co/5511993891773/7492/{chapter_uri}/{article_uri}>;  
                v:is_part_of <https://omc.co/5511993891773/7492/{parent_id}> .'''

                upload(rdf,'5511993891773')
                #print(rdf)
                ttl_lines.append(rdf+'\n')             

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
