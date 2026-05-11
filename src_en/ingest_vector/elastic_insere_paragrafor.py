import os, warnings, time, requests, random, json,re,unicodedata, uuid

from dotenv                         import load_dotenv
from elasticsearch                  import Elasticsearch

import openai

load_dotenv()

warnings.filterwarnings('ignore')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')
ELASTICSEARCH_PORT = 9200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "pdf", "GPDR_full.pdf")

#elastic_client = Elasticsearch( ELASTICSEARCH_HOST,basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),verify_certs=False)
#elastic_client = Elasticsearch( "http://35.188.215.94:9200",basic_auth=("omc", "kwd#Omc76"),verify_certs=False)

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

def upload():
    return

with open("src_en/ingest/gdpr.json", encoding="utf-8") as f:
    json_gdpr = json.load(f)

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

                text = no_spaces(point.get("text"))
                dados = text or ""

                tipo      = 'Point' 
                parent_id = f"chapter_{chapter['number'].lower()}_article_{article['number']}"

                rdf = f'''<https://omc.co/5511993891773/7492/{chapter_uri}/{article_uri}/{paragraph_uri}>
                a v:{tipo} ;            
                v:fullText "{dados}"@en ;
                v:breadcrumb        "chapter {chapter['number']}, article {article['number']}, point {paragraph_number}"@en ;
                v:breadcrumbLabel   "chapter {chapter['number']} ({chapter['title']}), article {article['number']} ({article['title']}), point {paragraph_number}"@en ;
                v:isPartOfArticle <https://omc.co/5511993891773/7492/{chapter_uri}/{article_uri}>;  
                v:refersTo <https://omc.co/5511993891773/7492/{parent_id}> .'''

                print('-> paragrado')
                upload(rdf,'5511993891773')

                for subpoint_index, subpoint in enumerate(point.get("subpoints") or [], start=1):
                    
                    subpoint_number = subpoint.get("number") or f"subpoint_{subpoint_index}"
                    subpoint_paragraph_number = f"{paragraph_number}{subpoint_number}"
                    subpoint_uri = "Point_" + str(subpoint_paragraph_number)
                    sub_text = no_spaces(subpoint.get("text"))

                    rdf = f'''<https://omc.co/5511993891773/7492/{chapter_uri}/{article_uri}/{subpoint_uri}>
                    a v:{tipo} ;            
                    v:fullText "{sub_text}"@en ;
                    v:breadcrumb        "chapter {chapter['number']}, article {article['number']}, point {subpoint_paragraph_number}"@en ;
                    v:breadcrumbLabel   "chapter {chapter['number']} ({chapter['title']}), article {article['number']} ({article['title']}), point {subpoint_paragraph_number}"@en ;
                    v:isPartOfArticle <https://omc.co/5511993891773/7492/{chapter_uri}/{article_uri}>;  
                    v:refersTo <https://omc.co/5511993891773/7492/{parent_id}> .'''

                    print('-> sub paragrado')
                    upload(rdf,'5511993891773')

