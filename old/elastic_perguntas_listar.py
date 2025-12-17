import os, re, time, warnings, requests, json, datetime as dt

from rich           import print
from elasticsearch  import Elasticsearch, helpers
from typing         import Dict, List, Any

import openai
import logging

from urllib.parse               import quote_plus,urlparse

from requests.auth              import HTTPBasicAuth
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_openai           import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms   import Ollama

from dotenv         import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client  = openai.OpenAI(api_key=OPENAI_API_KEY)

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')

OLLAMA = 'http://localhost:11434/api/generate'

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST,
    basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)

print('\n--- inicio ---\n')

def diff_time(legenda,inicio):

    fim = time.time()
    tpo  = fim - inicio

    print( f'{legenda}{tpo:.2f}s\n' )

    return

query = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks"],
    "query"     : {"match_all":{}}, 
    "size"      : 1
}

query_todos = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","chunks","avaliacoes"],
    "query"     : {"match_all":{}}, 
    "size"      : 1500
}

query_unico = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","chunks","avaliacoes","score_grafo"],
    "query": {
        "ids": {
            "values": ["iumbb5oB89dtCZp8OR-z"]
        }
    },
    "size": 1
}

elastic_client.delete_by_query(
    index="perguntas",
    body={
        "query": {
            "ids": {
                "values": ["<built-in function id>"]
            }
        }
    },
    refresh=True,
    conflicts="proceed"
)

resp = elastic_client.search(index="perguntas", body=query)

inicio = time.time()

'''
gpt-oss:20b 
gemma3:latest 
gemma3n:latest

kimi-k2:1t-cloud
minimax-m2:cloud
'''

modelo = "kimi-k2:1t-cloud" 
resp   = elastic_client.search(index="perguntas", body=query)

print( resp )

diff_time('-> Buscar Elastic OK: ', inicio)

'''
for index, item in enumerate(resp["hits"]["hits"],start=1):  
    
    id       = item['_id']
    pergunta = item['_source']['pergunta']
    resposta = item['_source']['resposta']
    contexto = item['_source']['contexto']
    
    print( pergunta )
    print( resposta )

    print(  '-'*100 )

    diff_time('-> Resp OK: ', inicio)
'''

    