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

def criar_resposta_v1(pergunta,contexto,modelo):

    print('-> criar resposta')

    system = (
        "Você é um assistente jurídico de condomínios no Brasil. "
        "Responda APENAS com base nas evidências fornecidas. "
        "Se houver uma regra explícita, cite."
        "Se não houver base suficiente, diga isso."
        "Relacionamentos, mostre atributos que trazem o relacionamento entre os itens usados na resposta"
        "Saída OBRIGATÓRIA: JSON válido com EXATAMENTE três campos: "
        '{"resposta": string, "resposta_completa": string, "chunks": array de strings}. Sem texto extra.'
    )

    user = f'''
        Você é um assistente jurídico condominial. Responda SEMPRE em português claro e objetivo.

        Use SOMENTE as informações do CONTEXTO abaixo. 
        Se a resposta não estiver clara no contexto, diga que não é possível responder com segurança.

        PERGUNTA:
        {pergunta}

        CONTEXTO:
        {contexto}

        Agora responda ao morador em no máximo 4 linhas, sendo bem direto:
    '''

    llm = Ollama(model=modelo)

    msg = llm.invoke([("system", system), ("user", user)])

    msg_json = json.loads(msg)

    return msg_json

query = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks"],
    "query"     : {"match_all":{}}, 
    "size"      : 1
}

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

diff_time('-> Buscar Elstic OK: ', inicio)

for index, item in enumerate(resp["hits"]["hits"],start=1):  
    
    id       = item['_id']
    pergunta = item['_source']['pergunta']
    resposta = item['_source']['resposta']
    contexto = item['_source']['contexto']
    
    try:
        grafo = item['_source']['grafo']
    except Exception as erro:
        grafo = ''

    if( grafo!='' ):
        print( 'OK, possui resposta' )
    else: 
        print( item )

    inicio = time.time()

    resp = criar_resposta_v1(pergunta,contexto,modelo)

    print( resp['resposta'] )
    print(  '-'*100 )
    print( resp )

    diff_time('-> Resp OK: ', inicio)


    