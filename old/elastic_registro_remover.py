import os, re, time, warnings, requests, json, datetime as dt

from elasticsearch  import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from typing         import Dict, List, Any

import openai
import logging

from urllib.parse               import quote_plus,urlparse

from rich           import print

from dotenv         import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client  = openai.OpenAI(api_key=OPENAI_API_KEY)

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST,
    basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)

print('\n--- inicio ---\n')

inicio = time.time()

lista = '''
kumlb5oB89dtCZp8DCA0
k-mlb5oB89dtCZp8DCA0
'''.strip()

for document_id in lista.splitlines():

    try:
        resp_elastic = elastic_client.delete(index="perguntas", id=document_id)
        print(f'-> Elastic: {resp_elastic["result"]}')
    except NotFoundError:
        print(f'-> Elastic: documento {document_id} nÃOo encontrado')


