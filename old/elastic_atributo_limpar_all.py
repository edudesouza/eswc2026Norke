import os, re, time, warnings, requests, json, datetime as dt

from elasticsearch  import Elasticsearch, helpers
from typing         import Dict, List, Any

import openai
import logging

from urllib.parse               import quote_plus,urlparse

from requests.auth              import HTTPBasicAuth
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_openai           import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms   import Ollama

import numpy as np, torch
from scipy.special  import softmax
from rich           import print

from transformers           import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.utils     import logging as hf_logging
from sentence_transformers  import SentenceTransformer, util

from bert_score import BERTScorer
from bert_score import score

from dotenv         import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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

query = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks"],
    "query"     : {"match_all":{}}, 
    "size"      : 1500
}

resp = elastic_client.search(index="perguntas", body=query)

'''
for index, item in enumerate(resp["hits"]["hits"],start=1):

    id = item['_id']

    body_limpar = {
        "doc": {            
            "saf_vetor_v2":{}
        },
        "doc_as_upsert": True
    }

    body_remover = {
        "doc": {            
            "source": "ctx._source.remove('saf_vetor_v2')",
            "lang": "painless"
        },
        "upsert": {}
    }

    resp_elastic = elastic_client.update(index="perguntas", id=id, body=body_remover)

    print( f'-> Elastic: {resp_elastic["result"]} {id} {index}' )
'''   

body = {
  "query": {"exists": {"field": "saf_grafo_v4"}},
  "script": {
    "source": "ctx._source.remove('saf_grafo_v4')",
    "lang": "painless"
  }
}

resp = elastic_client.update_by_query(
    index="perguntas",
    body=body,
    conflicts="proceed",
    refresh=True,
    slices="auto"
)
print(resp)

    