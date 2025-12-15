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

inicio = time.time()

lista = '''Iemzb5oB89dtCZp8xSKK
9emob5oB89dtCZp8pSCX
6emob5oB89dtCZp8pSCX
4Omob5oB89dtCZp8pSCX
Aumxb5oB89dtCZp8gCJO
fOmub5oB89dtCZp8EiG9
Demzb5oB89dtCZp8xSKK
Dumzb5oB89dtCZp8xSKK
COmzb5oB89dtCZp8xSKK
nOm5b5oB89dtCZp8UCLc
2Omxb5oB89dtCZp8gCFN
E-mzb5oB89dtCZp8xSKK
DOmzb5oB89dtCZp8xSKK
Bumzb5oB89dtCZp8xSKK
FOmzb5oB89dtCZp8xSKK
C-mzb5oB89dtCZp8xSKK
lem5b5oB89dtCZp8UCLc
Cemzb5oB89dtCZp8xSKK
Kemzb5oB89dtCZp8xSKK
Humzb5oB89dtCZp8xSKK
KOmzb5oB89dtCZp8xSKK
Bemzb5oB89dtCZp8xSKK
MOmzb5oB89dtCZp8xSKK
Lumzb5oB89dtCZp8xSKK
M-mzb5oB89dtCZp8xSKK
Lemzb5oB89dtCZp8xSKK
Memzb5oB89dtCZp8xSKK
J-mzb5oB89dtCZp8xSKK
OOmqb5oB89dtCZp8eiGW
BOmzb5oB89dtCZp8xSKJ
Cumzb5oB89dtCZp8xSKK
H-mzb5oB89dtCZp8xSKK
B-mzb5oB89dtCZp8xSKK
Gemzb5oB89dtCZp8xSKK
O-mqb5oB89dtCZp8eiGW
'''.strip()

body = {
    "doc": {            
        "source": "ctx._source.remove('saf_grafo')",
        "lang": "painless"
    },
    "upsert": {}
}

for document_id in lista.splitlines():

    try:
        resp_elastic = elastic_client.update(index="perguntas", id=id, body=body)
        print(f'-> Elastic: {resp_elastic["result"]}')
    except Exception as erro:
        print(f'-> Elastic: erro {erro}')



print( f'-> Elastic: {resp_elastic["result"]}' )
print( resp_elastic )
    

    