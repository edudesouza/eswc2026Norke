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

OLLAMA = 'http://localhost:11434/api/generate'

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST,
    basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)

print('\n--- inicio ---\n')

model_name  = "joeddav/xlm-roberta-large-xnli"
model_sim   = SentenceTransformer("rufimelo/Legal-BERTimbau-sts-large-ma-v3")

#EMB_MODEL_NAME = "neuralmind/bert-base-portuguese-cased" 
EMB_MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large-ma-v3" 
#EMB_MODEL_NAME = "xlm-roberta-large"

NLI_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
#NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
#NLI_MODEL_NAME = "wilsonmarciliojr/bertimbau-embed-nli"
#NLI_MODEL_NAME = "roberta-large-mnli"
#NLI_MODEL_NAME = "ruanchaves/bert-base-portuguese-cased-faquad-nli"

''' 
-> NLI
roberta-large-mnli
joeddav/xlm-roberta-large-xnli
MoritzLaurer/deberta-v3-large-zeroshot-v2
facebook/bart-large-mnli
cross-encoder/nli-deberta-v3-base
ruanchaves/bert-base-portuguese-cased-faquad-nli
wilsonmarciliojr/bertimbau-embed-nli
'''

smodel        = SentenceTransformer(EMB_MODEL_NAME)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=False)
nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
config        = AutoConfig.from_pretrained(NLI_MODEL_NAME)

inicio = time.time()

def nli( referencia,candidato ):

    print("-> NLI")

    model_input = nli_tokenizer(
        *([referencia],[candidato]), 
        padding=True, 
        return_tensors="pt"
    )

    with torch.no_grad():

        output  = nli_model(**model_input)
        scores  = output[0][0].detach().numpy()
        scores  = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        score_por_label = {config.id2label[i]: scores[i] for i in range(len(scores))}

        entailment      = score_por_label.get('entailment', 0)
        neutral         = score_por_label.get('neutral', 0)
        contradiction   = score_por_label.get('contradiction', 0)  
        
        return {
            "entailment":float(entailment), 
            "contradiction":float(contradiction), 
            "neutral":float(neutral)
        }

def bertscore( referencia,candidato ):

    print("-> Bert score")

    scorer = BERTScorer(
        #model_type="neuralmind/bert-base-portuguese-cased", num_layers=12,
        model_type="xlm-roberta-large", num_layers=24,
        #model_type="rufimelo/Legal-BERTimbau-sts-large-ma-v3", num_layers=12,
        lang="pt", 
        rescale_with_baseline=False,    
    )

    P, R, F1 = scorer.score(candidato, referencia, verbose=False)

    
    precision = P.item()
    recall    = R.item()
    f1        = F1.item()

    return {
        "precision":float(f1),
        "recall":float(recall),
        "f1":float(f1)
    }

def avaliar_resposta_completa(entailment, neutral, contradiction, f1):

    print('-> score combinado')

    # Score de fidelidade (NLI)
    fidelidade = entailment + (neutral * 0.5)  # Tolera neutral parcialmente
    
    # Score de relevância (BERT)
    relevancia = f1
    
    # Penalidade por contradição
    penalidade = contradiction
    
    # Score final: média ponderada com penalidade
    score_final = (
        (fidelidade * 0.4) +      # 40% fidelidade
        (relevancia * 0.6) -      # 60% relevância (mais importante)
        (penalidade * 2.0)        # Penalidade forte para contradição
    )
    
    # Garantir que fica entre 0 e 1
    score_final = max(0, min(1, score_final))
    
    # Classificação
    if score_final >= 0.8:
        categoria = "EXCELENTE"
    elif score_final >= 0.65:
        categoria = "BOA"
    elif score_final >= 0.5:
        categoria = "ACEITÁVEL"
    else:
        categoria = "INADEQUADA"
    
    return {
        'score_final': score_final,
        'categoria': categoria,
        'detalhes': {
            'fidelidade': fidelidade,
            'relevancia': relevancia,
            'penalidade': penalidade
        }
    }

def atualizar_elastic(id,nli,bertsocre,score_combinado):

    body = {
        "doc": {            
            "score_grafo":
                {
                "nli_score":nli,
                "bert_score":bertsocre,
                "score_combinado": score_combinado
            }            
        },
        "doc_as_upsert": True
    }

    res = elastic_client.update(index="perguntas", id=id, body=body)
 
    return res

def diff_time(legenda,inicio):

    fim = time.time()
    tpo  = fim - inicio

    print( f'{legenda}{tpo:.2f}s\n' )

    return

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

resp   = elastic_client.search(index="perguntas", body=query_unico)

#recuperar apenas 1 item
'''resp = elastic_client.get(
    index="perguntas",
    id="kumbb5oB89dtCZp8OR-z"
)'''

'''
gpt-oss:20b 
gemma3:latest 
gemma3n:latest

kimi-k2:1t-cloud
minimax-m2:cloud
'''

modelo = "kimi-k2:1t-cloud" 

diff_time('-> Buscar Elstic OK: ', inicio)

total = len(resp["hits"]["hits"])

for index, item in enumerate(resp["hits"]["hits"],start=1):  

    inicio = time.time() 
    
    id       = item['_id']
    pergunta = item['_source']['pergunta']
    contexto = item['_source']['contexto']  

    try:

        print( f'Confiabilidade: {item['_source']['avaliacoes'][0]['confiabilidade']}, Relevância: {item['_source']['avaliacoes'][0]['relevancia']}' )
        print( '-'*100 )

        score_nli = nli(item['_source']['contexto'],item['_source']['avaliacoes'][0]['grafo'] )      
        #print( score_nli )

        score_bertscore = bertscore([item['_source']['contexto']],[item['_source']['avaliacoes'][0]['grafo']] )      
        #print( score_bertscore )

        score_combinado = avaliar_resposta_completa(
            score_nli['entailment'],
            score_nli['contradiction'],
            score_nli['neutral'],
            score_bertscore['f1']
        )

        print( score_combinado )
        print( '-'*100 )

        resp_elastic = atualizar_elastic(id,score_nli,score_bertscore,score_combinado)
        print( f'-> Elastic: {resp_elastic['result']} {index} de {total}' )

        diff_time('\n-> Resp: ', inicio)
    
    except Exception as erro:
        print( f'ERRO: {id}')
    

    