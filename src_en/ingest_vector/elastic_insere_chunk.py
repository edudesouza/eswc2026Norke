import os
import warnings
import time
import requests
import random

from dotenv                         import load_dotenv
from typing                         import List
from elasticsearch                  import Elasticsearch
from langchain_text_splitters       import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters.base  import Language

import pymupdf4llm
import fitz

import openai

from src_en.config import settings

load_dotenv()

warnings.filterwarnings('ignore')

#index_name  = "a1_convencao"
#caminho_pdf = "Nouveaux_Convencao_2015.pdf"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')
ELASTICSEARCH_PORT = 9200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "pdf", "GPDR_full.pdf")

elastic_client = Elasticsearch( settings.ELASTICSEARCH_HOST,basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASS),verify_certs=False)

def criar_indice(index_name):
    
    index_body = {
        "mappings": {
            "properties": {
                "id_usuario": {"type": "keyword"},
                "id_externo": {"type": "keyword"},
                "texto": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                },
                "chunk_id": {"type": "integer"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1            
        }

    }
    
    if not elastic_client.indices.exists(index=index_name):
        elastic_client.indices.create(index=index_name, body=index_body)

def criar_embedding(texto: str) -> List[float]:
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )

    return response.data[0].embedding

def extrair_texto_pdf(caminho_pdf: str) -> str:

    return pymupdf4llm.to_markdown(caminho_pdf, header=False, footer=False)

def processamento(id_usuario: int,id_arquivo: int,caminho_pdf: str, index_name: str, full_text: str):

    inicio = time.time()

    t_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,chunk_size=1000,chunk_overlap=200)

    if full_text != '':        

        try:            

            chunks = t_splitter.split_text(full_text)
            #print("Chunks: ", chunks)
            print("Length of chunks: ", len(chunks))

            '''
            chunks = c_splitter.split_text(full_text)
            #print("Chunks: ", chunks)
            print("Length of chunks: ", len(chunks))
            '''
            
            #criar_indice(index_name)

            for i, chunk in enumerate(chunks):

                embedding = criar_embedding(chunk)
                doc_id    = f"{id_usuario}_{id_arquivo}_{i}"
                
                documento = {  
                    'texto': chunk,
                    'embedding': embedding,
                    'file_url':caminho_pdf,
                    'id_usuario':id_usuario,
                    'id_externo':id_arquivo,
                    'chunk_id': i
                }                

                elastic_client.index(
                    index=index_name,
                    document=documento,
                    id=doc_id
                )

                print(f"Chunk {i + 1}/{len(chunks)} processado e armazenado")

            fim = time.time()
            tempo_execucao = fim - inicio
            print(f"-> execução: {tempo_execucao:.2f}s")
            print("-> finalizado")

            return {"status":"OK","msg":"finalizado com sucesso","tempo":f"{tempo_execucao:.2f}s","chunks":len(chunks)}

        except Exception as e:

            fim = time.time()
            tempo_execucao = fim - inicio
            
            print(f"Erro ao processar o PDF: {e}")
            return {"status":"ERRO","msg":f"Erro ao processar o PDF: {e}","tempo":f"{tempo_execucao:.2f}s","chunks":0}

    else:

        fim = time.time()
        tempo_execucao = fim - inicio

        print(f"Erro ao receber full text")
        return {"status":"OK","msg":f"Erro ao receber full text","tempo":f"{tempo_execucao:.2f}s","chunks":0}

criar_indice("documentos")
full_text = extrair_texto_pdf(PDF_PATH)
processamento(5511993891773, 7942, PDF_PATH, "documentos", full_text)

