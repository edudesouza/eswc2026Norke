import os

from elasticsearch import Elasticsearch

from dotenv import load_dotenv
load_dotenv()

# LLMs
GEMINI_API_KEY   = os.getenv('GEMINI_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
OPENAI_API_KEY   = os.getenv('OPENAI_API_KEY')
OLLAMA           = 'http://localhost:11434/api/generate'

# Database
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_BASE_URL_PROD")
GRAPHDB_USERNAME = os.getenv('GRAPHDB_USERNAME')
GRAPHDB_PASSWORD = os.getenv('GRAPHDB_PASSWORD_PROD')
repositorio      = 'omc_v1'

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')

# Models
EMB_MODEL_NAME       = "rufimelo/Legal-BERTimbau-sts-large-ma-v3"
NLI_MODEL_NAME       = "joeddav/xlm-roberta-large-xnli"
BERTSCORE_MODEL_NAME = "xlm-roberta-large"

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST
    ,basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)