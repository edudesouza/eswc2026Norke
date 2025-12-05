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

from deepeval           import evaluate as ev_deep
from deepeval.test_case import LLMTestCase
from deepeval.metrics   import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models    import OllamaModel
from langchain_together import ChatTogether

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

import langextract as lx

from rich           import print
from dotenv         import load_dotenv
load_dotenv()

os.environ['CONFIDENT_METRIC_LOGGING_VERBOSE'] = '0'
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "200"

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

TOGETHER_API_KEY   = os.getenv('TOGETHER_API_KEY')
OPENAI_API_KEY     = os.getenv('OPENAI_API_KEY')
openai_client      = openai.OpenAI(api_key=OPENAI_API_KEY)

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST
    ,basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)

CANON_TYPES = {
    "EntidadeFisica","CultoReligioso","Pessoa","Papel","Documento","Regra","Ocorrencia","Veiculo","InstituicaoExterna",
    "Condominio","Edificacao","Unidade","AreaComum","Garagem","VagaDeEstacionamento","Piscina",
    "SalaoDeFestas","Playground","Academia","SistemaControleAcesso",
    "Sindico","SubSindico","Conselheiro","Morador","Inquilino","Visitante","Funcionario",
    "Zelador","Porteiro","Faxineiro","Vigia","Jardineiro","PrestadorDeServico",
    "Administradora","Contador","Advogado","Seguradora","Banco","Fornecedor",
    "Convencao","RegimentoInterno","AtaDeAssembleia","Orcamento","Balancete",
    "ContratoPrestacaoServicos","Notificacao","Multa",
    "Assembleia","Manutencao","Obra","ReservaAreaComum","Seguranca","Correspondencia","Usuario","Chunk"
}

CANON_RELATIONSHIPS = {
    "parteDe","localizadaEm","possuiUnidade","possuiAreaComum","desempenhaPapel","papelEm",
    "aplicaA","permitidoEm","proibidoEm","registradoEm","ocorreuEm","executadoPor","temResponsavel",
    "contrata","temContaEm","envolveVeiculo","reservadoPor","refereSeA",
    "proibidoSe","responsabilizadoPor","podeUsar","tipoDe","inclui","citadoEm","usa","ocupa","regidaPor",
    "estaContidoEm","possuiDocumento","responsavelPor","nome","identificador","cpf","cnpj","data","valor","texto",
    "descricao","conteudo","arquivo","idChunk","idUsuario","idExterno"
}

ALLOWED_NODE_PROPS = {"descricao"}  # mantenha enxuto

allowed_rels = sorted(list(CANON_RELATIONSHIPS))

#--------------------------------------------------------------

REL_ALIASES = {
    # CAPS/_ → camelCase
    "PERMITIDO_EM": "permitidoEm",
    "APLICA_A": "aplicaA",
    "PARTE_DE": "parteDe",
    "PROIBIDO_SE": "proibidoSe",
    "REFERE_SE_A": "refereSeA",
    "RESPONSABILIZADO_POR": "responsabilizadoPor",
    "PODE_USAR": "podeUsar",
    "TIPO_DE": "tipoDe",
    "INCLUI": "inclui",
    "CITADO_EM": "citadoEm",
    "USA": "usa",
    "OCUPA": "ocupa",
    "REGIDA_POR": "regidaPor",
    "ESTA_CONTIDO_EM":"estaContidoEm",
    "PARTE_DE":"parteDe"
}

REL_MAP = {
    "REGISTRADOEM":   "registradoEm",
    "REGIDAPOR":      "regidaPor",
    "CONTRATA":       "contrata",
    "TEMRESPONSAVEL": "temResponsavel",
    "REFERESEA":      "refereSeA",
}

def criar_regra(texto,modelo): 

    print('-> criar regra')

    llm_model = os.getenv("LLM_MODEL",modelo)
    llm = ChatOpenAI(
        model=llm_model,   
        model_kwargs={
            "response_format": {"type": "text"}
        }     
    )

    '''llm = Ollama(
        model='kimi-k2:1t-cloud',
        temperature=0.3,
        format="json"
    )'''

    '''llm = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        temperature=0.3,
        model=resolvedor,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )''' 

    system = (
        f'''Você é um extrator de regras jurídicos em pt-BR.
        Tipos permitidos: {sorted(CANON_TYPES)}
        Relações permitidas: {allowed_rels}
        Propriedades literais de nó permitidas:  {sorted(ALLOWED_NODE_PROPS)}
        Regras são normas ou determinação aplicáveis no condomínio.
        Sempre que necessário crie mais de uma regra, pois um texto pode conter várias regras, contectadas à várias entidades.
        Relacionamento entre entidades deve ser representado em grafos de conhecimento, é como uma regra se conecta a outras entidades.


        Eom do discuro:
        user termos coloquiais, linguagem simples e direta, sem jargões técnicos, não seja ambíguio, não use afirmação e negação juntas (ex.: “sim, não é permitido”).
        
        Exemplos e regras:
        Proibições: proibido entrar no elevador social em traje de banho.
        Permissões: obras e reformas são mermitidas de segunda à sexta das 8 até 17hs.
        Penalidades: o morador que descumprir a regra será notificado e em caso de reincidência será multado.

        Exemplo de relacionamento entre entidades:
        Cardinalidade de vagas/veículos: crie Regra e ligue com :aplicaA → (VagaDeEstacionamento|Garagem|Condominio)
        Proibições: (Veiculo)-[:proibidoEm|:proibidoSe]->(EntidadeFisica ou Limite textual)
        Permissões: (Veiculo)-[:permitidoEm]->(VagaDeEstacionamento|Garagem)
        Penalidades: (Morador)-[:responsabilizadoPor {{motivo}}]->(Multa|Notificacao)
        
        Output:
        tipo: <proibição | permissão | penalidade>
        descrição: <descrição da regra>
        relacionamento: <relacionamento entre a regra e outras entidades>
        entidades envolvidas: <lista de entidades envolvidas na regra>

        Saída fiel e sucinta, compatível com texto apresentado, em texto puro.'''
    )   

    user = f'''       
        Texto:
        {texto}        
    '''

    msg = llm.invoke([("system", system), ("user", user)])

    print('-> regra criada\n')
    
    #return msg.content
  
    return msg.content 

def atualizar_elastic(regra,id):

    embeddings     = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_rico = embeddings.embed_query(regra) 

    body = body = {
        "script": {
            "lang": "painless", 
            "source": """
                ctx._source.texto_rico = params.texto_rico;
                ctx._source.embeddings_rico = params.embeddings_rico;
            """,           
            "params": {
                "texto_rico":regra,
                "embeddings_rico":embedding_rico
                }
            }
        }      
    
    res = elastic_client.update(index="documentos", id=id, body=body)
 
    return res

def criar_mapping():

    mapping = {
        "properties": {
            "embedding_rico": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
            "texto_rico": {
                "type": "text"
            }
        }
    }

    '''elastic_client.indices.put_mapping(
        index="documentos",
        body=mapping
    )'''

    # Print do mapping completo do índice
    mapping_atual = elastic_client.indices.get_mapping(index="documentos")
    print(json.dumps(mapping_atual.body, indent=2, ensure_ascii=False))

    return

def diff_time(legenda,inicio):

    fim = time.time()
    tpo  = fim - inicio

    print( f'{legenda}{tpo:.2f}s\n' )

    return

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

#-------------------------------------------------------

print( '\n--- inicio ---\n' )

query = {
    "_source"   : ["file_url", "id_usuario", "id_externo", "texto"],
    "query": {
        "bool": {
            "filter": [
                { "term": { "id_usuario": "5511993891773" } },
                { "term": { "id_externo": "749" } }
            ]
        }
    },
    "size": 1500
}

query_unico = {
    "_source"   : ["file_url", "id_usuario", "id_externo", "texto"],
    "query": {
        "ids": {"values": ["5511993891773_749_9"]},  
    },
    "size": 1
}

resp = elastic_client.search(index="documentos", body=query)

total = len(resp["hits"]["hits"])

#print( resp["hits"]["hits"][0] )
#texto = normalize_ws( resp["hits"]["hits"][0]["_source"]["texto"])
#regra = criar_regra(texto,modelo="o4-mini")
#input = normalize_ws( resp["hits"]["hits"][0]["_source"]["texto"])+'\n\nRegra(s):\n'+regra
#print( input )

# mapear os novos campos no elastic
#criar_mapping()

for index, item in enumerate(resp["hits"]["hits"],start=1):    
    
    id    = item["_id"]
    texto = normalize_ws( item["_source"]["texto"])
    regra = criar_regra(texto,modelo="o4-mini")

    input = normalize_ws( resp["hits"]["hits"][0]["_source"]["texto"])+'\n\nRegra(s):\n'+regra
    print( input )

    update = atualizar_elastic(input,id)

    try:   
        print( f'\n-> Elastic: {update['result']} {index} de {total}\n' )
    except Exception as erro:
        print( f'\nERRO: {update}\n' ) 

print( '\n--- fim ---\n' )

print( f'\nTotal perguntas: {total}\n' )