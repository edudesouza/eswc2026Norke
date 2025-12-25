
from elasticsearch import Elasticsearch

from src.config import settings

elastic_client = Elasticsearch( 
    settings.ELASTICSEARCH_HOST,basic_auth=(settings.ELASTICSEARCH_USER,settings.ELASTICSEARCH_PASS),verify_certs=False
)

def elastic_load_one(elastic_index,elastic_id):

    if elastic_id=='':
        return 'Elastic ID missing'
    
    if elastic_index=='':
        return 'Elastic INDEX missing'

    fields = {
        "_source"   : ["id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks","saf_grafo_v2"]
    }

    #res = elastic_client.get(index=elastic_index, id=elastic_id, "_source"=fields)
    res = elastic_client.get(index=elastic_index, id=elastic_id)
 
    return res

def elastic_load_batch(elastic_index,elastic_id):

    if elastic_id=='':
        return 'Elastic ID missing'
    
    if elastic_index=='':
        return 'Elastic INDEX missing'

    query = {
        "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks","saf_grafo_v2"],
        "query"     : {"match_all":{}}, 
        "size"      : 1500
    }

    res = elastic_client.search(index=elastic_index, body=query)
 
    return res
