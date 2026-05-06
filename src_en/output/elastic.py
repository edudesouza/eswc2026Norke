
from elasticsearch import Elasticsearch

from src_en.config import settings

elastic_client = Elasticsearch( 
    settings.ELASTICSEARCH_HOST,basic_auth=(settings.ELASTICSEARCH_USER,settings.ELASTICSEARCH_PASS),verify_certs=False
)

def elastic_update_field(elastic_index,elastic_id,complexity,saf,nli,sim,response,model,field):

    if elastic_id=='':
        return 'Elastic ID missing'
    
    if elastic_index=='':
        return 'Elastic INDEX missing'

    body = {
        "doc": {            
            field:{
                "model":model,
                "response":response,
                "complexity":complexity,
                "nli":nli,
                "sim":sim,
                "saf": saf
            }            
        },
        "doc_as_upsert": True
    }

    res = elastic_client.update(index=elastic_index, id=elastic_id, body=body)
 
    return res
