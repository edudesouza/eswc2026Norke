
import time,asyncio

from rich           import print
from elasticsearch  import Elasticsearch

from src.utils      import diff_time, normalize
from src.config     import settings
from src.ingest     import graph_ingest

async def run_batch():
    
    print( '\n--- inicio ---\n') 

    inicio = time.time()

    elastic_client = Elasticsearch( 
        settings.ELASTICSEARCH_HOST,basic_auth=(settings.ELASTICSEARCH_USER,settings.ELASTICSEARCH_PASS),verify_certs=False
    )

    query = {
        "_source"   : ["file_url", "id_usuario", "id_externo","texto"],
        "query"     : {
            "bool":{
                "filter": [
                    {"term": { "id_usuario":"5511993891773" }},
                    {"term": { "id_externo":"749" }}
                ]
            }
        }, 
        "size" : 1500
    }

    resp      = elastic_client.search(index="documentos", body=query)
    total     = len(resp["hits"]["hits"])
    processar = 0

    for index, item in enumerate(resp["hits"]["hits"],start=1): 

        id   = item['_id']
        text = item['_source']['texto']
        file = item['_source']['file_url'],   

        data = {
            "id":id,
            "arquivo":file,
            "id_usuario":"5511993891773",
            "id_externo":749,
            "texto":normalize(text)
        }  

        result = await graph_ingest(data) 
        try:                
            print( f'-> res: {result}' ) 
        except Exception as erro:
            print( f'-> ERRO processar ({id})' )  
            
        processar = total-index
        print ( f'{index} de {total}, {processar}\n')

    print( f'[red] --- fim ---' )
    diff_time('\n-> fim ingest: ', inicio)

if __name__ == "__main__":   
    asyncio.run( run_batch() )
