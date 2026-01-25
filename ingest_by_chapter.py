
import time,asyncio

from rich           import print
from elasticsearch  import Elasticsearch

from src.utils      import diff_time, normalize
from src.config     import settings
from src.ingest     import graphdb_insert_by_chapter

async def run_batch():
    
    print( '\n--- inicio ---') 
    print( f'--- Repo: {settings.repositorio} ---\n') 

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
    full_text = ''
    file      = ''

    for index, item in enumerate(resp["hits"]["hits"],start=1):    

        id   = item['_id']
        text = item['_source']['texto']
        file = item['_source']['file_url'], 

        full_text += f'Chunk:{id}\n{normalize(text)}\n'         
            
        processar = total-index
        print ( f'{index} de {total}, {processar}')
    
    print( f'[red] --- fim ---' )    

    for i in range(1, 23):
        capitulo_nome = f"Capítulo {i}"
        print(f"Processando {capitulo_nome}...")

        data = {
            "id":capitulo_nome,
            "arquivo":file,
            "id_usuario":"5511993891773",
            "id_externo":749,
            "texto":full_text
        }

        result = await graphdb_insert_by_chapter(data, capitulo_nome) 
                
        try:                
            print( f'-> res: {result}' ) 
        except Exception as erro:
            print( f'-> ERRO processar ({capitulo_nome})' )
    
    diff_time('\n-> fim ingest: ', inicio)

if __name__ == "__main__":   
    asyncio.run( run_batch() )
