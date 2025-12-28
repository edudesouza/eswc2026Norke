
import time,asyncio

from rich           import print
from elasticsearch  import Elasticsearch

from src.utils      import diff_time, normalize
from src.config     import settings
from src.ingest     import graph_ingest

CONCURRENCY = 5

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
    hits      = resp["hits"]["hits"]
    processar = 0

    sem   = asyncio.Semaphore(CONCURRENCY)
    tasks = []

    async def process_one(index, item):
        async with sem: 

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

            return await graph_ingest(data) 
            
    for i, item in enumerate(hits, start=1):
        tasks.append(asyncio.create_task(process_one(i, item)))

    # Aguarda só no final (batch)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Resumo (sem depender de prints internos do graph_ingest)
    ok = 0
    fail = 0
    for r in results:
        if isinstance(r, Exception):
            fail += 1
        else:
            ok += 1

    print(f"\n[green]OK[/green]: {ok}  [red]ERROS[/red]: {fail}")
    print('[red] --- fim ---')
    diff_time('\n-> fim ingest: ', inicio)

if __name__ == "__main__":   
    asyncio.run( run_batch() )
