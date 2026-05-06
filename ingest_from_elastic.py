
import time,asyncio

from rich           import print
from elasticsearch  import Elasticsearch

from src.utils      import diff_time, normalize
from src.config     import settings
from src.ingest     import graph_ingest

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
            #"texto":"**Artigo 9º -** É proibida a utilização de qualquer espaço das áreas comuns do **CONDOMÍNIO** para eventos ou entrevistas comerciais, religiosos, profissionais, políticos ou de divulgação de produtos ou serviços e, constatado tal uso o Síndico poderá negar ou cassar, a qualquer momento, a licença concedida para utilização\n\ndo espaço, sem prejuízo de multa, conforme descrito no Capítulo de Penalidades deste Regulamento Interno. Ao **CONDÔMINO** ou visitante da unidade que não respeitar esta obrigatoriedade, terá a unidade punida com multa equivalente a 100% (cem por cento) da cota condominial ordinária das unidades tipo de finais 3 e 4.",
        }  

        result = await graph_ingest(data,debug=True) 
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
