
import time,asyncio,json

from rich           import print
from elasticsearch  import Elasticsearch

from src.utils      import diff_time, normalize
from src.config     import settings
from src.ingest     import graph_ingest

async def run_batch():
    
    print( '\n--- inicio ---') 
    print( f'--- Repo: {settings.repositorio} ---\n') 

    inicio = time.time()

    with open("./src/ingest/output/chunks_normativos.json", encoding="utf-8") as f:
        json_chunks = json.loads(f.read())
        
    chunks    = [item for item in json_chunks if item.get("paragrafo") == "caput"]
    total     = len(chunks)
    processar = 0

    for index, item in enumerate(chunks,start=1): 

        id   = item['id']
        text = item['text']
        file = 'https://storage.googleapis.com/comtodos-607d6.appspot.com/5511969033344/5511993891773/Nouveaux_Regulamento_2023.pdf',   

        data = {
            "id":id,
            "arquivo":file,
            "id_usuario":"5511993891773",
            "id_externo":749,
            "texto":normalize(text)
            #"texto":"Realizar apresentação para clientes ou reunião de negócios"
        }  

        result = await graph_ingest(data,debug=False) 
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
