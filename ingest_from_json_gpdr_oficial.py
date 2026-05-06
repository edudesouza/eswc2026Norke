
import time,asyncio,json

from rich           import print
from elasticsearch  import Elasticsearch

from src_en.utils      import diff_time, normalize
from src_en.config     import settings
from src_en.ingest     import graph_ingest_gpdr_omc

async def run_batch():
    
    print( '\n--- inicio ---') 
    print( f'--- Repo: {settings.repositorio} ---\n') 

    inicio = time.time()

    with open("src_en/ingest/gdpr.json", encoding="utf-8") as f:
        json_gdpr = json.load(f)   
    
    for chapter in json_gdpr["chapters"]:
        
        for item in chapter["contents"]:
            
            articles = []

            if item["type"] == "article":
                articles.append(item)

            if item["type"] == "section":
                articles.extend(
                    content
                    for content in item["contents"]
                    if content["type"] == "article"
                )

            for article in articles:
                
                id = f"chapter_{chapter['number']}_article_{article['number']}".lower()
                dados = f"Article {article['number']}: {article['title']}"

                for point in article["contents"]:
                    
                    number = point.get("number")
                    text = point.get("text")

                    if number and text:
                        dados += f"\n{number} {text}"
                    elif text:
                        dados += f"\n{text}"

                    for subpoint in point.get("subpoints", []):
                        sub_number = subpoint.get("number")
                        sub_text = subpoint.get("text")

                        if sub_number and sub_text:
                            dados += f"\n({sub_number}) {sub_text}"
                        elif sub_text:
                            dados += f"\n{sub_text}"

                print(id)
                print('-'*100)

                data = {
                    "id":id,
                    "arquivo":'GDPR',
                    "id_usuario":"5511993891773",
                    "id_externo":7492,
                    "texto":normalize(dados)
                }

                result = await graph_ingest_gpdr_omc(data,debug=False) 

                try:                
                    print( f'-> res: {result}' ) 
                except Exception as erro:
                    print( f'-> ERRO processar ({id})' )   
    
    print( f'[red] --- fim ---' )
    diff_time('\n-> fim ingest: ', inicio)

if __name__ == "__main__":   
    asyncio.run( run_batch() )
