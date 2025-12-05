
import time,asyncio

from src.services import keywords_create, graph_search, vector_search, response_create, ground_truth
from src.utils    import diff_time

async def main():  
    
    inicio = time.time()  

    pergunta = "Quero saber até que horas posso usar a piscina"
    user_id  = 5511993891773

    print( f'\n{pergunta}' )
    print( '-'*100 )

    palavras_chave  = keywords_create(pergunta)
    diff_time('-> #1 OK: ', inicio)
    #print( palavras_chave )
    
    inicio = time.time()  

    #grafo           = graph_search(palavras_chave,pergunta,user_id)
    vetor           = vector_search(palavras_chave,pergunta,'documentos',user_id)

    diff_time('-> #2 OK: ', inicio)
    inicio = time.time() 

    task_ground_truth = asyncio.to_thread(ground_truth,vetor,pergunta,'gpt',5)
    task_response_llm = asyncio.to_thread(response_create,palavras_chave,pergunta,vetor,'gpt')

    response_gt, response_llm = await asyncio.gather(task_ground_truth, task_response_llm)

    diff_time('-> #3 OK: ', inicio)

    print( f'{response_llm['resposta']}\n' )

if __name__ == "__main__":
    asyncio.run( main() )
