
import time,asyncio

from rich import print

from src.services   import keywords_create, graph_search, vector_search, response_create, ground_truth
from src.utils      import diff_time
from src.evaluation import saf
from src.evaluation import score_dynamic_gt
from src.config     import settings

async def main():  

    print( f'[red] \n--- inicio ---' )
    
    inicio = time.time()  

    pergunta = "Oi, bom dia hoje acordei pensado se posso fazer um culto de final de ano com os irmãos aqui do predio, vi que o salão não está ocupado e como é só pessoal daqui mesmo, acho que não precisa pagar né? obrigado deus te abençõe!"
    user_id  = 5511993891773

    print( f'\n{pergunta}' )
    print( '-'*100,'\n' )

    # gpt-4.1, settings.OPENAI_API_KEY
    # gemini-2.5-flash, settings.GEMINI_API_KEY
    palavras_chave  = keywords_create(pergunta,'gpt-4.1',settings.OPENAI_API_KEY)
    diff_time('\n-> #1 expandir query OK: ', inicio)
    
    inicio = time.time()  

    grafo     = graph_search(palavras_chave,pergunta,user_id)
    contexto  = grafo['response']
    knowledge = grafo['dataset']

    '''vetor     = vector_search(palavras_chave,pergunta,'documentos',user_id)
    contexto  = vetor['response']
    knowledge = vetor['dataset']'''

    diff_time('\n-> #2 buscar dados OK: ', inicio)
    inicio = time.time() 

    task_ground_truth           = asyncio.to_thread(ground_truth,contexto,pergunta,palavras_chave,'gpt',5)
    task_response_llm           = asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'gpt')
    response_gt, response_llm   = await asyncio.gather(task_ground_truth, task_response_llm)
    resposta                    = response_llm['resposta']

    diff_time('\n-> #3 ground truth e resposta OK: ', inicio)
    inicio  = time.time()
    
    print( f'{response_llm['resposta']}\n' )
    
    saf_score                   = saf(knowledge,resposta,pergunta,debug=False )
    dyn_score                   = asyncio.to_thread( score_dynamic_gt,response_gt,resposta )
    response_saf, response_dyn  = await asyncio.gather(saf_score, dyn_score)

    nli_val   = response_dyn.get('score_nli', {}).get('score', 0)
    sim_val   = response_dyn.get('score_sim', {}).get('score', 0)
    match_txt = response_dyn['matched']    
    
    print( f'\n {match_txt}' )
    print( f"-> nli: {nli_val:.2f}" )
    print( f"-> sim: {sim_val:.2f}" )
    print( f'-> saf: {response_saf:.2f}' )
    
    diff_time('\n-> #4 factualidade e comparação: ', inicio)   

    if( nli_val==0 and sim_val==0 and response_saf>0.5 ):

        print( '[red]*** Resposta com alto grau de ambiguidade, llm ou gt problemático\n' )

        '''print( palavras_chave )
        print( '-'*100 )
        print( contexto )
        print( '-'*100 )
        print( response_gt )'''


    print( f'[red] --- fim ---\n' )

if __name__ == "__main__":
    asyncio.run( main() )
