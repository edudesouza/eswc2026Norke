
import csv,os,sys,time,asyncio

from rich import print

from elasticsearch  import Elasticsearch

from src.services   import keywords_create, graph_search, vector_search, response_create, ground_truth
from src.utils      import diff_time
from src.evaluation import saf, score_dynamic_gt
from src.config     import settings

elastic_client = Elasticsearch( 
    settings.ELASTICSEARCH_HOST,basic_auth=(settings.ELASTICSEARCH_USER,settings.ELASTICSEARCH_PASS),verify_certs=False
)

def atualizar_elastic(id,complexity,nli,sim,saf,response):

    body = {
        "doc": {            
            "saf_grafo_v2":{
                "model":"gpt-oss:120b-cloud",
                "response":response,
                "complexity":complexity,
                "nli":nli,
                "sim":sim,
                "saf": saf
            }            
        },
        "doc_as_upsert": True
    }

    res = elastic_client.update(index="perguntas", id=id, body=body)
 
    return res

async def main(id,user_id,pergunta,retrieval='grafo',retrieval_size=5,size_gt=5,debug_all=False,debug_one=[],output=False):  

    print( f'[red] \n--- inicio ---' )

    if not pergunta:
        print('Nenhuma pergunta econtrada!')
        exit()
    
    inicio = time.time()  
    inicio_global = time.time()    

    print( f'\nUSER: {pergunta}' )
    print( '-'*100,'\n' )

    # query expantion
    #--------------------------------------------------------------------------

    # gpt-4.1, settings.OPENAI_API_KEY
    # gemini-2.5-flash, settings.GEMINI_API_KEY
    expantion = keywords_create(pergunta,'gemini-2.5-flash',settings.GEMINI_API_KEY)
    palavras_chave   = expantion['keywords']
    complexity_score = expantion['complexity_score']

    # quanto maior a similaridade mais próximo a um tema único
    if complexity_score<0.75 :
        print( f'-> complexidade alta: {complexity_score:.2f}' )
        retrieval_size = 20
        size_gt = 10
    else:
        print( f'-> complexidade baixa: {complexity_score:.2f}' )

    diff_time('\n-> #1 expandir query OK: ', inicio)

    if debug_all or 'query' in debug_one:
        print( f'[yellow]// Debug keywords:\n{palavras_chave}\n' )

    # retriever
    #--------------------------------------------------------------------------
    
    inicio = time.time() 

    if retrieval=='grafo':

        recuperacao = graph_search(palavras_chave,pergunta,user_id,retrieval_size)
        contexto  = recuperacao['response']
        knowledge = recuperacao['dataset']  
    
    else:  

        recuperacao = vector_search(palavras_chave,pergunta,'documentos',user_id,retrieval_size)
        contexto  = recuperacao['response']
        knowledge = recuperacao['dataset']    

    diff_time('\n-> #2 buscar dados, OK: ', inicio)
    
    if debug_all or 'retriever' in debug_one:
        print( f'[yellow]// Debug contexto  {retrieval}:\n{recuperacao['response']}' )
        print( f'[yellow]// Debug knowledge {retrieval}:\n{recuperacao['dataset']}' )

    # ground truth and response
    #--------------------------------------------------------------------------

    inicio = time.time() 

    task_ground_truth           = asyncio.to_thread(ground_truth,contexto,pergunta,palavras_chave,'ollama',size_gt)
    task_response_llm           = asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'ollama')
    response_gt, response_llm   = await asyncio.gather(task_ground_truth, task_response_llm)
    resposta                    = response_llm['resposta']

    print( f'\nLLM: {resposta}\n' )

    diff_time('\n-> #3 ground truth e resposta OK: ', inicio)

    if debug_all or 'ground_truth' in debug_one:
        print( f'[yellow]// Debug GT:\n{response_gt}\n' )

    # metrics
    #--------------------------------------------------------------------------
    
    inicio  = time.time()    
    
    saf_score                   = saf(knowledge,resposta,pergunta,debug_all)
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

    if output==True:

        csv_file = "bench_grafo_0812.csv"
        output_row = [ retrieval, pergunta, f'{saf_score:.2f}', f'{nli_val:.2f}', f'{sim_val:.2f}', resposta ]

        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
    
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, delimiter=';')
            
            if not file_exists:
                writer.writerow(['tipo', 'chave', 'saf_score', 'nli_score', 'sim_score', 'resposta_llm'])
            
            writer.writerow(output_row)

    if( nli_val==0 and sim_val==0 and response_saf>0.5 ):

        print( '[red]*** Resposta com alto grau de ambiguidade: retriver, llm ou gt problemático ***\n' )

        '''print( palavras_chave )
        print( '-'*100 )
        print( contexto )
        print( '-'*100 )
        print( response_gt )'''

    resp_elastic = atualizar_elastic( id,f'{complexity_score:.2f}',f'{nli_val:.2f}', f'{sim_val:.2f}', f'{response_saf:.2f}',resposta )
    print( f'-> Elastic: {resp_elastic['result']}' )

    diff_time('-> Tempo total: ', inicio_global)     

async def run_batch():
    
    print( '\n--- inicio benchmark ---') 

    inicio = time.time()

    query = {
        "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks"],
        "query"     : {"match_all":{}}, 
        "size"      : 1500
    }
   
    resp = elastic_client.search(index="perguntas", body=query)

    total = len(resp["hits"]["hits"])

    for index, item in enumerate(resp["hits"]["hits"],start=1): 

        id       = item['_id']
        pergunta = item['_source']['pergunta']

        try:           
            resposta = item['_source']['saf_grafo_v2']    
            print('-> next...')   
        except Exception as erro:
            await main(id,'5511993891773',pergunta,'grafo',10,5,False,[],False)      

        print ( f'{index} de {total}')
        print( f'[red] --- fim ---' )

    diff_time('\n-> fim benchmark: ', inicio)

if __name__ == "__main__":   
    asyncio.run( run_batch() )
