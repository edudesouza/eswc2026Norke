
import csv,os,sys,time,asyncio

from rich import print

from elasticsearch  import Elasticsearch

from src.services   import keywords_create, graph_search, vector_search, response_create, ground_truth
from src.utils      import diff_time
from src.evaluation import saf, score_dynamic_gt
from src.output     import elastic_update_field, csv_create
from src.config     import settings

elastic_client = Elasticsearch( 
    settings.ELASTICSEARCH_HOST,basic_auth=(settings.ELASTICSEARCH_USER,settings.ELASTICSEARCH_PASS),verify_certs=False
)

def atualizar_elastic(id,complexity,nli,sim,saf,response):

    body = {
        "doc": {            
            "saf_vetor_v3":{
                "model":"kimi-k2:1t-cloud",
                "response":response,
                "complexity":complexity,
                "nli":nli,
                "sim":sim,
                "saf":saf
            }            
        },
        "doc_as_upsert": True
    }

    res = elastic_client.update(index="perguntas", id=id, body=body)
 
    return res

async def main(id,user_id,pergunta,retrieval='vetor',retrieval_size=5,size_gt=5,debug_all=False,debug_one=None,output=None,threshold=0.60):  

    print( f'[red] \n--- inicio ---' )

    if debug_one is None:
        debug_one = []

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
    expantion        = keywords_create(pergunta,'gemini-2.5-flash',settings.GEMINI_API_KEY)
    palavras_chave   = expantion['keywords']
    complexity_score = expantion['complexity_score']
    query_canonical  = expantion['query_expansion']['canonical']

    print( f'\n-> cononical fact: {query_canonical}' )

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

        for kw_key,kw_value in expantion['query_expansion'].items():        
            if kw_key and kw_value not in ['NULL','null']: 
                print( f'[yellow]- {kw_key}: {kw_value}' )

    # retriever
    #--------------------------------------------------------------------------
    
    inicio = time.time() 

    if retrieval=='grafo':

        recuperacao = graph_search(palavras_chave,pergunta,user_id,retrieval_size)
        contexto    = recuperacao['response']
        knowledge   = recuperacao['dataset']  
    
    else:  

        recuperacao = vector_search(palavras_chave,query_canonical,'documentos',user_id,retrieval_size)
        contexto    = recuperacao['response']
        knowledge   = recuperacao['dataset']    

    diff_time('\n-> #2 buscar dados, OK: ', inicio)
    
    if debug_all or 'retriever' in debug_one:
        print( f'[yellow]// Debug contexto  {retrieval}:\n{recuperacao['response']}' )
        print( f'[yellow]// Debug knowledge {retrieval}:\n{recuperacao['dataset']}' )

    # ground truth and response
    #--------------------------------------------------------------------------

    inicio = time.time() 

    task_ground_truth           = asyncio.to_thread(ground_truth,contexto,pergunta,palavras_chave,query_canonical,'ollama',size_gt)
    task_response_llm           = asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'ollama')
    response_gt, response_llm   = await asyncio.gather(task_ground_truth, task_response_llm)
    resposta                    = response_llm['resposta']

    print( f'\nLLM: {resposta}' )

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

    if output:

        output_list = [output] if isinstance(output, str) else output

        for out_type in output_list:

            match out_type:

                case 'elastic':
                    elastic_id      = ''
                    elastic_index   = ''
                    elastic_update_field(elastic_index,elastic_id,complexity_score,response_saf,nli_val,sim_val,response_llm,'','')

                case 'csv': 
                    csv_create('teste_1.csv',retrieval,pergunta,complexity_score,response_saf,nli_val,sim_val,response_llm,'ollama') 

    # 1. CASO OURO: IA fiel ao documento e alinhada ao gabarito
    if response_saf > threshold and (nli_val > threshold or sim_val > threshold):
        print('[green]*** Resposta APROVADA: Fiel ao documento e ao GT ***\n')

    # 2. CASO DIVERGÊNCIA: IA fiel ao documento, mas longe do GT
    elif response_saf > threshold and (nli_val <= threshold and sim_val <= threshold):
        print('[yellow]*** REVISÃO NECESSÁRIA: IA fiel ao documento, mas diverge do GT (GT Neutro ou IA Técnica?) ***\n')

    # 3. CASO ALUCINAÇÃO: IA não encontrou base no documento
    elif response_saf <= threshold:
        print('[red]*** Resposta NEGADA: alto grau de ambiguidade: retriver, llm ou gt problemático ***\n')

    # 4. CASO INCONSISTÊNCIA: Bate com o GT por sorte, mas não tem no documento
    else:
        print('[magenta]*** Resposta NEGADA: Inconsistência (Bate com GT, mas SAF baixo) ***\n')

    '''print( palavras_chave )
    print( '-'*100 )
    print( contexto )
    print( '-'*100 )
    print( response_gt )'''

    if resposta !='':
        resp_elastic = atualizar_elastic( id,f'{complexity_score:.2f}',f'{nli_val:.2f}', f'{sim_val:.2f}', f'{response_saf:.2f}',resposta )
        print( f'-> Elastic: {resp_elastic['result']}' )
    else:
        print('ERRO: resposta vazia')

    diff_time('-> Tempo total: ', inicio_global)     

async def run_batch():
    
    print( '\n--- inicio benchmark ---') 

    inicio = time.time()

    query = {
        "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks","saf_vetor_v2"],
        #"query"     : {"match_all":{}}, 
        "query": {
            "bool": {"must_not":{"exists": {"field": "saf_vetor_v3"}}}
        },
        "size"      : 1500
    }
   
    resp = elastic_client.search(index="perguntas", body=query)

    total = len(resp["hits"]["hits"])

    processar = 0

    for index, item in enumerate(resp["hits"]["hits"],start=1): 

        id       = item['_id']
        pergunta = item['_source']['pergunta']

        try:           
            resposta = item['_source']['saf_vetor_v3']    
            print('-> next...')   
        except Exception as erro:
            print( f'-> processar ({id})...' )  
            processar +=1 
            await main(id,'5511993891773',pergunta,'vetor',10,5,False,[],False)  

        print ( f'{index} de {total}, {processar}')
        print( f'[red] --- fim ---' )

    diff_time('\n-> fim benchmark: ', inicio)

if __name__ == "__main__":   
    asyncio.run( run_batch() )
