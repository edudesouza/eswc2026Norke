
import time,asyncio,json

from rich import print

from src.services   import keywords_create, graph_search, vector_search, response_create, class_extraction
from src.utils      import diff_time
from src.evaluation import saf, score_dynamic_gt, sim, nli
from src.output     import csv_create
from src.config     import settings

def carregar_perguntas(json_path):
    """Carrega perguntas do arquivo JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def main(id,resposta_gt,user_id,pergunta,retrieval='grafo',retrieval_size=5,debug_all=False,debug_one=None,output=None,threshold=0.60):  

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
    
    expantion        = keywords_create(pergunta,'gpt-4.1',settings.OPENAI_API_KEY,None)
    #expantion        = keywords_create(pergunta,'gemini-2.5-pro',settings.GEMINI_API_KEY,'')

    palavras_chave   = expantion['keywords']
    complexity_score = expantion['complexity_score']
    query_canonical  = expantion['query_expansion']['canonical']

    print( f'\n-> cononical fact: {query_canonical}' )
    print( f'\n-> rewriting: {palavras_chave}' )
    print( '-'*100,'\n' )

    #palavras_chave = 'Churrasco, Uso de churrasqueira, Regulamento do condomínio, Permissão para churrasco'

    # quanto maior a similaridade mais próximo a um tema único
    if complexity_score<0.55 :
        print( f'-> complexidade alta: {complexity_score:.2f}' )
        retrieval_size = 20
    else:
        print( f'-> complexidade baixa: {complexity_score:.2f}' )

    diff_time('\n-> #1 expandir query OK: ', inicio)

    if debug_all or 'query' in debug_one: 
        print( f'[yellow]// Debug keywords:' )
        
        for kw_key,kw_value in expantion['query_expansion'].items():        
            if kw_key and kw_value not in ['NULL','null']: 
                print( f'[yellow]- {kw_key}: {kw_value}' )      

    # retriever
    #--------------------------------------------------------------------------
    
    inicio = time.time() 

    if retrieval=='grafo':        
        class_rules = class_extraction(palavras_chave,pergunta,query_canonical,'gpt')
        recuperacao = graph_search(class_rules,palavras_chave,pergunta,user_id,retrieval_size)
        contexto    = recuperacao['response']
        knowledge   = recuperacao['dataset']  
    
    else: 
        recuperacao = vector_search(palavras_chave,query_canonical,'documentos',user_id,retrieval_size)
        contexto    = recuperacao['response']
        knowledge   = recuperacao['dataset']    

    diff_time('\n-> #2 buscar dados, OK: ', inicio)
    
    if debug_all or 'retriever' in debug_one:
        print( f'[yellow]// Debug contexto  {retrieval}:\n{recuperacao["response"]}' )
        print( f'[yellow]// Debug knowledge {retrieval}:\n{recuperacao["dataset"]}' )

    # response
    #--------------------------------------------------------------------------

    inicio = time.time()

    response_llm    = await asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'gpt')
    resposta        = response_llm['resposta']

    print( f'\nLLM: {resposta}' )
    print( f'GT:  {resposta_gt}' )

    diff_time('\n-> #3 resposta OK: ', inicio)

    # metrics
    #--------------------------------------------------------------------------
    
    inicio  = time.time()    
    
    saf_score   = saf(knowledge,resposta,pergunta,debug_all)
    task_sim    = asyncio.to_thread(sim,resposta_gt,resposta)
    task_nli    = asyncio.to_thread(nli,resposta_gt,resposta)
    response_saf, score_sim_result, score_nli_result = await asyncio.gather(saf_score, task_sim, task_nli)

    nli_val = score_nli_result['score']
    sim_val = score_sim_result['score']

    print( f"-> nli: {nli_val:.2f}" )
    print( f"-> sim: {sim_val:.2f}" )
    print( f'-> saf: {response_saf:.2f}' )
    
    diff_time('\n-> #4 factualidade e comparação: ', inicio)  

    if output:
        csv_create(
            output, retrieval, pergunta, complexity_score,
            response_saf, nli_val, sim_val,
            response_llm, 'maritaca'
        )       
    status = ''
    
    # 1. CASO OURO: IA fiel ao documento e alinhada ao gabarito
    if response_saf > threshold and (nli_val > 0.75 or sim_val > threshold):
        print('[green]*** Resposta APROVADA: Fiel ao documento e ao GT ***\n')
        status = 'aprovada'

    # 2. CASO DIVERGÊNCIA: IA fiel ao documento, mas longe do GT
    elif response_saf > threshold:
        print('[yellow]*** REVISÃO NECESSÁRIA: IA fiel ao documento, mas diverge do GT ***\n')
        status = 'revisao'

    # 3. CASO ALUCINAÇÃO: IA não encontrou base no documento
    else:
        print('[red]*** Resposta NEGADA: alto grau de ambiguidade: retriever, llm ou gt problemático ***\n')
        status = 'negada'

    '''print( palavras_chave )
    print( '-'*100 )
    print( contexto )
    print( '-'*100 )
    print( response_gt )'''

    diff_time('-> Tempo total: ', inicio_global) 
    print( f'[red] --- fim ---\n' )

# 100_perguntas_lgpd.json
# lgpd_perguntas_ambiguas_100.json
async def run_batch(json_path='100_perguntas_lgpd.json', output_csv='100_perguntas_lgpd_grafoV2_o4.csv'):

    print('\n--- inicio benchmark ---')

    inicio = time.time()

    perguntas = carregar_perguntas(json_path)
    #perguntas = dados_json['faq']
    total = len(perguntas)

    for index, item in enumerate(perguntas[64:], start=64):

        id          = str(index)
        pergunta    = item['question']
        resposta_gt = item['answer']

        print( pergunta )

        await main(id, resposta_gt, '5511993891773', pergunta,'grafo', 10, False, [], output_csv)

        print(f'{index} de {total}')
        print(f'[red] --- fim ---')

    diff_time('\n-> fim benchmark: ', inicio)

if __name__ == "__main__":   

    print('Iniciando benchmark...')
    asyncio.run( run_batch() )
