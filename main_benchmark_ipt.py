
import csv,os,sys,time,asyncio,json

from rich import print

from src.services   import keywords_create, graph_search, vector_search, response_create, class_extraction
from src.utils      import diff_time
from src.evaluation import saf, score_dynamic_gt, sim, nli
from src.output     import csv_create
from src.config     import settings

def salvar_resultados_csv(id, status, complexity, nli, sim, saf):
    
    """Salva os resultados em um arquivo CSV."""

    filename = "benchmark_ipt.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([
                "ID", "Status", "Complexity", "NLI", "SIM", "SAF"
            ])
        
        writer.writerow([
            id, status, complexity, nli, sim, saf
        ])

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

    palavras_chave   = expantion['keywords']
    complexity_score = expantion['complexity_score']
    query_canonical  = expantion['query_expansion']['canonical']

    print( f'\n-> cononical fact: {query_canonical}' )
    print( f'\n-> rewriting: {palavras_chave}' )
    print( '-'*100,'\n' )

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
        contexto  = recuperacao['response']
        knowledge = recuperacao['dataset']  
    
    else: 
        recuperacao = vector_search(palavras_chave,query_canonical,'documentos',user_id,retrieval_size)
        contexto  = recuperacao['response']
        knowledge = recuperacao['dataset']    

    diff_time('\n-> #2 buscar dados, OK: ', inicio)
    
    if debug_all or 'retriever' in debug_one:
        print( f'[yellow]// Debug contexto  {retrieval}:\n{recuperacao["response"]}' )
        print( f'[yellow]// Debug knowledge {retrieval}:\n{recuperacao["dataset"]}' )

    # response
    #--------------------------------------------------------------------------

    inicio = time.time()

    response_llm    = await asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'maritaca')
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

        output_list = [output] if isinstance(output, str) else output

        for out_type in output_list:

            match out_type:

                case 'elastic':
                    elastic_id      = ''
                    elastic_index   = ''
                    elastic_update_field(elastic_index,elastic_id,complexity_score,response_saf,nli_val,sim_val,response_llm,'','')

                case 'csv': 
                    csv_create('teste_1.csv',retrieval,pergunta,complexity_score,response_saf,nli_val,sim_val,response_llm,'ollama')       
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

    if resposta !='':
        salvar_resultados_csv(id, status, f'{complexity_score:.2f}', f'{nli_val:.2f}', f'{sim_val:.2f}', f'{response_saf:.2f}' )
        print('-> Resultados salvos em CSV')
    else:
        print('ERRO: resposta vazia')   

    diff_time('-> Tempo total: ', inicio_global) 
    print( f'[red] --- fim ---\n' )

async def run_batch():
    
    print( '\n--- inicio benchmark ---') 

    inicio = time.time()

    # Carregar perguntas e respostas do arquivo JSON
    with open('qa_lgpd.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    faq_list = data.get('faq', [])
    total    = len(faq_list)

    for index, item in enumerate(faq_list, start=1):
        
        pergunta    = item['pergunta']
        resposta_gt = item['resposta']
        id          = index  # Usar o índice como ID

        print( pergunta )

        await main(id, resposta_gt, '5511993891773', pergunta, 'grafo', 10, False, [], None)

        print(f'{index} de {total}')
        print(f'[red] --- fim ---')

if __name__ == "__main__":   

    print('Iniciando benchmark...')
    asyncio.run( run_batch() )