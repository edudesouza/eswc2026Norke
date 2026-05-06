
import os,sys,time,asyncio,json,re

from rich import print

from src.services   import keywords_create, graph_search, vector_search, response_create, ground_truth, class_extraction
from src.utils      import diff_time
from src.evaluation import saf, score_dynamic_gt, sim, nli
from src.output     import elastic_update_field, csv_create
from src.config     import settings

from deepeval           import evaluate as ev_deep
from deepeval.test_case import LLMTestCase
from deepeval.metrics   import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models    import OllamaModel
from langchain_together import ChatTogether

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from dotenv import load_dotenv
load_dotenv()

os.environ['CONFIDENT_METRIC_LOGGING_VERBOSE'] = '0'
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "200"

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def build_retrieval_context(items, top_k=100, max_chars=1500, prefix_ids=True):
    
    # aceita: str, dict, list[dict]
    if items is None:
        return []

    # se já for string: devolve como 1 contexto (ou quebra em blocos se quiser)
    if isinstance(items, str):
        txt = normalize_ws(items)
        return [txt[:max_chars] + ("..." if len(txt) > max_chars else "")] if txt else []

    # se for dict: converte para list[dict] usando key como id e value como texto
    if isinstance(items, dict):
        items = [{"id_chunk": k, "texto_chunk": v, "score": 0} for k, v in items.items()]

    # agora assume list/iterável de dicts
    items_sorted = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    seen = set()
    out = []

    for it in items_sorted:
        raw = it.get("texto_chunk") or it.get("descricao_regra") or it.get("texto_regra") or ""
        txt = normalize_ws(raw)
        if not txt:
            continue

        if prefix_ids:
            ident = it.get("id_chunk") or it.get("entidade") or ""
            if ident:
                txt = f"[{ident}] {txt}"

        if len(txt) > max_chars:
            txt = txt[:max_chars] + "..."

        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(txt)
        if len(out) >= top_k:
            break

    return out

def carregar_perguntas(json_path):
    """Carrega perguntas do arquivo JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

#async def main(id,pergunta,retrieval='grafo',retrieval_size=5,size_gt=5,debug_all=False,debug_one=None,output=None,threshold=0.60):
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

    #palavras_chave = 'Churrasco, Uso de churrasqueira, Regulamento do condomínio, Permissão para churrasco'

    # quanto maior a similaridade mais próximo a um tema único
    if complexity_score<0.55 :
        print( f'-> complexidade alta: {complexity_score:.2f}' )
        retrieval_size = 20
        size_gt = 10
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

    # ground truth and response
    #--------------------------------------------------------------------------

    inicio = time.time() 

    response_llm = await asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'maritaca')
    resposta     = response_llm['resposta']

    print( f'\nLLM: {resposta}' )
    print( f'GT:  {resposta_gt}' )

    diff_time('\n-> #3 ground truth e resposta OK: ', inicio)

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

    #deepeval
    resolvedor = "gpt-oss:120b-cloud"
    avaliador  = "gpt-oss:120b-cloud"

    print( f'\n-> resolvedor: {resolvedor}' )
    print( f'-> avaliador: {avaliador}' )

    model = OllamaModel(
        model = avaliador,
        base_url = "http://localhost:11434",
        temperature=0
    )
    
    answer_relevancy = AnswerRelevancyMetric(model=model,include_reason=True)
    faithfulness     = FaithfulnessMetric(model=model,include_reason=True)

    retrieval_ctx = build_retrieval_context(recuperacao["dataset"], top_k=20)          

    test_case = LLMTestCase(
        input             = pergunta,
        actual_output     = resposta,
        expected_output   = resposta_gt,
        retrieval_context = retrieval_ctx,      
        context           = [recuperacao["response"]]
    )

    try:
        answer_relevancy.measure(test_case)
        print("- Relevancia: ", answer_relevancy.score)
        print("- Reason: ", answer_relevancy.reason)
        print('-'*100)
    except Exception as erro:
        print( f'ERRO relevancia: {erro}' )

    try:
        faithfulness.measure(test_case)
        print("- Confiabilidade: ", faithfulness.score)
        print("- Reason: ", faithfulness.reason)
        print('-'*100)
    except Exception as erro:
        print( f'ERRO confiabilidade: {erro}' )
  

    if output:
        csv_create(
            output, retrieval, pergunta, resposta_gt, response_llm, 
            complexity_score, response_saf, nli_val, sim_val,answer_relevancy.score,faithfulness.score,
            'maritaca'
        ) 
   
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
    
    diff_time('-> Tempo total: ', inicio_global) 
    print( f'[red] --- fim ---\n' )

async def run_batch(json_path='lgpd_perguntas_100_revisado.json', output_csv='lgpd_perguntas_100_revisado_grafoV2b_maritaca_deepeval.csv'):

    print('\n--- inicio benchmark ---')

    inicio = time.time()

    perguntas = carregar_perguntas(json_path)
    total = len(perguntas)

    #for index, item in enumerate(perguntas[0:1], start=0): rodar só a primeira pergunta
    for index, item in enumerate(perguntas[95:], start=95):

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
