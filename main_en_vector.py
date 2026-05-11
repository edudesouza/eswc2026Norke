
import os,sys,time,asyncio,json,re

from rich import print

from src_en.services   import keywords_create, graph_search, vector_search, response_create, ground_truth, class_extraction
from src_en.utils      import diff_time
from src_en.evaluation import saf, score_dynamic_gt, sim, nli
from src_en.output     import elastic_update_field, csv_create
from src_en.config     import settings

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

# https://openscience.adaptcentre.ie/ontologies/GDPRtEXT/deliverables/docs/ontology.xml
# https://github.com/coolharsh55/GDPRtEXT
# https://w3c.github.io/dpv/2.1/dpv/
# https://gdpr-info.eu/art-4-gdpr/

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _build_retrieval_context(items, top_k=100, max_chars=1500, prefix_ids=True):
    
    # ordena por score desc (se houver)
    items_sorted = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    seen = set()
    out = []

    for it in items_sorted:
        # escolha do texto: Chunk -> texto_chunk; Regra -> descricao_regra
        raw = it.get("texto_chunk") or it.get("descricao_regra") or ""
        txt = normalize_ws(raw)
        if not txt:
            continue

        # opcional: prefixar id_chunk p/ rastreabilidade
        if prefix_ids:
            ident = it.get("id_chunk") or it.get("entidade") or ""
            if ident:
                txt = f"[{ident}] {txt}"

        # corta muito longo (evita estourar prompt)
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "..."

        # dedup
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(txt)
        if len(out) >= top_k:
            break

    return out

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

async def main(user_id,pergunta,resposta_gt,retrieval='grafo',retrieval_size=5,size_gt=5,debug_all=False,debug_one=None,output=None,threshold=0.60):  

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

    #expantion       = keywords_create(pergunta,'gpt-4.1',settings.OPENAI_API_KEY,None)
    expantion        = keywords_create(pergunta,'gemini-2.5-flash',settings.GEMINI_API_KEY,'')
    #expantion       = keywords_create(pergunta,'qwen3-next:80b-cloud',settings.OLLAMA_API_KEY,'http://localhost:11434')

    palavras_chave   = expantion['keywords']
    complexity_score = expantion['complexity_score']
    query_canonical  = expantion['query_expansion']['canonical']
    article          = expantion['query_expansion']['article']
    chapter          = expantion['query_expansion']['chapter']

    print( f'\n-> cononical fact: {query_canonical}' )
    print( f'-> breadcrumb: {article}, {chapter}' )
    print( f'\n-> rewriting: {palavras_chave}' )
    print( '-'*100,'\n' )

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
        recuperacao = graph_search(class_rules,expantion,palavras_chave,pergunta,user_id,retrieval_size)
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

    #task_ground_truth           = asyncio.to_thread(ground_truth,contexto,pergunta,palavras_chave,query_canonical,'ollama',size_gt)
    #task_response_llm           = asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'gpt')
    #response_gt, response_llm   = await asyncio.gather(task_ground_truth, task_response_llm)
    #resposta                    = response_llm['resposta']

    response_llm = await asyncio.to_thread(response_create,palavras_chave,pergunta,contexto,'ollama')
    
    try:
        
        resposta      = response_llm['answer']
        resposta_full = response_llm['full_answer']
        grounding     = response_llm['chunks']
        #key_snippet  = response_llm['key_snippet']

        print( f'\nQuestion:    {pergunta}' )
        print( f'\nLLM full:    {resposta_full}' )
        print( f'\nLLM:         {resposta}' )
        print( f'Grounding:     {grounding}' )
        print( f'\nGold answer: {resposta_gt}' )

    except Exception as erro:
        print( f'ERRO LLM: {erro}' )
        print(response_llm)
        sys.exit()
    

    diff_time('\n-> #3 ground truth e resposta OK: ', inicio)

    if debug_all or 'ground_truth' in debug_one:
        print( f'[yellow]// Debug GT:\n{resposta_gt}\n' )

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

    resolvedor = "gpt-oss:120b-cloud"
    avaliador  = "gpt-oss:120b-cloud"

    print( f'\n-> resolvedor: {resolvedor}' )
    print( f'-> avaliador: {avaliador}' )

    model = OllamaModel(
        model = avaliador,
        base_url = "http://localhost:11434",
        temperature=0
    )

    #deepeval
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

    diff_time('-> Tempo total: ', inicio_global) 
    print( f'[red] --- fim ---\n' )

if __name__ == "__main__":
    
    _pergunta = "What are the grounds for a data subject to request erasure of their personal data according to Article 17?"
    _resposta = "The grounds for a data subject to request erasure of their personal data according to Article 17 are: (a) the personal data are no longer necessary in relation to the purposes for which they were collected or otherwise processed; (b) the data subject withdraws consent on which the processing is based according to point (a) of Article 6(1), or point (a) of Article 9(2), and where there is no other legal ground for the processing; (c) the data subject objects to the processing pursuant to Article 21(1) and there are no overriding legitimate grounds for the processing, or the data subject objects to the processing pursuant to Article 21(2); (d) the personal data have been unlawfully processed; (e) the personal data have to be erased for compliance with a legal obligation in Union or Member State law to which the controller is subject; (f) the personal data have been collected in relation to the offer of information society services referred to in Article 8(1)." 

    _pergunta = "What measures should the controller take to provide information to the data subject under Articles 13 and 14 and any communication under Articles 15 to 22 and 34?"
    _resposta = "The controller shall take appropriate measures to provide any information referred to in Articles 13 and 14 and any communication under Articles 15 to 22 and 34 relating to processing to the data subject in a concise, transparent, intelligible and easily accessible form, using clear and plain language, in particular for any information addressed specifically to a child. The information shall be provided in writing, or by other means, including, where appropriate, by electronic means. When requested by the data subject, the information may be provided orally, provided that the identity of the data subject is proven by other means."
    
    _pergunta = 'What is the subject matter of Article 1 of the GDPR Regulation?'
    _resposta = 'Article 1 lays down rules relating to the protection of natural persons with regard to the processing of personal data and rules relating to the free movement of personal data.'

    # GPT
    _pergunta = 'My 15-year-old son s school requires facial recognition for entry and attendance. They say it is for security and asked for consent, but students who refuse cannot enter. The system is run by an external company that also uses the data to improve its technology. Since it is mandatory and for safety, I assume this is allowed under GDPR, correct?'
    _resposta = 'The use of facial recognition for a 15-year-old student involves the processing of biometric data, which is classified as a special category of personal data under the GDPR and is generally prohibited unless a valid legal basis applies. Consent in this context is unlikely to be valid, as it must be freely given, and denying access to the school in case of refusal indicates coercion. Additionally, children are afforded specific protection, and their consent may not be sufficient depending on national age thresholds. The involvement of a third-party company and the use of data for additional purposes further require a clear legal basis, transparency, and strict purpose limitation. Therefore, the described processing is likely not compliant with GDPR.'
    
    # Claude
    _pergunta = 'Our school wants to speed up student entry at the gate, so we thought about using facial recognition. Since it is only for internal access control and parents already signed the enrollment contract, I believe consent is already covered. Can we share this data with the camera system provider?'
    _resposta = 'This request involves three GDPR violations that the enrollment contract cannot cover. Facial recognition produces biometric data under Article 9 — a special category requiring explicit, purpose-specific consent. A generic contract clause is insufficient. Since students are minors, Article 8 requires parental consent that is freely given and specific to biometric processing — not bundled into enrollment terms. Sharing data with the camera provider constitutes a third-party transfer under Article 28, requiring a formal Data Processing Agreement regardless of internal intent. The school must obtain explicit parental consent for biometrics, sign a DPA with the provider, and establish a lawful basis under Article 9 before proceeding.'

    _pergunta = 'Can a school require facial recognition?'
    _resposta = 'No, Article 9 GDPR prohibits the processing of special categories of personal data without a valid legal basis and permission to process special categories of personal data with explicit consent'

    pergunta = 'Under what circumstances can a supervisory authority confer investigative powers on the members or staff of a seconding supervisory authority?'
    resposta = 'A supervisory authority may, in accordance with Member State law, and with the seconding supervisory authority s authorisation, confer powers, including investigative powers on the seconding supervisory authority s members or staff involved in joint operations or, in so far as the law of the Member State of the host supervisory authority permits, allow the seconding supervisory authority s members or staff to exercise their investigative powers in accordance with the law of the Member State of the seconding supervisory authority.'
    
    # debug_one [query,retriever,ground_truth]
    # user_id,pergunta,retrieval=grafo|vetor,retrieval_size=5,size_gt=5,debug_all=False,debug_one=None,output=None,threshold=0.60):
    asyncio.run( main('5511993891773',pergunta,resposta,'vetor',20,5,False,None,None,0.75) )

