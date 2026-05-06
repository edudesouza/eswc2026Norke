
import os,sys,time,asyncio,json,re, csv

from rich import print

from src_en.evaluation import saf, score_dynamic_gt, sim, nli
from src_en.config     import settings
from src_en.output     import csv_create

from deepeval           import evaluate as ev_deep
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics   import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.models    import OllamaModel, AnthropicModel, GeminiModel, GPTModel

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from dotenv import load_dotenv
load_dotenv()

os.environ['CONFIDENT_METRIC_LOGGING_VERBOSE'] = '0'
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "200"

# https://www.kaggle.com/datasets/iuliabunescu23/gdpr-qa-test-dataset


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

async def metricas(id, pergunta, resposta_gt, resposta, model):

    debug_all = False
    knowledge = {"pergunta": resposta_gt}

    # calcular SAF, NLI, SIM
    saf_score   = saf(knowledge,resposta,"GDPR - law about data privacy and protection",debug_all)
    task_sim    = asyncio.to_thread(sim,resposta_gt,resposta)
    task_nli    = asyncio.to_thread(nli,resposta_gt,resposta)
    response_saf, score_sim_result, score_nli_result = await asyncio.gather(saf_score, task_sim, task_nli)

    nli_val = score_nli_result['score']
    sim_val = score_sim_result['score']

    print( f"-> nli: {nli_val:.2f}" )
    print( f"-> sim: {sim_val:.2f}" )
    print( f'-> saf: {response_saf:.2f}' )

    # deep eval
    #avaliador  = "gpt-oss:120b-cloud"
    #avaliador = "deepseek-v3.2:cloud"
    #avaliador = "mistral-large-3:675b-cloud"      
    
    answer_relevancy = AnswerRelevancyMetric(model=model, threshold=0.70,include_reason=True)

    legal_correctness = GEval(
        name="Legal Correctness GDPR",
        criteria="""
        STRICT JSON OUTPUT REQUIRED.

        You MUST return ONLY a valid JSON object.
        Do NOT include markdown, explanations, or backticks.

        Format:
        {
            "score": <float between 0 and 1>,
            "reason": "<short explanation>"
        }

        Evaluate whether the actual answer is legally correct according to the expected answer.

        Consider:
        - Whether the legal conclusion is correct.
        - Whether the relevant GDPR rule, right, obligation, exception, or condition is covered.
        - Whether the answer omits an essential legal requirement.
        - Whether the answer introduces unsupported or legally incorrect claims.
        - Different wording is acceptable if the legal meaning is equivalent.

        Penalize:
        - Wrong legal basis.
        - Missing mandatory condition.
        - Overgeneralization.
        - Confusing controller, processor, data subject, or supervisory authority duties.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.75
    )

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        expected_output=resposta_gt
    )

    deepeval_relevancy_score = None
    deepeval_relevancy_reason = ""

    deepeval_legal_score = None
    deepeval_legal_reason = ""

    try:

        answer_relevancy.measure(test_case)
        deepeval_relevancy_score = answer_relevancy.score
        deepeval_relevancy_reason = answer_relevancy.reason

        print(f"-> DeepEval Relevancy: {deepeval_relevancy_score}")
        print(f"-> Reason: {deepeval_relevancy_reason}")
        print("-" * 100,'\n')

    except Exception as erro:
        print(f"[red]ERRO relevância:[/red] {erro}")

    try:

        legal_correctness.measure(test_case)
        deepeval_legal_score = legal_correctness.score
        deepeval_legal_reason = legal_correctness.reason

        print(f"-> DeepEval Legal Correctness: {deepeval_legal_score}")
        print(f"-> Reason: {deepeval_legal_reason}")
        print("-" * 100,'\n')

    except Exception as erro:
        print(f"[red]ERRO legal correctness:[/red] {erro}")

    output='gpdr_kaggle_basemodel_deepeval.csv'

    csv_create(
        output, id, 'grafo', pergunta, resposta_gt, resposta, 
        0, response_saf, nli_val, sim_val, answer_relevancy.score,legal_correctness.score,
        'kaggle - basemodel'
    )

    print(f'-> resultados salvos, {id}\n')

    return

async def run_batch():

    json_path = '_GDPR_qa_test_dataset_v2.csv'

    with open(json_path, 'r', encoding='utf-8') as f:
        perguntas = list(csv.reader(f))

    '''model = OllamaModel(
        model = avaliador,
        base_url = "http://localhost:11434",
        temperature=0
    )'''

    '''model = GPTModel(
        model='gpt-4.1', 
        api_key=settings.OPENAI_API_KEY,
        temperature=0
    )'''

    model = GeminiModel(
        model='gemini-2.5-flash', 
        api_key=settings.GEMINI_API_KEY,
        temperature=0
    )

    for index, item in enumerate(perguntas[1:], start=1):
    #for index, item in enumerate(perguntas[2:3], start=2):  #rodar só a primeira pergunta

        # Question,Correct Answer,Quantized Base Model Answer,Fine-tuned Model Answer,Base Model Answer
        pergunta           = item[0]
        resposta_gt        = item[1] 
        resposta_quantized = item[2]
        resposta_finetuned = item[3]
        resposta_basemodel = item[4]

        print(f'Question:       {pergunta}\n')
        print(f'Gold standard:  {resposta_gt}\n')
        print(f'Answer:         {resposta_quantized}\n')

        await metricas(index, pergunta, resposta_gt, resposta_basemodel, model)

if __name__ == "__main__":   
    
    print('\nIniciando benchmark...\n')
    asyncio.run( run_batch() )



