
import numpy as np, torch
from scipy.special  import softmax
from rich           import print

from transformers           import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.utils     import logging as hf_logging
from sentence_transformers  import SentenceTransformer, util

from bert_score import BERTScorer
from bert_score import score

from src.config import settings

sim_model     = SentenceTransformer(settings.EMB_MODEL_NAME)
nli_tokenizer = AutoTokenizer.from_pretrained(settings.NLI_MODEL_NAME, use_fast=False)
nli_model     = AutoModelForSequenceClassification.from_pretrained(settings.NLI_MODEL_NAME)
config        = AutoConfig.from_pretrained(settings.NLI_MODEL_NAME)

def sim(reference,candidate):

    print("-> SIM")
    
    emb_gold = sim_model.encode(reference)
    emb_cand = sim_model.encode(candidate)
    
    sim = util.cos_sim(emb_gold, emb_cand).item()   
    
    if sim >= 0.85:
        return {"status":"EXCELENTE (aprovada)", "score":float(sim)}
    elif sim >= 0.75:
        return {"status":"BOA (aprovada)", "score":float(sim)}
    elif sim >= 0.65:
        return {"status":"RAZOÁVEL (revisar)", "score":float(sim)}
    elif sim >= 0.50:
        return {"status":"RUIM (reprovar)", "score":float(sim)}
    else:
        return {"status":"PÉSSIMA (completamente errada)", "score":float(sim)}

def nli(reference,candidate):

    print("-> NLI")

    model_input = nli_tokenizer(
        *([referencia],[candidato]), 
        padding=True, 
        return_tensors="pt"
    )

    with torch.no_grad():

        output  = nli_model(**model_input)
        scores  = output[0][0].detach().numpy()
        scores  = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        score_por_label = {config.id2label[i]: scores[i] for i in range(len(scores))}

        entailment      = score_por_label.get('entailment', 0)
        neutral         = score_por_label.get('neutral', 0)
        contradiction   = score_por_label.get('contradiction', 0)     

        score_final = entailment - (contradiction * 2.5) - (neutral * 0.6)
        
        return {
            "score":float(score_final),
            "entailment":float(entailment), 
            "contradiction":float(contradiction), 
            "neutral":float(neutral)
        }

def bertscore(reference,candidate):

    print("-> Bert score")

    scorer = BERTScorer(
        #model_type="neuralmind/bert-base-portuguese-cased", num_layers=12,
        model_type=settings.BERTSCORE_MODEL_NAME, num_layers=24,
        #model_type="rufimelo/Legal-BERTimbau-sts-large-ma-v3", num_layers=12,
        lang="pt", 
        rescale_with_baseline=False,    
    )

    P, R, F1 = scorer.score(candidato, referencia, verbose=False)

    
    precision = P.item()
    recall    = R.item()
    f1        = F1.item()

    return {
        "precision":float(f1),
        "recall":float(recall),
        "f1":float(f1)
    }
