
import numpy as np, torch
from scipy.special  import softmax
from rich           import print

from sentence_transformers import util

from src.config import settings
from src.models import embedding_model, nli_tokenizer, nli_model, nli_config

def sim(reference,candidate):
    
    emb_gold = embedding_model.encode(reference)
    emb_cand = embedding_model.encode(candidate)
    
    sim = util.cos_sim(emb_gold, emb_cand).item()   
    
    if sim >= 0.85:
        return {"status":"EXCELLENT (approved)", "score":float(sim)}
    elif sim >= 0.75:
        return {"status":"GOOD (approved)", "score":float(sim)}
    elif sim >= 0.65:
        return {"status":"FAIR (review)", "score":float(sim)}
    elif sim >= 0.50:
        return {"status":"POOR (failed)", "score":float(sim)}
    else:
        return {"status":"TERRIBLE (completely wrong)", "score":float(sim)}

def nli(reference,candidate):

    model_input = nli_tokenizer(
        *([reference],[candidate]), 
        padding=True,
        truncation=True,
        max_length=512, 
        return_tensors="pt"
    )

    with torch.no_grad():

        output  = nli_model(**model_input)
        scores  = output[0][0].detach().numpy()
        scores  = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        score_por_label = {nli_config.id2label[i]: scores[i] for i in range(len(scores))}

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
