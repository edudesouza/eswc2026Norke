
import json

from .metrics import sim, nli

def score_dynamic_gt(dynamic_gt:str, response_llm:str, threshold: float = 0.75):    

    print( '--> gt score ')

    gt_dict = json.loads(dynamic_gt)

    best_score_sum       = 0
    best_match           = None

    candidates = gt_dict.get('perguntas_respostas', [])

    for item in candidates:
        
        score_sim = sim( item['resposta'], response_llm ) 
        score_nli = nli( item['resposta'], response_llm )  
        gt_text   = item.get('resposta', '')
   
        if score_sim['score'] > threshold and score_nli['score'] > threshold:
            
            total = score_sim['score'] + score_nli['score']
            
            if total  > best_score_sum:
                
                best_score_sum = total
               
                best_match = {
                    "score_sim": score_sim,
                    "score_nli": score_nli,
                    "matched": gt_text
                }

    if best_match:
        return best_match    

    else:       

        return {
            "score_sim": {"score":0},
            "score_nli": {"score":0},
            "matched": 'Nenhuma correspondência encontrada'
        } 
