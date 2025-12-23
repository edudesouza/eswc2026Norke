
import json

from .metrics import sim, nli

def score_dynamic_gt(dynamic_gt:str, response_llm:str, threshold: float = 0.60):    

    print( '--> gt score ')

    try:
        gt_dict = json.loads(dynamic_gt)
    except Exception as erro:
        print( dynamic_gt )
        print( erro )
        exit()

    score_sim  = 0
    score_nli  = 0
    score_best = 0
    best_match = {
        "score_sim": {"score": 0},
        "score_nli": {"score": 0},
        "matched": 'No matches found'
    }

    candidates = gt_dict.get('perguntas_respostas', [])

    for item in candidates:        
        
        score_sim = sim( item['resposta'], response_llm ) 
        score_nli = nli( item['resposta'], response_llm )  
        gt_text   = item.get('resposta', '')   

        current = (score_sim['score'] + score_nli['score']) / 2

        if current > score_best:   
            score_best = current
            best_match = {"score_sim": score_sim,"score_nli": score_nli,"matched": gt_text}
    
    return best_match 

    '''if score_best>=threshold:
        return best_match 
    else:
        return {
            "score_sim": {"score": 0},
            "score_nli": {"score": 0},
            "matched": 'No matches found'
        }'''