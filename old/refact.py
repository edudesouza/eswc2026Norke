class CondoEvaluator:

    def __init__(self, sbert_model, nli_model, llm_client):
        
        self.sbert = sbert_model
        self.nli = nli_model
        self.llm = llm_client

    def avaliar(self, resposta_llm, ground_truth_doc):
        
        # 1. SAF (Breakdown): Usa LLM para extrair fatos atômicos
        # Ex: "Não pode fumar" | "Multa de 100%"

        fatos = self._extrair_fatos_atomicos(resposta_llm) 

        score_total = 0
        detalhes = []

        for fato in fatos:
            
            # 2. SIM (Retrieval/Filter): Acha o melhor trecho no Ground Truth
            melhor_trecho = self._achar_melhor_contexto_com_sbert(fato, ground_truth_doc)
            
            # 3. NLI (Verification): O Juiz rigoroso
            score_nli = self.nli.predict(melhor_trecho, fato)
            
            # Penalidade personalizada (aquela fórmula que criamos)
            if score_nli < 0.5: # Contradição
                 print(f"ALERTA DE MENTIRA: {fato}")
            
            score_total += score_nli
            detalhes.append((fato, score_nli))

        return score_total / len(fatos)