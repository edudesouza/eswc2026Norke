import numpy as np

from sklearn.cluster        import KMeans
from sentence_transformers  import SentenceTransformer, util

from src.config import settings

model = SentenceTransformer(settings.EMB_MODEL_NAME)

def keyword_complexity(query_5w3h):
    
    # Corrigir se vier como lista
    if isinstance(query_5w3h, list):
        query_5w3h = query_5w3h[0]
    
    # Extrair componentes (removendo valores vazios ou None)
    components = [v for v in query_5w3h.values() if v and str(v).strip()]
    
    if len(components) < 2:
        return {
            'is_coherent': True,
            'coherence_score': 1.0,
            'themes': {0: components},
            'n_components': len(components)
        }
    
    # Calcular embeddings
    embeddings = model.encode(components)
    
    # Coesão geral (matriz de similaridade)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    
    # Média excluindo diagonal (auto-similaridade)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    coherence_score = sim_matrix[mask].mean().item()
    
    # Detectar clusters se coesão baixa
    if coherence_score < 0.6 and len(components) >= 2:
        n_clusters = min(3, len(components))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Agrupar por tema
        theme_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in theme_groups:
                theme_groups[cluster_id] = []
            theme_groups[cluster_id].append(components[i])
        
        return {
            'is_coherent': False,
            'coherence_score': coherence_score,
            'themes': theme_groups,
            'n_themes': len(theme_groups),
            'n_components': len(components)
        }
    
    return {
        'is_coherent': True,
        'coherence_score': coherence_score,
        'themes': {0: components},
        'n_themes': 1,
        'n_components': len(components)
    }