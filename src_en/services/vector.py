from src_en.config     import settings
from src_en.utils.text import normalize

from langchain_openai import OpenAIEmbeddings

def vector_search(palavras_chave, pergunta, index_name, user_id, retrieval_size):

    print('--> search vector')

    resp_toon = ''
    knowledge_base = {}

    try:

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )

        query_embedding = embeddings.embed_query(palavras_chave) 

        script_query = {
            "size": retrieval_size,
            "_source": ["file_url", "id_usuario", "id_externo", "texto"],
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 50,
                "num_candidates": 500,         
                "filter": { "term": { "id_usuario": user_id } }
            },
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": pergunta,
                                "fields": ["texto"],
                                 "boost": 0.3
                            }
                        }
                    ],
                    "filter": [{ "term": { "id_usuario": user_id } }]
                }
            }                 
        }

        resp = settings.elastic_client.search(
            index=index_name,
            body=script_query
        )

        candidates = resp.get('hits', {}).get('hits', [])
        
        knowledge_base = {}
        
        linhas_csv = ['id_chunk;score;texto'] 

        for item in candidates:
            
            _id     = item.get('_id')
            _score  = item.get('_score')
            source  = item.get('_source', {})
            
            texto_rico  = normalize(source.get('texto', ''))            
            texto_limpo = texto_rico.replace(';', ',')

            linhas_csv.append(f"{_id};{_score};{texto_limpo}")
            
            knowledge_base[_id] = texto_limpo

        resp_toon = "\n".join(linhas_csv)

        return {
            'status': 'OK',
            'response': resp_toon,
            'dataset': knowledge_base
        }

    except Exception as e:
        
        error_msg = f"Elastic/Embedding ERROR: {str(e)}"
        print(f"--> {error_msg}")
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }