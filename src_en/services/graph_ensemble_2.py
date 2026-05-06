
import requests, re, textwrap
from langchain_classic.schema               import BaseRetriever, Document
from langchain_classic.callbacks.manager    import CallbackManagerForRetrieverRun
from typing                                 import List, Dict, Optional
from requests.auth                          import HTTPBasicAuth

from src_en.config     import settings
from src_en.utils.text import normalize

# NormalizedScoreFusion

class GraphRulesRetriever(BaseRetriever):
    """Retriever para buscar REGRAS no grafo"""
    
    named_graph: str
    retrieval_size: int = 20
    
    class Config:
        """Configuração do Pydantic"""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str,
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Busca regras estruturadas no grafo"""
        
        keyword = re.sub(r'[\\/]+', ' ', query)
        
        sparql_query = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?score ?idChunk ?texto ?regraURI ?tipo ?descricao
            FROM <https://omc.co/graph/{self.named_graph}>
            WHERE {{
            {{
                SELECT ?regra (MAX(?s) AS ?score)
                WHERE {{
                ?q a luc-index:omc_full ;
                    luc:query '{keyword}' ;
                    luc:entities ?regra .
                ?regra luc:score ?s .
                }}
                GROUP BY ?regra
                ORDER BY DESC(?score)
                LIMIT 500
            }}

            ?regra a/rdfs:subClassOf* :Regra .
            OPTIONAL {{ ?regra a ?tipo }}
            ?regra :descricao ?descricao . FILTER(LANG(?descricao) = "" || LANGMATCHES(LANG(?descricao), "pt"))

            OPTIONAL {{
                {{ ?regra ?p ?chunkNode . }} UNION {{ ?chunkNode ?p ?regra . }}
                ?chunkNode a :Chunk ;
                    :texto ?textoBruto .
                FILTER(LANG(?textoBruto) = "" || LANGMATCHES(LANG(?textoBruto), "pt"))
                BIND(?textoBruto AS ?texto)
                OPTIONAL {{ ?chunkNode :idChunk ?id }}
                BIND(?id AS ?idChunk)
            }}

            BIND(?regra AS ?regraURI)
            }}
            ORDER BY DESC(?score)
            LIMIT {self.retrieval_size}
        '''
        
        url = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json"
        }
        
        try:
            response = requests.post(
                url,
                data=sparql_query,
                headers=headers,
                auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD),
                timeout=30
            )
            response.raise_for_status()
        except Exception as e:
            print(f"[GraphRulesRetriever] Erro na query SPARQL: {e}")
            return []
        
        documents = []
        
        if response.status_code == 200:
            results = response.json()
            bindings = results.get("results", {}).get("bindings", [])
            
            for item in bindings:
                score = float(item.get("score", {}).get("value", 0))
                id_chunk = item.get("idChunk", {}).get("value", "")
                regra_uri = item.get("regraURI", {}).get("value", "")
                descricao = normalize(item.get("descricao", {}).get("value", ""))
                
                doc = Document(
                    page_content=descricao,
                    metadata={
                        "id": regra_uri,
                        "id_chunk": id_chunk,
                        "score": score,
                        "source": "regra",
                        "type": "rule"
                    }
                )
                documents.append(doc)
        
        return documents

class GraphChunksRetriever(BaseRetriever):
    """Retriever para buscar CHUNKS no grafo"""
    
    named_graph: str
    retrieval_size: int = 10
    
    class Config:
        """Configuração do Pydantic"""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str,
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Busca chunks de texto no grafo"""
        
        keyword = re.sub(r'[\\/]+', ' ', query)
        
        sparql_query = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?idChunk ?score ?texto ?descricao ?documento ?tipo
            FROM <https://omc.co/graph/{self.named_graph}>
            WHERE {{  
            {{
                SELECT ?chunk (MAX(?s) AS ?score) (SAMPLE(?id) AS ?idChunk) (SAMPLE(?t) AS ?texto) (SAMPLE(?descRegra) AS ?descricao)
                WHERE {{
                    ?q a luc-index:omc_full ;
                    luc:query '{keyword}' ;
                    luc:entities ?chunk .
                    ?chunk luc:score ?s .
                    OPTIONAL {{ ?chunk :idChunk ?id }}
                    OPTIONAL {{ ?chunk :texto ?t . FILTER(LANG(?t) = "" || LANGMATCHES(LANG(?t), "pt")) }}
                    OPTIONAL {{
                        ?chunk a/rdfs:subClassOf* :Regra ;
                        :descricao ?descRegra .
                    }}
                }}
                GROUP BY ?chunk
                ORDER BY DESC(?score)
                LIMIT 500
            }}

            OPTIONAL {{
            {{ 
                ?chunk :estaContidoEm ?documento .
                OPTIONAL {{ ?documento a ?tipo }}
            }}UNION{{
                BIND(?chunk AS ?documento) .
                OPTIONAL {{ ?documento a ?tipo }}
            }}
            }}

            FILTER EXISTS {{
                VALUES ?tipoPermitido {{ :Chunk }}
                ?chunk a ?tipoPermitido .
            }}
        }}
        ORDER BY DESC(?score)
        LIMIT {self.retrieval_size}
        '''
        
        url = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json"
        }
        
        try:
            response = requests.post(
                url,
                data=sparql_query,
                headers=headers,
                auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD),
                timeout=30
            )
            response.raise_for_status()
        except Exception as e:
            print(f"[GraphChunksRetriever] Erro na query SPARQL: {e}")
            return []
        
        documents = []
        
        if response.status_code == 200:
            results = response.json()
            bindings = results.get("results", {}).get("bindings", [])
            
            for item in bindings:
                score = float(item.get("score", {}).get("value", 0))
                id_chunk = item.get("idChunk", {}).get("value", "")
                texto = normalize(item.get("texto", {}).get("value", ""))
                
                doc = Document(
                    page_content=texto,
                    metadata={
                        "id": id_chunk,
                        "score": score,
                        "source": "chunk",
                        "type": "text"
                    }
                )
                documents.append(doc)
        
        return documents

class NormalizedScoreFusion:
    """
    Combina múltiplos retrievers normalizando scores e aplicando pesos.
    
    Vantagens sobre EnsembleRetriever:
    - Usa scores reais do Lucene (não apenas posição)
    - Normalização min-max para comparar scores de diferentes escalas
    - Pesos configuráveis e transparentes
    - Deduplicação por ID
    - Zero latência adicional
    """
    
    def __init__(self, retrievers: List[BaseRetriever], weights: Optional[List[float]] = None):
        """
        Args:
            retrievers: Lista de retrievers a combinar
            weights: Lista de pesos (mesmo tamanho que retrievers)
                    Ex: [1.5, 1.0] = primeiro retriever vale 50% mais
        """
        self.retrievers = retrievers
        
        if weights is None:
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Número de pesos deve ser igual ao número de retrievers")
            self.weights = weights
    
    def _normalize_scores(self, docs: List[Document]) -> List[Document]:
        """
        Normalização Min-Max: transforma scores em range [0, 1]
        
        Formula: normalized = (score - min) / (max - min)
        """
        if not docs:
            return docs
        
        scores = [d.metadata.get('score', 0) for d in docs]
        min_score = min(scores)
        max_score = max(scores)
        
        # Se todos os scores são iguais, atribui 1.0 para todos
        if max_score == min_score:
            for doc in docs:
                doc.metadata['normalized_score'] = 1.0
        else:
            for doc in docs:
                original_score = doc.metadata.get('score', 0)
                normalized = (original_score - min_score) / (max_score - min_score)
                doc.metadata['normalized_score'] = normalized
        
        return docs
    
    def get_relevant_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Busca e combina documentos de todos os retrievers.
        
        Processo:
        1. Busca em cada retriever
        2. Normaliza scores de cada conjunto (0-1)
        3. Aplica pesos configurados
        4. Combina scores (usa máximo para duplicatas)
        5. Ordena por score final
        
        Args:
            query: Query de busca
            top_k: Número de documentos a retornar
            
        Returns:
            Lista de documentos ordenados por relevância
        """
        all_docs: Dict[str, Document] = {}  # {id: doc}
        
        print(f'\n[NormalizedScoreFusion] Iniciando busca para: "{query}"')
        print(f'[NormalizedScoreFusion] Retrievers: {len(self.retrievers)}, Pesos: {self.weights}')
        
        for idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            retriever_name = retriever.__class__.__name__
            print(f'\n[{retriever_name}] Buscando... (peso={weight})')
            
            # Busca documentos
            docs = retriever._get_relevant_documents(query)
            print(f'[{retriever_name}] Encontrados: {len(docs)} documentos')
            
            if not docs:
                continue
            
            # Mostra range de scores originais
            original_scores = [d.metadata['score'] for d in docs]
            print(f'[{retriever_name}] Scores originais: [{min(original_scores):.4f} - {max(original_scores):.4f}]')
            
            # Normaliza scores (0-1)
            docs = self._normalize_scores(docs)
            
            # Mostra alguns exemplos após normalização
            print(f'[{retriever_name}] Exemplos após normalização:')
            for doc in docs[:3]:
                print(f'  - ID: {doc.metadata["id"][:50]}... | '
                      f'Original: {doc.metadata["score"]:.4f} | '
                      f'Normalizado: {doc.metadata["normalized_score"]:.4f} | '
                      f'Ponderado: {doc.metadata["normalized_score"] * weight:.4f}')
            
            # Combina documentos
            for doc in docs:
                doc_id = doc.metadata.get('id')
                if not doc_id:
                    continue
                
                normalized_score = doc.metadata['normalized_score']
                weighted_score = normalized_score * weight
                
                # Armazena peso aplicado para debug
                doc.metadata['weight_applied'] = weight
                doc.metadata['retriever_source'] = retriever_name
                
                if doc_id in all_docs:
                    # Documento já existe: usa o MAIOR score (estratégia MAX)
                    old_score = all_docs[doc_id].metadata['combined_score']
                    new_score = max(old_score, weighted_score)
                    
                    if new_score > old_score:
                        # Atualiza com o novo documento (que tem score maior)
                        doc.metadata['combined_score'] = new_score
                        all_docs[doc_id] = doc
                        print(f'  [DEDUP] ID duplicado, mantendo maior score: {new_score:.4f}')
                    else:
                        print(f'  [DEDUP] ID duplicado, mantendo score anterior: {old_score:.4f}')
                else:
                    # Novo documento
                    doc.metadata['combined_score'] = weighted_score
                    all_docs[doc_id] = doc
        
        # Ordena por score combinado (decrescente)
        result = sorted(
            all_docs.values(),
            key=lambda d: d.metadata['combined_score'],
            reverse=True
        )
        
        print(f'\n[NormalizedScoreFusion] Resultado final:')
        print(f'  - Total único de documentos: {len(result)}')
        print(f'  - Retornando top {top_k}')
        print(f'\n  Top 5 documentos:')
        for i, doc in enumerate(result[:5], 1):
            print(f'    {i}. Score: {doc.metadata["combined_score"]:.4f} | '
                  f'Tipo: {doc.metadata["type"]} | '
                  f'Fonte: {doc.metadata["retriever_source"]} | '
                  f'ID: {doc.metadata["id"][:60]}...')
        
        return result[:top_k]

def graph_search_ensemble(keyword, question, named_graph, retrieval_size):
    
    """
    Busca no grafo usando NormalizedScoreFusion.
    
    Mantém compatibilidade com a interface original.
    """
    print('\n' + '='*80)
    print('--> GRAPH SEARCH (NormalizedScoreFusion mode)')
    print('='*80)

    try:
        # Cria retrievers
        rules_retriever = GraphRulesRetriever(
            named_graph=named_graph,
            retrieval_size=retrieval_size
        )
        
        chunks_retriever = GraphChunksRetriever(
            named_graph=named_graph,
            retrieval_size=retrieval_size
        )
        
        # Cria fusion com pesos otimizados
        # Regras têm 50% mais peso que chunks (ajuste conforme necessário)
        fusion = NormalizedScoreFusion(
            retrievers=[rules_retriever, chunks_retriever],
            weights=[1.5, 1.0]  # Você pode ajustar esses valores!
        )
        
        # Busca documentos
        documents = fusion.get_relevant_documents(keyword, top_k=retrieval_size)
        
        # Formata resposta no formato original
        resp_rules_toon  = 'id_chunk;texto_regra\n'
        resp_chunks_toon = 'id_chunk;score;texto_chunk\n'
        knowledge_base = {}
        
        for doc in documents:
            metadata = doc.metadata
            
            if metadata.get('type') == 'rule':
                # É uma regra
                resp_rules_toon += f'{metadata.get("id_chunk", "")};{doc.page_content}\n'
                knowledge_base[metadata['id']] = doc.page_content
            
            elif metadata.get('type') == 'text':
                # É um chunk
                final_score = metadata.get('combined_score', metadata.get('score', 0))
                resp_chunks_toon += f'{metadata["id"]};{final_score:.4f};{doc.page_content}\n'
                knowledge_base[metadata['id']] = doc.page_content
        
        resp_final = f'''Regras:
        {textwrap.dedent(resp_rules_toon)}

        Chunks:
        {textwrap.dedent(resp_chunks_toon)}
        '''
        
        print('\n' + '='*80)
        print(f'SUCESSO: {len(knowledge_base)} documentos únicos recuperados')
        print('='*80 + '\n')
        
        return {
            'status': 'OK',
            'response': resp_final,
            'dataset': knowledge_base
        }

    except Exception as e:
        error_msg = f"Graph ERROR: {str(e)}"
        print(f"\n[ERROR] {error_msg}\n")
        
        return {
            'status': 'ERROR',
            'response': error_msg,
            'dataset': {}
        }
