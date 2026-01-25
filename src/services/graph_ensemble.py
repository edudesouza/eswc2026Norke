
import requests, re, textwrap

from langchain_classic.schema               import BaseRetriever, Document
from langchain_classic.retrievers           import EnsembleRetriever
from langchain_classic.callbacks.manager    import CallbackManagerForRetrieverRun

from typing         import List
from requests.auth  import HTTPBasicAuth

from src.config     import settings
from src.utils.text import normalize

# original

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
        
        response = requests.post(
            url,
            data=sparql_query,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD)
        )
        
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
        
        response = requests.post(
            url,
            data=sparql_query,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD)
        )
        
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

def graph_search_ensemble(keyword, question, named_graph, retrieval_size):

    print('--> search graph (ensemble mode)')

    try:

        rules_retriever = GraphRulesRetriever(
            named_graph=named_graph,
            retrieval_size=retrieval_size
        )
        
        chunks_retriever = GraphChunksRetriever(
            named_graph=named_graph,
            retrieval_size=retrieval_size
        )
        
        # Criar ensemble com pesos
        ensemble = EnsembleRetriever(
            retrievers=[rules_retriever, chunks_retriever],
            weights=[0.6, 0.4]  # Priorizar regras (60%) sobre chunks (40%)
        )

        try:
            documents = ensemble.invoke(keyword)  # Versão mais nova
        except AttributeError:
            documents = ensemble._get_relevant_documents(keyword)  # Versão antiga
        
        # Formatar resposta no formato original
        resp_rules_toon  = 'id_chunk;texto_regra\n'
        resp_chunks_toon = 'id_chunk;score;texto_chunk\n'
        knowledge_base   = {}
        
        for doc in documents:
            
            metadata = doc.metadata
            
            if metadata.get('type') == 'rule':
                
                # É uma regra
                resp_rules_toon += f'{metadata.get("id_chunk", "")};{doc.page_content}\n'
                knowledge_base[metadata['id']] = doc.page_content
            
            elif metadata.get('type') == 'text':
                
                # É um chunk
                resp_chunks_toon += f'{metadata["id"]};{metadata["score"]};{doc.page_content}\n'
                knowledge_base[metadata['id']] = doc.page_content
        
        resp_final = f'''Regras:
        {textwrap.dedent(resp_rules_toon)}
        Chunks:
        {textwrap.dedent(resp_chunks_toon)}
        '''
        
        return {
            'status': 'OK',
            'response': resp_final,
            'dataset': knowledge_base
        }

    except Exception as e:
        error_msg = f"Graph ERROR: {str(e)}"
        print(f"--> {error_msg}")
        
        return {
            'status': 'ERROR',
            'response': error_msg,
            'dataset': {}
        }
