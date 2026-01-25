"""
vector_graph_pipeline.py

Pipeline completo e compatível com SAF-Eval (simple retriever):
- Recupera REGRAS no GraphDB (Lucene connector)
- Recupera EVIDÊNCIAS (chunks) vinculadas às regras no GraphDB
- Recupera CONTEXTO adicional no Elasticsearch (híbrido BM25 + KNN) e funde por RRF
- Retorna:
    - dataset_structured (rico: rules/evidences/elastic em listas de dicts)
    - dataset            (flat: dict[str,str]) -> compatível com saf-eval simple provider

Requisitos:
- settings.GRAPHDB_BASE_URL, settings.repositorio, settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD
- settings.elastic_client (cliente ES)
- settings.OPENAI_API_KEY
- normalize() em src.utils.text
"""

import re
import textwrap
from typing import List, Dict, Any, Tuple

import requests
from requests.auth import HTTPBasicAuth

from langchain_classic.schema import BaseRetriever, Document
from langchain_classic.callbacks.manager import CallbackManagerForRetrieverRun

from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.utils.text import normalize


# =========================
# SAF compatibility helper
# =========================

def dataset_to_flat_text(dataset: Dict[str, Any]) -> Dict[str, str]:
    """
    Converte dataset estruturado (rules/evidences/elastic) em dict[str,str]
    compatível com saf-eval (simple retriever).
    """
    if not dataset:
        return {}

    # Se já veio flat, mantém
    if isinstance(dataset, dict) and dataset and all(isinstance(v, str) for v in dataset.values()):
        return dataset

    flat: Dict[str, str] = {}

    # Estruturado (padrão deste pipeline)
    for r in dataset.get("rules", []) or []:
        uri = (r.get("regra_uri") or "unknown").strip()
        txt = (r.get("descricao") or "").strip()
        if txt:
            flat[f"rule::{uri}"] = txt

    for e in dataset.get("evidences", []) or []:
        cid = (e.get("id_chunk") or "unknown").strip()
        uri = (e.get("regra_uri") or "").strip()
        txt = (e.get("texto") or "").strip()
        if txt:
            flat[f"evidence::{cid}::{uri}"] = txt

    for d in dataset.get("elastic", []) or []:
        _id = (d.get("id") or "unknown").strip()
        txt = (d.get("texto") or "").strip()
        if txt:
            flat[f"elastic::{_id}"] = txt

    return flat

# =========================
# Utilities
# =========================

def _escape_lucene_query(q: str, max_len: int = 300) -> str:
    """
    Sanitiza string para uso em luc:query (GraphDB Lucene connector).
    Evita quebrar string SPARQL e reduz drift por operadores.
    """
    if not q:
        return ""
    q = q.strip()[:max_len]
    q = re.sub(r"[\\/]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # não deixar quebrar string SPARQL
    q = q.replace("'", " ").replace('"', " ")

    # remove caracteres que costumam atuar como operadores / causar ruído
    q = re.sub(r"[\(\)\{\}\[\]\^\~\:\!\+\-\=\<\>\|]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _post_sparql(repo_url: str, sparql_query: str, timeout: Tuple[float, float] = (3.05, 30)) -> Dict[str, Any]:
    
    headers = {
        "Content-Type": "application/sparql-query",
        "Accept": "application/sparql-results+json",
    }
    resp = requests.post(
        repo_url,
        data=sparql_query,
        headers=headers,
        auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD),
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"GraphDB HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()

def _rrf_fusion(
    ranked_lists: List[List[str]],
    weights: List[float],
    k: int = 60,
    topn: int = 20
) -> List[str]:
    """
    Weighted Reciprocal Rank Fusion (RRF) sobre listas de IDs.
    score(d) = sum_i w_i * 1/(k + rank_i(d))
    """
    scores: Dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(lst, start=1):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + (w * (1.0 / (k + rank)))

    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]]


# =========================
# GraphDB Retrievers
# =========================

class GraphRulesRetriever(BaseRetriever):
    """
    Recupera REGRAS (:Regra) via Lucene connector.
    Retorna Document.page_content = descricao (norma).
    """
    named_graph: str
    retrieval_size: int = 20

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        keyword = _escape_lucene_query(query)
        if not keyword:
            return []

        sparql_query = f"""
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
        PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?score ?regraURI ?descricao ?idChunk
        FROM <https://omc.co/graph/{self.named_graph}>
        WHERE {{
        {{
            SELECT ?regra (MAX(?s) AS ?score)
            WHERE {{
            ?q a luc-index:omc_full ;
                luc:query "{keyword}" ;
                luc:entities ?regra .
            ?regra luc:score ?s .
            }}
            GROUP BY ?regra
            ORDER BY DESC(?score)
            LIMIT 500
        }}

        ?regra a/rdfs:subClassOf* :Regra .
        ?regra :descricao ?descricao .
        FILTER(LANG(?descricao) = "" || LANGMATCHES(LANG(?descricao), "pt"))

        OPTIONAL {{
            {{ ?regra ?p ?chunkNode . }} UNION {{ ?chunkNode ?p ?regra . }}
            ?chunkNode a :Chunk ;
                    :idChunk ?idChunk .
        }}

        BIND(?regra AS ?regraURI)
        }}
        ORDER BY DESC(?score)
        LIMIT {self.retrieval_size}
        """

        #print( sparql_query )

        repo_url = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        results = _post_sparql(repo_url, sparql_query)

        documents: List[Document] = []
        for item in results.get("results", {}).get("bindings", []) or []:
            score = float(item.get("score", {}).get("value", 0.0))
            regra_uri = item.get("regraURI", {}).get("value", "")
            descricao = normalize(item.get("descricao", {}).get("value", ""))
            id_chunk = item.get("idChunk", {}).get("value", "")

            if not regra_uri or not descricao:
                continue

            documents.append(
                Document(
                    page_content=descricao,
                    metadata={
                        "id": regra_uri,
                        "id_chunk": id_chunk,
                        "score": score,
                        "source": "graph",
                        "type": "rule",
                    },
                )
            )
        return documents

class GraphEvidenceByRuleRetriever(BaseRetriever):
    """
    Dado um conjunto de regras (URIs), recupera chunks evidência vinculados.
    Retorna Document.page_content = texto do chunk.
    """
    named_graph: str
    rules_uris: List[str]
    retrieval_size: int = 30

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,  # mantido por compat, não usado
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        if not self.rules_uris:
            return []

        values = " ".join([f"<{u}>" for u in self.rules_uris if isinstance(u, str) and u.startswith("http")])
        if not values:
            return []

        sparql_query = f"""
        PREFIX :     <https://omc.co/vocabulary/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?regraURI ?idChunk ?texto
        FROM <https://omc.co/graph/{self.named_graph}>
        WHERE {{
        VALUES ?regra {{ {values} }}
        ?regra a/rdfs:subClassOf* :Regra .

        {{ ?regra ?p ?chunk . }} UNION {{ ?chunk ?p ?regra . }}
        ?chunk a :Chunk ;
                :idChunk ?idChunk ;
                :texto ?texto .
        FILTER(LANG(?texto) = "" || LANGMATCHES(LANG(?texto), "pt"))

        BIND(?regra AS ?regraURI)
        }}
        LIMIT {self.retrieval_size}
        """

        #print( sparql_query )

        repo_url = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        results = _post_sparql(repo_url, sparql_query)

        documents: List[Document] = []
        for item in results.get("results", {}).get("bindings", []) or []:
            regra_uri = item.get("regraURI", {}).get("value", "")
            id_chunk = item.get("idChunk", {}).get("value", "")
            texto = normalize(item.get("texto", {}).get("value", ""))

            if not id_chunk or not texto:
                continue

            documents.append(
                Document(
                    page_content=texto,
                    metadata={
                        "id": id_chunk,
                        "rule_uri": regra_uri,
                        "source": "graph",
                        "type": "evidence",
                    },
                )
            )
        return documents

# =========================
# Elasticsearch Retriever (BM25 + KNN + RRF)
# =========================

class ElasticsearchHybridRetriever(BaseRetriever):
    """
    Faz duas buscas no Elasticsearch:
      1) BM25 (multi_match)
      2) KNN (embedding)
    e funde os IDs por RRF (mais estável do que misturar scores).
    """
    index_name: str
    user_id: str
    retrieval_size: int = 10
    knn_k: int = 50
    knn_candidates: int = 500

    class Config:
        arbitrary_types_allowed = True

    def _search_bm25(self, query: str) -> List[Dict[str, Any]]:
        q = {
            "size": max(self.retrieval_size, 25),
            "_source": ["file_url", "id_usuario", "id_externo", "texto_rico"],
            "query": {
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query": query,
                            "fields": ["texto_rico"],
                        }
                    }],
                    "filter": [{"term": {"id_usuario": self.user_id}}],
                }
            },
        }
        resp = settings.elastic_client.search(index=self.index_name, body=q)
        return resp.get("hits", {}).get("hits", []) or []

    def _search_knn(self, query_vector: List[float]) -> List[Dict[str, Any]]:
        # Ajuste este body se seu cluster exigir outra sintaxe de KNN.
        q = {
            "size": max(self.retrieval_size, 25),
            "_source": ["file_url", "id_usuario", "id_externo", "texto_rico"],
            "knn": {
                "field": "embedding_rico",
                "query_vector": query_vector,
                "k": self.knn_k,
                "num_candidates": self.knn_candidates,
                "filter": {"term": {"id_usuario": self.user_id}},
            },
        }
        resp = settings.elastic_client.search(index=self.index_name, body=q)
        return resp.get("hits", {}).get("hits", []) or []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        if not query or not query.strip():
            return []

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)
        query_vector = embeddings.embed_query(query)

        bm25_hits = self._search_bm25(query)
        knn_hits  = self._search_knn(query_vector)

        bm25_ids = [h.get("_id") for h in bm25_hits if h.get("_id")]
        knn_ids = [h.get("_id") for h in knn_hits if h.get("_id")]

        fused_ids = _rrf_fusion(
            ranked_lists=[bm25_ids, knn_ids],
            weights=[0.5, 0.5],
            k=60,
            topn=self.retrieval_size,
        )

        hits_by_id: Dict[str, Dict[str, Any]] = {}
        for h in bm25_hits + knn_hits:
            _id = h.get("_id")
            if _id and _id not in hits_by_id:
                hits_by_id[_id] = h

        documents: List[Document] = []
        for _id in fused_ids:
            h = hits_by_id.get(_id, {}) or {}
            src = h.get("_source", {}) or {}

            texto_rico = normalize(src.get("texto_rico", ""))
            if not texto_rico:
                continue

            documents.append(
                Document(
                    page_content=texto_rico,
                    metadata={
                        "id": _id,
                        "score": h.get("_score", 0.0),
                        "source": "elastic",
                        "type": "text",
                        "file_url": src.get("file_url", ""),
                        "id_externo": src.get("id_externo", ""),
                    },
                )
            )
        return documents

# =========================
# Main pipeline (Graph rules -> Graph evidences + Elastic context)
# =========================

def vector_graph_ensemble(
    palavras_chave: str,
    pergunta: str,
    index_name: str,
    user_id: str,
    named_graph: str,
    top_rules: int = 10,
    top_evidences: int = 15,
    top_elastic: int = 10,
) -> Dict[str, Any]:
    """
    Retorno:
      {
        status: "OK"|"ERROR",
        response: str,
        dataset_structured: dict,
        dataset: dict[str,str]   # FLAT para SAF-Eval
      }
    """
    try:
        q = (palavras_chave or pergunta or "").strip()
        if not q:
            return {"status": "ERROR", "response": "Query vazia.", "dataset_structured": {}, "dataset": {}}

        # 1) Regras no grafo
        rules_retriever = GraphRulesRetriever(named_graph=named_graph, retrieval_size=top_rules)
        rules_docs = rules_retriever._get_relevant_documents(q)
        rules_uris = [d.metadata.get("id") for d in rules_docs if d.metadata.get("id")]
        rules_uris = list(dict.fromkeys(rules_uris))  # dedup mantendo ordem

        # 2) Evidências vinculadas às regras
        evidence_docs: List[Document] = []
        if rules_uris:
            evidence_retriever = GraphEvidenceByRuleRetriever(
                named_graph=named_graph,
                rules_uris=rules_uris,
                retrieval_size=top_evidences,
            )
            evidence_docs = evidence_retriever._get_relevant_documents(q)

        # Dedup evidências por idChunk
        seen = set()
        evidence_docs_dedup: List[Document] = []
        for d in evidence_docs:
            cid = d.metadata.get("id")
            if cid and cid not in seen:
                seen.add(cid)
                evidence_docs_dedup.append(d)

        # 3) Elastic como contexto adicional/fallback
        elastic_retriever = ElasticsearchHybridRetriever(index_name=index_name, user_id=user_id, retrieval_size=top_elastic)
        elastic_docs = elastic_retriever._get_relevant_documents(q)

        # =============
        # Output textual
        # =============
        resp_rules = "id_chunk;score;regra_uri;texto_regra\n"
        for d in rules_docs:
            resp_rules += (
                f'{d.metadata.get("id_chunk","")};'
                f'{d.metadata.get("score",0)};'
                f'{d.metadata.get("id","")};'
                f'{d.page_content.replace(";",",")}\n'
            )

        resp_evid = "id_chunk;regra_uri;texto_chunk\n"
        for d in evidence_docs_dedup:
            resp_evid += (
                f'{d.metadata.get("id","")};'
                f'{d.metadata.get("rule_uri","")};'
                f'{d.page_content.replace(";",",")}\n'
            )

        resp_elastic = "id_doc;score;id_externo;file_url;texto\n"
        for d in elastic_docs:
            md = d.metadata
            resp_elastic += (
                f'{md.get("id","")};'
                f'{md.get("score",0)};'
                f'{md.get("id_externo","")};'
                f'{md.get("file_url","")};'
                f'{d.page_content.replace(";",",")}\n'
            )

        response = f"""Regras (Grafo):
        {textwrap.dedent(resp_rules)}

        Evidências vinculadas às regras (Grafo):
        {textwrap.dedent(resp_evid)}

        Contexto adicional (Elastic - BM25+KNN via RRF):
        {textwrap.dedent(resp_elastic)}
        """

        # =============
        # Structured dataset
        # =============
        dataset_structured: Dict[str, Any] = {
            "rules": [
                {
                    "regra_uri": d.metadata.get("id"),
                    "descricao": d.page_content,
                    "score": d.metadata.get("score", 0.0),
                    "id_chunk": d.metadata.get("id_chunk", ""),
                }
                for d in rules_docs
            ],
            "evidences": [
                {
                    "id_chunk": d.metadata.get("id"),
                    "regra_uri": d.metadata.get("rule_uri"),
                    "texto": d.page_content,
                }
                for d in evidence_docs_dedup
            ],
            "elastic": [
                {
                    "id": d.metadata.get("id"),
                    "texto": d.page_content,
                    "file_url": d.metadata.get("file_url"),
                    "id_externo": d.metadata.get("id_externo"),
                }
                for d in elastic_docs
            ],
        }

        # =============
        # Flat dataset (SAF-Eval expects dict[str,str])
        # =============
        dataset_flat = dataset_to_flat_text(dataset_structured)

        return {
            "status": "OK",
            "response": response,
            "dataset_structured": dataset_structured,
            "dataset": dataset_flat,  # <-- PASSE ESTE PARA O SAF
        }

    except Exception as e:
        return {"status": "ERROR", "response": f"Retrieval ERROR: {str(e)}", "dataset_structured": {}, "dataset": {}}

def vector_graph_ensemble_manual():
    pass