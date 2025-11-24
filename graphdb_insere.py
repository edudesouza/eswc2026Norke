import os, re, unicodedata, warnings, time
from typing import List, Dict, Any
from pprint import pprint
from urllib.parse import quote

from elasticsearch  import Elasticsearch, helpers

# RDF / SPARQL
from rdflib             import Graph, URIRef, Literal, Namespace, ConjunctiveGraph
from rdflib.namespace   import RDF, RDFS, XSD, OWL

# HTTP
import requests
from requests.auth import HTTPBasicAuth

# LLM / Extração
from langchain_openai           import ChatOpenAI
from langchain_google_genai     import ChatGoogleGenerativeAI

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts     import ChatPromptTemplate
from langchain_core.documents   import Document

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

'''
-> PRODUÇÃO

- criar named graph, com usuario
- criar documento

Ingestão de documentos → extração de grafo com LLM → geração de RDF (Turtle) → upload no GraphDB.

- Classes (tipos): PascalCase
- Predicados (propriedades/relações): camelCase
- IRIs de instâncias preservam underscores/ids (ex.: doc_978_2, vaga_101)

Pré-requisitos:
- Variáveis de ambiente:
  GRAPHDB_BASE_URL, GRAPHDB_USERNAME, GRAPHDB_PASSWORD (opcional)
  OMC_BASE_NS (ex.: https://omc.co/vocabulary/)
  OMC_LLM_MODEL (ex.: o4-mini)
- No vocabulário .ttl:
  - :Documento (classe)
  - :identificador (DatatypeProperty xsd:string)
  - :conteudo (DatatypeProperty xsd:string)
  - :extraidoDe (ObjectProperty Resource -> :Documento)   # recomendado
    OU use :refereSeA (Documento -> Resource) e ajuste P_EXTRACTED_FROM abaixo
'''

# =========================
# Config / Constantes
# =========================

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST,
    basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)

GRAPHDB_BASE_URL   = os.getenv("GRAPHDB_BASE_URL_PROD")
GRAPHDB_USERNAME   = os.getenv("GRAPHDB_USERNAME") 
GRAPHDB_PASSWORD   = os.getenv("GRAPHDB_PASSWORD_PROD")
GRAPHDB_REPOSITORY = os.getenv("GRAPHDB_REPOSITORY", "omc_v1")
GRAPHDB_REPO_URL   = f"{GRAPHDB_BASE_URL}/repositories/{GRAPHDB_REPOSITORY}/statements"

BASE_NS            = os.getenv("OMC_BASE_NS", "https://omc.co/vocabulary/").strip()

VOCAB_NS = "https://omc.co/vocabulary/"
ROOT_NS  = "https://omc.co/"
GR_NS    = "https://omc.co/graph/usuario/"

DOC_CLASS    = "Documento"
CHUNK_CLASS  = "Chunk"
USER_CLASS   = "Usuario"

P_DOC_ID          = "identificador"
P_DOC_FILE        = "arquivo"
P_USER_ID         = "idUsuario"
P_CHUNK_ID        = "idChunk"
P_CHUNK_TEXT      = "texto"
P_CONTAINS        = "estaContidoEm"     # Chunk -> Documento
P_HAS_DOCUMENT    = "possuiDocumento"   # Usuario -> Documento
P_RESPONSAVEL_POR = "responsavelPor"    # Usuario -> Documento
P_EXTRACTED_FROM  = "extraidode"        # Entidade -> Chunk
P_DESCRICAO       = "descricao"

LLM_MODEL        = os.getenv("OMC_LLM_MODEL", "o4-mini")
#LLM_MODEL        = os.getenv("OMC_LLM_MODEL","gpt-4.1-mini")
#LLM_MODEL        = os.getenv("OMC_LLM_MODEL","gpt-5")

# normalização
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def pascal_case(s: str) -> str:
    
    """PascalCase para Classes/Tipos."""
    s = _strip_accents(str(s))
    s = re.sub(r'[^0-9A-Za-z]+', ' ', s)
    return ''.join(p.capitalize() for p in s.split() if p) or 'Recurso'

def camel_case(s: str) -> str:

    """camelCase para propriedades/predicados."""
    s = _strip_accents(str(s))
    s = s.replace('-', '_')
    parts = re.split(r'[^0-9A-Za-z]+', s)
    parts = [p for p in parts if p]
    if not parts:
        return 'prop'
    head, tail = parts[0].lower(), [p.capitalize() for p in parts[1:]]
    return head + ''.join(tail)

def safe_localname(s: str) -> str:

    """Localname para instâncias (preserva underscores/ids úteis)."""
    s = _strip_accents(str(s))
    s = re.sub(r'[^0-9A-Za-z_]+', '_', s).strip('_')
    return s or 'instancia'

def iri_class(ns: Namespace, name: str) -> URIRef:
    return URIRef(ns[pascal_case(name)])

def iri_prop(ns: Namespace, name: str) -> URIRef:
    return URIRef(ns[camel_case(name)])

def iri_instance(ns: Namespace, name: str) -> URIRef:
    return URIRef(ns[safe_localname(name)])

#fim

CANON_TYPES = {
    "EntidadeFisica","CultoReligioso","Pessoa","Papel","Documento","Regra","Ocorrencia","Veiculo","InstituicaoExterna",
    "Condominio","Edificacao","Unidade","AreaComum","Garagem","VagaDeEstacionamento","Piscina",
    "SalaoDeFestas","Playground","Academia","SistemaControleAcesso",
    "Sindico","SubSindico","Conselheiro","Morador","Inquilino","Visitante","Funcionario",
    "Zelador","Porteiro","Faxineiro","Vigia","Jardineiro","PrestadorDeServico",
    "Administradora","Contador","Advogado","Seguradora","Banco","Fornecedor",
    "Convencao","RegimentoInterno","AtaDeAssembleia","Orcamento","Balancete",
    "ContratoPrestacaoServicos","Notificacao","Multa",
    "Assembleia","Manutencao","Obra","ReservaAreaComum","Seguranca","Correspondencia","Usuario","Chunk"
}

CANON_RELATIONSHIPS = {
    "parteDe","localizadaEm","possuiUnidade","possuiAreaComum","desempenhaPapel","papelEm",
    "aplicaA","permitidoEm","proibidoEm","registradoEm","ocorreuEm","executadoPor","temResponsavel",
    "contrata","temContaEm","envolveVeiculo","reservadoPor","refereSeA",
    "proibidoSe","responsabilizadoPor","podeUsar","tipoDe","inclui","citadoEm","usa","ocupa","regidaPor",
    "estaContidoEm","possuiDocumento","responsavelPor","nome","identificador","cpf","cnpj","data","valor","texto",
    "descricao","conteudo","arquivo","idChunk","idUsuario","idExterno"
}

ALLOWED_NODE_PROPS = {"descricao"}  # mantenha enxuto

#--------------------------------------------------------------

REL_ALIASES = {
    # CAPS/_ → camelCase
    "PERMITIDO_EM": "permitidoEm",
    "APLICA_A": "aplicaA",
    "PARTE_DE": "parteDe",
    "PROIBIDO_SE": "proibidoSe",
    "REFERE_SE_A": "refereSeA",
    "RESPONSABILIZADO_POR": "responsabilizadoPor",
    "PODE_USAR": "podeUsar",
    "TIPO_DE": "tipoDe",
    "INCLUI": "inclui",
    "CITADO_EM": "citadoEm",
    "USA": "usa",
    "OCUPA": "ocupa",
    "REGIDA_POR": "regidaPor",
    "ESTA_CONTIDO_EM":"estaContidoEm",
    "PARTE_DE":"parteDe"
}

REL_MAP = {
    "REGISTRADOEM":   "registradoEm",
    "REGIDAPOR":      "regidaPor",
    "CONTRATA":       "contrata",
    "TEMRESPONSAVEL": "temResponsavel",
    "REFERESEA":      "refereSeA",
}

def canon_rel(r: str) -> str:
    r = REL_ALIASES.get(r, r)
    return camel_case(r)

def build_graph_transformer() -> LLMGraphTransformer:
    
    allowed_rels = sorted(list(CANON_RELATIONSHIPS))

    system_txt = (
        "Você é um extrator de grafos jurídicos em pt-BR. "
        "Respeite: TIPOS (PascalCase), RELAÇÕES (camelCase).\n"
        "Tipos permitidos: " + ", ".join(sorted(CANON_TYPES)) + "\n"
        "Relações permitidas: " + ", ".join(allowed_rels) + "\n"
        "Propriedades literais de nó permitidas: " + ", ".join(sorted(ALLOWED_NODE_PROPS)) + "\n"
        "Use nomes amigáveis, exemplo: Regra_crianca_garagem\n"
        "Regras são normas ou determinação aplicáveis no condomínio\n"
        "Regras:\n"
        "1) Cardinalidade de vagas/veículos: crie Regra e ligue com :aplicaA → (VagaDeEstacionamento|Garagem|Condominio).\n"
        "2) Proibições: (Veiculo)-[:proibidoEm|:proibidoSe]->(EntidadeFisica ou Limite textual).\n"
        "3) Permissões: (Veiculo)-[:permitidoEm]->(VagaDeEstacionamento|Garagem).\n"
        "4) Penalidades: (Morador)-[:responsabilizadoPor {{motivo}}]->(Multa|Notificacao).\n"
        "Saída fiel e sucinta, compatível para MERGE entre chunks."
    )

    pt_template = ChatPromptTemplate.from_messages([
        ("system", system_txt),
        ("human", "Texto:\n{input}\n")
    ])

    gpt = ChatOpenAI(model=LLM_MODEL)

    google = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )
    
    return LLMGraphTransformer(
        llm=gpt,
        prompt=pt_template,
        allowed_nodes=list(CANON_TYPES),
        allowed_relationships=allowed_rels,
        node_properties=list(ALLOWED_NODE_PROPS),
    )

def extract_graph_docs(text, meta):
    
    transformer = build_graph_transformer()
    docs = [Document(page_content=text, metadata=meta)]
    return transformer.convert_to_graph_documents(docs)

def to_literal(v):
    
    if isinstance(v, bool):   return Literal(v)
    if isinstance(v, int):    return Literal(v, datatype=XSD.integer)
    if isinstance(v, float):  return Literal(v, datatype=XSD.decimal)
    return Literal(str(v))  # string

def _to_literal(v):
    if isinstance(v, bool):
        return Literal(v)
    if isinstance(v, int):
        return Literal(v, datatype=XSD.integer)
    if isinstance(v, float):
        return Literal(v, datatype=XSD.decimal)
    return Literal(str(v))

def _canon_rel(t: str) -> str | None:
    t = (t or "").strip().upper()
    return REL_MAP.get(t)

def _safe_id(s: str) -> str:
    """
    Gera um sufixo seguro para IRIs (sem espaços/acentos/especiais).
    Mantém legibilidade usando URL-encode quando necessário.
    """
    s = (s or "").strip()
    return quote(s, safe="-._~")  # não deixa espaços; preserva alguns chars

def _to_literal(v):
    if isinstance(v, bool):  return Literal(v)
    if isinstance(v, int):   return Literal(v, datatype=XSD.integer)
    if isinstance(v, float): return Literal(v, datatype=XSD.decimal)
    return Literal(str(v))

def _canon_rel(t: str) -> str | None:
    t = (t or "").strip().upper()
    return REL_MAP.get(t)

def _safe(s: str) -> str:
    
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)  # troca tudo que não é [0-9A-Za-z] por "_"
    s = re.sub(r"_+", "_", s).strip("_")  # colapsa múltiplos "_" e remove nas bordas
    
    return s or "item"

# -------- IRI builders no formato curto --------

def iri_user(u: str) -> URIRef:
    return URIRef(f"{ROOT_NS}{_safe(u)}")

def iri_doc(u: str, d: str) -> URIRef:
    return URIRef(f"{ROOT_NS}{_safe(u)}/{_safe(d)}")

def iri_chunk(u: str, d: str, chunk_id: str) -> URIRef:
    # usa o id completo do chunk, como você pediu (ex.: "5511993891773_749_2")
    return URIRef(f"{ROOT_NS}{_safe(u)}/{_safe(d)}/{_safe(chunk_id)}")

def iri_entity(u: str, d: str, eid: str) -> URIRef:
    # entidades agrupadas por usuário/documento
    return URIRef(f"{ROOT_NS}{_safe(u)}/{_safe(d)}/entidade/{_safe(eid)}")

def graphdocs_to_rdflib(graph_doc, base_ns: str = VOCAB_NS) -> ConjunctiveGraph:
    
    V = Namespace(base_ns.rstrip("/") + "/")

    cg = ConjunctiveGraph()
    cg.bind("", V); cg.bind("v", V)
    cg.bind("rdf", RDF); cg.bind("rdfs", RDFS); cg.bind("xsd", XSD); cg.bind("owl", OWL)

    meta        = graph_doc.source.metadata if getattr(graph_doc, "source", None) and getattr(graph_doc.source, "metadata", None) else {}
    id_usuario  = str(meta.get("id_usuario") or "").strip()     # "5511993891773"
    id_externo  = str(meta.get("id_externo") or "").strip()     # "749"
    chunk_full  = str(meta.get("id") or "").strip()             # "5511993891773_749_2"
    file_url    = meta.get("file_url")
    chunk_text  = getattr(graph_doc.source, "page_content", "") or ""

    # IRIs no formato curto
    user_iri  = iri_user(id_usuario)                            if id_usuario and id_usuario != "" else None
    doc_iri   = iri_doc(id_usuario, id_externo)                 if (id_usuario and id_externo) else None
    chunk_iri = iri_chunk(id_usuario, id_externo, chunk_full)   if (id_usuario and id_externo and chunk_full) else None

    # Contexto (named graph por usuário)
    ctx_iri = URIRef(f"{GR_NS}{_safe(id_usuario)}") if id_usuario else None
    g = cg.get_context(ctx_iri) if ctx_iri else cg.default_context

    # Usuario
    if user_iri:
        g.add((user_iri, RDF.type, V[USER_CLASS]))
        g.add((user_iri, V[P_USER_ID], Literal(id_usuario)))

    # Documento
    if doc_iri:
        g.add((doc_iri, RDF.type, V[DOC_CLASS]))
        g.add((doc_iri, V[P_DOC_ID], Literal(id_externo)))
        if file_url:
            g.add((doc_iri, V[P_DOC_FILE], Literal(file_url)))
        if user_iri:
            g.add((user_iri, V[P_HAS_DOCUMENT], doc_iri))
            g.add((user_iri, V[P_RESPONSAVEL_POR], doc_iri))

    # Chunk
    if chunk_iri:
        g.add((chunk_iri, RDF.type, V[CHUNK_CLASS]))
        # idChunk: você pode querer guardar também o sufixo numérico (opcional)
        g.add((chunk_iri, V[P_CHUNK_ID], Literal(chunk_full)))
        if chunk_text:
            g.add((chunk_iri, V[P_CHUNK_TEXT], Literal(chunk_text, lang="pt")))
        if doc_iri:
            g.add((chunk_iri, V[P_CONTAINS], doc_iri))  # Chunk -> Documento

    # Entidades extraídas
    id2iri = {}
    for n in getattr(graph_doc, "nodes", []) or []:
        n_id   = getattr(n, "id", None) or getattr(n, "name", None) or ""
        n_type = (getattr(n, "type", None) or "Entidade").strip()
        if not n_id:
            continue

        ent_iri = iri_entity(id_usuario, id_externo, n_id)
        id2iri[n_id] = ent_iri

        g.add((ent_iri, RDF.type, V[n_type]))

        props = getattr(n, "properties", {}) or {}
        for k, v in props.items():
            if k == "classe":
                continue
            # se quiser legibilidade, também adicione rdfs:label aqui:
            if k == P_DESCRICAO:
                g.add((ent_iri, V[k], Literal(str(v), lang="pt")))
            else:
                g.add((ent_iri, V[k], _to_literal(v)))

        # proveniência: ENTIDADE ->extraidode-> CHUNK
        if chunk_iri:
            g.add((ent_iri, V[P_EXTRACTED_FROM], chunk_iri))

    # Relações entre entidades
    for r in getattr(graph_doc, "relationships", []) or []:
        src = getattr(r, "source", None); tgt = getattr(r, "target", None)
        rty = _canon_rel(getattr(r, "type", None))
        if not (src and tgt and rty):
            continue
        s_id = getattr(src, "id", None); t_id = getattr(tgt, "id", None)
        if not (s_id and t_id):
            continue

        s_iri = id2iri.get(s_id) or iri_entity(id_usuario, id_externo, s_id)
        t_iri = id2iri.get(t_id) or iri_entity(id_usuario, id_externo, t_id)
        g.add((s_iri, V[rty], t_iri))

    return cg

def upload_turtle(ttl: bytes | str) -> bool:
    
    #print( ttl )
    
    params  = {"context": "<https://omc.co/graph/5511993891773>"}    
    headers = {"Content-Type": "text/turtle"}

    auth = HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD) if (GRAPHDB_USERNAME and GRAPHDB_PASSWORD) else None
    resp = requests.post(GRAPHDB_REPO_URL, params=params, data=ttl, headers=headers, auth=auth, timeout=120)
    resp.raise_for_status()
    
    return True

def main():
    
    print("-- início --")  
    inicio = time.time()      

    '''data = {
        "id":"5511993891773_748_68",
        "arquivo":"https://storage.googleapis.com/comtodos-607d6.appspot.com/5511969033344/5511993891773/Nouveaux_Regulamento_2023.pdf",
        "id_usuario":"5511995338283",
        "id_externo":748,
        "texto":"Essas multas são calculadas com base em qual taxa de condomínio? É aquela cota ordinária das unidades tipo de finais 3 e 4, certo?"
    }

    meta = {
        "id"        : str(data.get("id", "")),
        "doc_id"    : f'{data["id_usuario"]}_{data["id_externo"]}',
        "id_usuario": str(data["id_usuario"]),
        "id_externo": str(data["id_externo"]),
        "file_url"  : str(data["arquivo"])
    }'''

    query = {
        "_source"   : ["file_url", "id_usuario", "id_externo","texto"],
        "query"     : {
            "bool": {        
                "filter": [
                    { 
                        "term": { "id_usuario": "5511993891773" },
                        "term": { "id_externo": "749" }
                    } 
                ]
            },
        }, 
        "size": 1500
    }

    resp = elastic_client.search(index="documentos", body=query)

    total = len(resp["hits"]["hits"]) 

    for index, item in enumerate(resp["hits"]["hits"],start=1): 

        print( f'-> {index} de {total}')

        texto       = item['_source']['texto']
        id_usuario  = item['_source']['id_usuario']
        id_externo  = item['_source']['id_externo']
        file_url    = item['_source']['file_url']

        meta = {
            "id"        : str(item['_id']),
            "doc_id"    : f'{id_usuario}_{id_externo}',
            "id_usuario": str(id_usuario),
            "id_externo": str(id_externo),
            "file_url"  : str(file_url)
        }

        # 1) Extração (LLMGraphTransformer)
        graph_docs = extract_graph_docs(texto, meta)
        print(graph_docs)
        print(f"Nodes:{graph_docs[0].nodes}")
        print(f"Relationships:{graph_docs[0].relationships}")

        nodes = graph_docs[0].nodes
        rels  = graph_docs[0].relationships

        '''print( nodes )
        print('-'*50)
        print( rels )
        print('-'*50)
        print( meta )
        print('-'*50)'''
        
        '''
        for node in nodes:
            print(f" id='{node.id}',")
            print(f" type='{node.type}',")
            print(f" properties={node.properties}")
            print('-'*50)

        for rel in rels:
            print("Relationship(")
            print(f"  source=Node(id='{rel.source.id}', type='{rel.source.type}'),")
            print(f"  target=Node(id='{rel.target.id}', type='{rel.target.type}'),")
            print(f"  type='{rel.type}',")
            print(f"  properties={rel.properties}")
            print(")")
            print('-'*50)    
        '''

        proc_triplas = time.time()
        tpo_triplas  = proc_triplas - inicio

        print(f"-> Chunk processado em triplas, {tpo_triplas:.2f}s")    

        # 2) Conversão p/ RDF (rdflib)
        g   = graphdocs_to_rdflib(graph_docs[0], BASE_NS)
        ttl = g.serialize(format="turtle")

        #print(ttl)

        # 3) Upload bulk Turtle para GraphDB
        upload_turtle(ttl)

        proc_db = time.time()
        tpo_fim = proc_db - inicio

        print(f"-> Upload Turtle no GraphDB, {tpo_fim:.2f}s")        

if __name__ == "__main__":
    
    try:
        main()
        print("-- fim --")
    except requests.HTTPError as e:
        print("ERRO HTTP ao enviar para GraphDB:", e.response.status_code, e.response.text[:500])
    except Exception as e:
        print("ERRO:", repr(e))
