import os, re, unicodedata, warnings, time
from typing import List, Dict, Any
from pathlib import Path
from pprint import pprint

from rich import print

# RDF / SPARQL
from rdflib             import Graph, URIRef, Literal, Namespace, ConjunctiveGraph
from rdflib.namespace   import RDF, RDFS, XSD, OWL

# HTTP
import requests
from requests.auth import HTTPBasicAuth

# LLM / Extração
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_openai           import ChatOpenAI, OpenAIEmbeddings
from langchain_together         import ChatTogether     
from langchain_ollama           import ChatOllama
from langchain_anthropic        import ChatAnthropic
from langchain_google_genai     import ChatGoogleGenerativeAI

from src_en.config import settings

load_dotenv()
warnings.filterwarnings("ignore")

# REF
# https://reference.langchain.com/v0.3/python/experimental/graph_transformers/langchain_experimental.graph_transformers.llm.LLMGraphTransformer.html

'''
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
GRAPHDB_BASE_URL   = settings.GRAPHDB_BASE_URL
GRAPHDB_USERNAME   = settings.GRAPHDB_USERNAME
GRAPHDB_PASSWORD   = settings.GRAPHDB_PASSWORD
GRAPHDB_REPOSITORY = settings.repositorio
GRAPHDB_REPO_URL   = f"{GRAPHDB_BASE_URL}/repositories/{GRAPHDB_REPOSITORY}/statements"

BASE_NS            = os.getenv("OMC_BASE_NS", "https://omc.co/vocabulary/").strip()

VOCAB_NS = "https://omc.co/vocabulary/"
ROOT_NS  = "https://omc.co/"
GR_NS    = "https://omc.co/graph/usuario/"

INGEST_DIR = Path(__file__).resolve().parent
ONTOLOGY_XML_PATH = INGEST_DIR / "ontology.xml"
PROMPT_ONTOLOGY_PATH = INGEST_DIR / "ontology.xml"

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
P_EXTRACTED_FROM  = "relacionamento"    # Entidade -> Chunk
P_DESCRICAO       = "descricao"
USUARIO           = ""

#LLM_MODEL = ChatOpenAI(model="gpt-5.3-chat-latest")
LLM_MODEL = ChatOpenAI(model="o4-mini")
#LLM_MODEL = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview",temperature=0)
#LLM_MODEL = ChatAnthropic(model="claude-sonnet-4-6")

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

def _local_name(uri: URIRef) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rstrip("/").rsplit("/", 1)[-1]

def load_ontology_classes(path: Path = ONTOLOGY_XML_PATH) -> set[str]:
    graph = Graph()
    graph.parse(str(path), format="xml")
    return {
        name
        for name in (_local_name(subject) for subject in graph.subjects(RDF.type, OWL.Class))
        if name
    }

def load_ontology_object_properties(path: Path = ONTOLOGY_XML_PATH) -> set[str]:
    graph = Graph()
    graph.parse(str(path), format="xml")
    return {
        name
        for name in (_local_name(subject) for subject in graph.subjects(RDF.type, OWL.ObjectProperty))
        if name
    }

ALLOWED_NODE_PROPS = {"descricao"}  # mantenha enxuto

#--------------------------------------------------------------

def _rel_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", (name or "").upper())

CANON_TYPES = load_ontology_classes()
CANON_RELATIONSHIPS = load_ontology_object_properties()
REL_MAP = {_rel_key(rel): rel for rel in CANON_RELATIONSHIPS}

def build_graph_transformer() -> LLMGraphTransformer:
    
    allowed_rels = sorted(list(CANON_RELATIONSHIPS))

    with open(PROMPT_ONTOLOGY_PATH, encoding="utf-8") as f:
        ontology_txt = f.read()   

    system_txt = (f'''You are a legal knowledge graph extractor for GDPR texts.
    Follow this ontology:
    {ontology_txt}
    Your main role is to extract normative rules that are explicitly stated or can be derived only when there is clear textual support.
    Represent each rule as an instance of :Rule connected to the rest of the graph, using only the ontology properties that are necessary and semantically appropriate.

    Definition of rule:
    A rule is a prescriptive statement in a normative document that regulates the behavior of agents by assigning a deontic modality
    such as obligation, permission, or prohibition to a specific action performed by an agent under defined applicability conditions.
    
    Rule extraction examples:
    1) Sensitive or biometric data:
    create an instance of :Rule and link it with:
    :appliesTo -> (:SensitivePersonalData or another applicable GDPR data class)
    :refersTo -> :Processing

    2) Prohibitions:
    create an instance of :Rule and link it with:
    :appliesTo -> (:PersonalData or :SensitivePersonalData)
    :refersTo -> the prohibited activity or concept represented in the ontology

    3) Permissions and conditions:
    create an instance of :Rule and link it with:
    :appliesTo -> (:PersonalData or :SensitivePersonalData)
    :refersTo -> :Processing
    :appliesToPurpose -> the applicable processing purpose when the text clearly states one

    4) Evidence:
    link each extracted :Rule or legal text element to the source chunk using:
    :hasEvidenceIn -> :Chunk

    Every rule must be connected to other ontology classes whenever the text provides clear support.
    Each article should have at least one deontic rule when the article states what is obligatory, permitted, or prohibited.

    Examples of concise descriptions for :Rule:
    - Prohibition of processing sensitive personal data without a valid legal basis
    - Permission to process personal data for a specified and legitimate purpose

    Produce faithful and concise output.
    ''')       

    pt_template = ChatPromptTemplate.from_messages([
        ("system", system_txt),
        ("human", "Text:\n{input}\n")
    ])
    
    return LLMGraphTransformer(
        llm=LLM_MODEL,
        prompt=pt_template,
        allowed_nodes=sorted(CANON_TYPES),
        allowed_relationships=allowed_rels,
        node_properties=list(ALLOWED_NODE_PROPS),
    )

async def extract_graph_docs(text, meta):

    transformer = build_graph_transformer()
    docs = [Document(page_content=text, metadata=meta)]
    return transformer.convert_to_graph_documents(docs)

def _to_literal(v):
    if isinstance(v, bool):
        return Literal(v)
    if isinstance(v, int):
        return Literal(v, datatype=XSD.integer)
    if isinstance(v, float):
        return Literal(v, datatype=XSD.decimal)
    return Literal(str(v))

def _canon_rel(t: str) -> str | None:
    return REL_MAP.get(_rel_key(t))

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
    user_iri  = iri_user(id_usuario)                            if (id_usuario and id_usuario != "") else None
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

def upload_turtle(ttl: bytes | str, USUARIO: str) -> bool:    
    
    params  = {"context": f"<https://omc.co/graph/{USUARIO}>"}    
    headers = {"Content-Type": "text/turtle"}

    auth = HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD) if (GRAPHDB_USERNAME and GRAPHDB_PASSWORD) else None
    resp = requests.post(GRAPHDB_REPO_URL, params=params, data=ttl, headers=headers, auth=auth, timeout=1200)
    resp.raise_for_status()
    
    return True

async def graph_ingest_gpdr(data,debug=False):
    
    print("--- ingest ---")  
    inicio = time.time()      

    _data = {
        "id":"5511993891773_749_2",
        "arquivo":"https://storage.googleapis.com/comtodos-607d6.appspot.com/5511969033344/5511993891773/Nouveaux_Regulamento_2023.pdf",
        "id_usuario":"5511993891773",
        "id_externo":749,
        "texto":"CONSIDERANDO a viv\u00eancia em comunidade e o interesse geral em prol de\numa vida solid\u00e1ria e que assegure o bem-estar de todos, os COND\u00d4MINOS\ndo CONDOM\u00cdNIO ALTO DO IPIRANGA NOUVEAUX aprovam o\nseguinte Regulamento Interno, em Assembleia soberana e regularmente convocada\npara tal fim e observando os ditames da lei.\n\nCAP\u00cdTULO I- DAS DISPOSI\u00c7\u00d5ES GERAIS\n\nArtigo 1\u00ba - Todos os propriet\u00e1rios de unidades, promitentes compradores,\ncession\u00e1rios e promitentes cession\u00e1rios, atuais e futuros, ocupantes e locat\u00e1rios,\ndoravante denominados comum e genericamente\nCOND\u00d4MINOS/COND\u00d4MINO ficam obrigados a cumprir e fazer cumprir\nas determina\u00e7\u00f5es constantes do presente Regulamento Interno.\n\nPar\u00e1grafo \u00danico. Empregado(s) ou prestador (es) de servi\u00e7o a servi\u00e7o do\nCONDOM\u00cdNIO s\u00e3o de responsabilidade do CONDOM\u00cdNIO, sob gest\u00e3o do\nSindico e devem obrigatoriamente seguir os termos do Contrato de Presta\u00e7\u00e3o de\nServi\u00e7o, bem como a Conven\u00e7\u00e3o e este Regulamento Interno."
    }

    _data = {
        "id":"5511993891773_749_1",
        "arquivo":"https://storage.googleapis.com/comtodos-607d6.appspot.com/5511969033344/5511993891773/Nouveaux_Regulamento_2023.pdf",
        "id_usuario":"5511993891773",
        "id_externo":749,
        "texto":"CAPÍTULO XVI DO GARAGE BAND CAPÍTULO XVII - DO ESPAÇO PARA RECREAÇÃO INFANTIL E BRINQUEDOTECA..... CAPÍTULO XVIII - DA LAN HOUSE (SALA DE JOGOS JUVENIL) CAPÍTULO XIX ? DO BICICLETÁRIO CAPÍTULO XX ? DO REDÁRIO .... CAPÍTULO XXI ? DAS PENALIDADES... CAPÍTULO XXII - DISPOSIÇÕES FINAIS ... Página 2 de 43  Regulamento Interno CONDOMÍNIO ALTO DO IPIRANGA NOUVEAUX INTRODUÇÃO CONSIDERANDO os termos da Convenção do CONDOMÍNIO ALTO DO IPIRANGA NOUVEAUX. CONSIDERANDO que tal Regulamento é obrigatório para proprietários de unidades, promitentes compradores, cessionários e promitentes cessionários e locatários, atuais e futuros, como para qualquer ocupante das unidades autônomas."
    }

    meta = {
        "id"        : str(data.get("id", "")),
        "doc_id"    : f'{data["id_usuario"]}_{data["id_externo"]}',
        "id_usuario": str(data["id_usuario"]),
        "id_externo": str(data["id_externo"]),
        "file_url"  : str(data["arquivo"])
    }

    print(f'--> doc: {str(data.get("id", ""))}')

    USUARIO = str(data["id_usuario"])

    # 1) Extração (LLMGraphTransformer)
    graph_docs = await extract_graph_docs(data["texto"], meta)

    if debug==True:
    
        print('--> triplas extraidas')
        print(graph_docs)
        print(f"Nodes:{graph_docs[0].nodes}")
        print(f"Relationships:{graph_docs[0].relationships}")  
        
        nodes = graph_docs[0].nodes
        rels  = graph_docs[0].relationships

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
    

    proc_triplas = time.time()
    tpo_triplas  = proc_triplas - inicio

    print(f"--> Chunk processado em triplas, {tpo_triplas:.2f}s")    

    # 2) Conversão p/ RDF (rdflib)
    g   = graphdocs_to_rdflib(graph_docs[0], BASE_NS)
    ttl = g.serialize(format="turtle")
    
    print('--> ttl pronto')

    if debug==False:
        # 3) Upload bulk Turtle para GraphDB
        upload_turtle(ttl,USUARIO)
    else:
        print( f'\n{ttl}' )

    proc_db = time.time()
    tpo_fim = proc_db - inicio

    if debug==False:
        print(f"--> Upload Turtle no GraphDB, {tpo_fim:.2f}s")
    else:
        print(f"--> Debug mode, {tpo_fim:.2f}s")

    print("-- fim --")

    return 'OK'

