
import requests,itertools
from datetime import datetime

import igraph as ig
import leidenalg

from requests.auth  import HTTPBasicAuth
from collections import defaultdict

from src.config import settings

GRAPHDB_BASE_URL   = settings.GRAPHDB_BASE_URL
GRAPHDB_USERNAME   = settings.GRAPHDB_USERNAME
GRAPHDB_PASSWORD   = settings.GRAPHDB_PASSWORD
GRAPHDB_REPOSITORY = 'omc_v2' #settings.repositorio
GRAPHDB_REPO_URL   = f"{GRAPHDB_BASE_URL}/repositories/{GRAPHDB_REPOSITORY}/statements"
USUARIO            = "5511993891773"

PESOS = {
    "regras": 3.0,
    "atividades": 2.0,
    "pessoas": 1.5,
    "instituicoes": 1.0,
    "fundamentais": 0.3,
    "outros": 0.5,
}

def load():
    
    result = []
    query_geral = '''
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX v:   <https://omc.co/vocabulary/>

        SELECT ?chunk ?ent ?tipoEnt
            WHERE {
            GRAPH <https://omc.co/graph/5511993891773> {

                ?chunk rdf:type v:Chunk ;
                    v:estaContidoEm <https://omc.co/5511993891773/749> .

                ?ent v:relacionamento ?chunk .

                OPTIONAL { ?ent rdf:type ?tipoEnt . }

                FILTER (?ent != ?chunk)
            }
        }
    '''
  
    try:

        url     = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

        resp_rules = requests.post(
            url,
            data=query_geral,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME,settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )

        if resp_rules.status_code == 200:

            results  = resp_rules.json()  
            bindings = results.get("results", {}).get("bindings", []) 

            for b in bindings:
                chunk = b.get("chunk", {}).get("value")
                ent = b.get("ent", {}).get("value")
                tipoEnt = b.get("tipoEnt", {}).get("value")  # pode ser None

                if not chunk or not ent:
                    continue

                result.append((chunk, ent, tipoEnt))

        return result

    except Exception as e:

        error_msg = f"Graph ERROR: {str(e)}"
        print(f"--> {error_msg}")
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }

def macro_tipo(ent_uri, tipoEnt_uri=None):
    """
    Classificação semântica baseada prioritariamente no rdf:type (TBox).
    """

    if tipoEnt_uri:
        t = tipoEnt_uri.lower()

        # 1. Regras e normas
        if any(x in t for x in [
            "regra", "artigo", "inciso", "paragrafo", "regimentointerno"
        ]):
            return "regras"

        # 2. Pessoas e papéis
        if any(x in t for x in [
            "morador", "sindico", "subsidico", "conselheiro", "papel", "visitante", "funcionario"
        ]):
            return "pessoas"

        # 3. Atividades sociais / uso
        if any(x in t for x in [
            "evento", "reserva", "atividade"
        ]):
            return "atividades"

        # 4. Instituições externas
        if any(x in t for x in [
            "administradora", "concessionaria", "empresa"
        ]):
            return "instituicoes"

        # 5. Estrutura fundamental do condomínio
        if any(x in t for x in [
            "condominio", "edificacao", "areacomum", "garagem", "documento"
        ]):
            return "fundamentais"

    # 6. Fallback (somente se tipo não existir)
    s = ent_uri.lower()

    if "regra" in s or "art" in s:
        return "regras"
    if "sindico" in s or "morador" in s:
        return "pessoas"

    return "outros"

def project_chunk_graph(incidencias, max_chunks_por_entidade=200):
    
    ent_to_chunks = defaultdict(list)
    ent_to_macro = {}

    for chunk, ent, tipoEnt in incidencias:
        ent_to_chunks[ent].append(chunk)
        ent_to_macro[ent] = macro_tipo(ent, tipoEnt)

    edge_w = defaultdict(float)

    for ent, chunks in ent_to_chunks.items():
        if len(chunks) < 2:
            continue

        peso = PESOS.get(ent_to_macro.get(ent, "outros"), 0.5)

        uniq = list(dict.fromkeys(chunks))
        if len(uniq) > max_chunks_por_entidade:
            uniq = uniq[:max_chunks_por_entidade]

        for a, b in itertools.combinations(uniq, 2):
            if a > b:
                a, b = b, a
            edge_w[(a, b)] += peso

    return edge_w

def run_leiden(edge_w, resolution=1.3, n_iterations=10):
    
    if not edge_w:
        raise RuntimeError("Grafo projetado vazio: não há arestas chunk–chunk derivadas das entidades.")

    chunks = sorted({c for e in edge_w for c in e})
    idx = {c: i for i, c in enumerate(chunks)}

    g = ig.Graph()
    g.add_vertices(len(chunks))
    g.add_edges([(idx[a], idx[b]) for (a, b) in edge_w.keys()])
    g.es["weight"] = [edge_w[(a, b)] for (a, b) in edge_w.keys()]

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    membership = part.membership
    chunk_to_comm = {chunks[i]: membership[i] for i in range(len(chunks))}

    return chunk_to_comm, part

def build_ttl(chunk_to_comm, user_id="5511993891773", doc_id="749", resolution=1.0, run_tag="r1"):
    
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    lines = []
    lines.append("@prefix v: <https://omc.co/vocabulary/> .")
    lines.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")

    # cria recursos de community
    comm_ids = sorted(set(chunk_to_comm.values()))
    for cid in comm_ids:
        comm_uri = f"https://omc.co/community/{user_id}/{doc_id}/leiden/{run_tag}/{cid:05d}"
        lines.append(f"<{comm_uri}> a v:Community ;")
        lines.append(f'  v:communityMethod "leiden" ;')
        lines.append(f'  v:communityResolution "{resolution}" ;')
        lines.append(f'  v:communityDoc "{doc_id}" ;')
        lines.append(f'  v:communityRun "{ts}" .\n')

    # liga chunks à community
    for chunk_uri, cid in chunk_to_comm.items():
        comm_uri = f"https://omc.co/community/{user_id}/{doc_id}/leiden/{run_tag}/{cid:05d}"
        lines.append(f"<{chunk_uri}> v:inCommunity <{comm_uri}> .")

    return "\n".join(lines)

def upload_turtle(ttl: bytes | str, USUARIO: str) -> bool:    
    
    params  = {"context": f"<https://omc.co/graph/{USUARIO}>"}    
    headers = {"Content-Type": "text/turtle"}

    auth = HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD) if (GRAPHDB_USERNAME and GRAPHDB_PASSWORD) else None
    resp = requests.post(GRAPHDB_REPO_URL, params=params, data=ttl, headers=headers, auth=auth, timeout=1200)
    resp.raise_for_status()
    
    return True

def community_ingest(resolution=1.0, n_iterations=10):
    
    incidencias = load()

    print(f"--> {len(incidencias)} itens, resolution: {resolution}")

    edge_w = project_chunk_graph(incidencias, max_chunks_por_entidade=200)
    print(f"[INFO] arestas projetadas: {len(edge_w)}")

    chunk_to_comm, part = run_leiden(edge_w, resolution=resolution, n_iterations=n_iterations)
    n_comms = len(set(chunk_to_comm.values()))
    print(f"[INFO] communities: {n_comms}")
    print(f"[INFO] modularity (aprox): {part.modularity:.4f}\n")
    
    ttl = build_ttl(chunk_to_comm, user_id="5511993891773", doc_id="749", resolution=resolution, run_tag="r1")

    print( ttl )

    upload_turtle(ttl,USUARIO)

    return chunk_to_comm, ttl