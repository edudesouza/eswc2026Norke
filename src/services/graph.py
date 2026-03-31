
import requests, re, textwrap, ast

from requests.auth  import HTTPBasicAuth

from src.config     import settings
from src.utils.text import normalize

def graph_search(class_rules,keyword,question,named_graph,retrieval_size):

    print( f'--> search graph ({settings.repositorio})')

    status         = ''
    knowledge_base = {}
    resp_final     = resp_rules_toon = resp_chunks_toon  = ""
    keyword        = re.sub(r'[\\/]+', ' ', keyword) 

    try:

        query_class_rules = f'''
            PREFIX :     <https://omc.co/vocabulary/>
            PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT DISTINCT ?regraURI ?descricao ?texto ?idChunk ?tipo
            FROM <https://omc.co/graph/5511993891773>
            WHERE {{

            VALUES ?classe {{
                {class_rules}
            }}

            # 1. Instância da classe extraída pelo LLM
            ?instancia rdf:type ?classe .

            # 2. Chunk hub conectado à instância (em qualquer direção)
            {{
                ?instancia :relacionamento ?chunk .
            }} UNION {{
                ?chunk :relacionamento ?instancia .
            }}
            ?chunk rdf:type :Chunk .

            # 3. Regras conectadas ao mesmo chunk hub
            {{
                ?regraURI :relacionamento ?chunk .
            }} UNION {{
                ?chunk :relacionamento ?regraURI .
            }}
            ?regraURI rdf:type/rdfs:subClassOf* :Regra .

            OPTIONAL {{
                ?regraURI :descricao ?descricao .
                FILTER(LANGMATCHES(LANG(?descricao), "pt") || LANG(?descricao) = "")
            }}
            OPTIONAL {{ ?regraURI rdf:type ?tipo }}
            OPTIONAL {{ ?chunk :texto ?textoBruto .
                FILTER(LANGMATCHES(LANG(?textoBruto), "pt") || LANG(?textoBruto) = "")
            }}
            OPTIONAL {{ ?chunk :idChunk ?idChunk }}

            BIND(COALESCE(?textoBruto, ?descricao) AS ?texto)
            }}
            ORDER BY ?regraURI
        '''

        query_rules = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?score ?idChunk ?texto ?regraURI ?tipo ?descricao ?predicado ?direcao
            FROM <https://omc.co/graph/{named_graph}>
            WHERE {{
            {{
                # 1. SUBQUERY: Busca focada na REGRA (onde a descrição é boa)
                SELECT ?regra (MAX(?s) AS ?score)
                WHERE {{
                ?q a luc-index:omc_full ;
                    # Usamos a query rica em palavras-chave que você forneceu
                    luc:query '{keyword}' ;
                    luc:entities ?regra .

                ?regra luc:score ?s .
                }}
                GROUP BY ?regra
                ORDER BY DESC(?score)
                LIMIT 500
            }}

            # 2. FILTRO DE TIPO COM REASONING
            # Garante que o item encontrado é uma Regra ou uma subclasse de Regra
            ?regra a/rdfs:subClassOf* :Regra .
            
            # Pega o tipo específico para exibir (ex: https://omc.co/vocabulary/Regra)
            OPTIONAL {{ ?regra a ?tipo }}

            # 3. RECUPERA A DESCRIÇÃO (O motivo do match)
            ?regra :descricao ?descricao . FILTER(LANG(?descricao) = "" || LANGMATCHES(LANG(?descricao), "pt"))

            # 4. EXPANSÃO: BUSCA O CHUNK CONECTADO (Para não vir vazio)
            OPTIONAL {{
                
                # Procura vizinhos em qualquer direção (Regra->Chunk OU Chunk->Regra)
                #{{ ?regra ?p ?chunkNode . }} UNION {{ ?chunkNode ?p ?regra . }}

                # O filtro é: esse vizinho é um Chunk e tem texto?
                #?chunkNode a :Chunk ;
                #    :texto ?textoBruto .

                {{
                    # Direção 1: Regra -> Chunk
                    ?regra ?p ?chunkNode .
                    ?chunkNode a :Chunk .
                    BIND(?p AS ?predicado)
                    BIND("regra→chunk" AS ?direcao)
                }} UNION {{
                    # Direção 2: Chunk -> Regra  
                    ?chunkNode ?p ?regra .
                    ?chunkNode a :Chunk .
                    BIND(?p AS ?predicado)
                    BIND("chunk→regra" AS ?direcao)
                }}

                # Texto do chunk (fora do UNION para não duplicar)
                ?chunkNode :texto ?textoBruto .
                
                # Valida idioma e atribui à variável de saída
                FILTER(LANG(?textoBruto) = "" || LANGMATCHES(LANG(?textoBruto), "pt"))
                BIND(?textoBruto AS ?texto)

                # Tenta pegar o ID do Chunk
                OPTIONAL {{ ?chunkNode :idChunk ?id }}
                BIND(?id AS ?idChunk)
            }}

            BIND(?regra AS ?regraURI)
            }}
            ORDER BY DESC(?score)
            LIMIT {retrieval_size}
        '''   

        query_rules_expandida = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?score ?idChunk ?texto ?regraURI ?tipo ?descricao
            FROM <https://omc.co/graph/5511993891773>
            WHERE {{

            # ── BRANCH A: pivot é uma :Regra ──────────────────────────────────────
            {{
                {{
                SELECT ?regra (MAX(?s) AS ?score)
                WHERE {{
                    ?q a luc-index:omc_full ;
                    luc:query '{keyword}' ;
                    luc:entities ?regra .
                    ?regra luc:score ?s .
                    ?regra a/rdfs:subClassOf* :Regra .
                }}
                GROUP BY ?regra
                ORDER BY DESC(?score)
                LIMIT 100
                }}

                OPTIONAL {{ ?regra a ?tipo }}
                OPTIONAL {{
                ?regra :descricao ?descricao .
                FILTER(LANGMATCHES(LANG(?descricao), "pt") || LANG(?descricao) = "")
                }}

                OPTIONAL {{
                {{
                    ?regra ?p ?chunkNode .
                    ?chunkNode a :Chunk .
                }} UNION {{
                    ?chunkNode ?p ?regra .
                    ?chunkNode a :Chunk .
                }}
                ?chunkNode :texto ?textoBruto .
                FILTER(LANGMATCHES(LANG(?textoBruto), "pt") || LANG(?textoBruto) = "")
                OPTIONAL {{ ?chunkNode :idChunk ?idChunk }}
                }}

                BIND(COALESCE(?textoBruto, ?descricao) AS ?texto)
                BIND(?regra AS ?regraURI)
            }}

            UNION

            # ── BRANCH B: pivot é um :Chunk direto ────────────────────────────────
            {{
                {{
                SELECT ?chunkNode (MAX(?s) AS ?score)
                WHERE {{
                    ?q a luc-index:omc_full ;
                    luc:query  '{keyword}' ;
                    luc:entities ?chunkNode .
                    ?chunkNode luc:score ?s .
                    ?chunkNode a :Chunk .
                }}
                GROUP BY ?chunkNode
                ORDER BY DESC(?score)
                LIMIT 100
                }}

                ?chunkNode :texto ?textoBruto .
                FILTER(LANGMATCHES(LANG(?textoBruto), "pt") || LANG(?textoBruto) = "")

                OPTIONAL {{
                {{
                    ?chunkNode ?p ?regra .
                    ?regra a/rdfs:subClassOf* :Regra .
                }} UNION {{
                    ?regra ?p ?chunkNode .
                    ?regra a/rdfs:subClassOf* :Regra .
                }}
                OPTIONAL {{
                    ?regra :descricao ?descricao .
                    FILTER(LANGMATCHES(LANG(?descricao), "pt") || LANG(?descricao) = "")
                }}
                }}

                OPTIONAL {{ ?chunkNode a ?tipo }}
                OPTIONAL {{ ?chunkNode :idChunk ?idChunk }}

                BIND(?textoBruto AS ?texto)
                BIND(COALESCE(?regra, ?chunkNode) AS ?regraURI)
            }}

            }}
            ORDER BY DESC(?score)
            LIMIT 20
        '''

        query_chunks = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?idChunk ?score ?texto ?descricao ?documento ?tipo
            FROM <https://omc.co/graph/{named_graph}>
            WHERE {{  
            {{
                SELECT ?chunk (MAX(?s) AS ?score) (SAMPLE(?id) AS ?idChunk) (SAMPLE(?t)  AS ?texto) (SAMPLE(?descRegra) AS ?descricao)
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
                # caso 1: chunk dentro de um documento
                ?chunk :estaContidoEm ?documento .
                OPTIONAL {{ ?documento a ?tipo }}
            }}UNION{{
                # caso 2: o próprio chunk é um documento/regra
                BIND(?chunk AS ?documento) .
                OPTIONAL {{ ?documento a ?tipo }}
            }}
            }}

            FILTER EXISTS {{
                VALUES ?tipoPermitido {{ :Chunk :Regra :Paragrafo }}
                #VALUES ?tipoPermitido {{ :Chunk }}
                ?chunk a ?tipoPermitido .
            }}
        }}
        ORDER BY DESC(?score)
        LIMIT {retrieval_size}
        '''

        query_similaridade = f'''
            PREFIX : <http://www.ontotext.com/graphdb/similarity/>
            PREFIX inst: <http://www.ontotext.com/graphdb/similarity/instance/>
            PREFIX v: <https://omc.co/vocabulary/>

            SELECT ?score ?idChunk ?texto
            FROM <https://omc.co/graph/{named_graph}>
            WHERE {{
                ?search a inst:similarity_v1 ;
                        :searchTerm '{keyword}';
                        :documentResult ?result .

                ?result :value ?documentID ;   		
                        :score ?score .    		

                OPTIONAL {{ ?chunk v:idChunk ?idChunk }}
                OPTIONAL {{ ?chunk v:texto  ?texto }}
                
            }}
            ORDER BY DESC(?score)
            LIMIT {retrieval_size} 
        '''

        url     = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

        '''print( query_class_rules )
        print( '-'*100 )
        print( query_chunks )'''

        # Buscar apenas Regras

        resp_rules = requests.post(
            url,
            data=query_class_rules,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME,settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )

        if resp_rules.status_code == 200:
            
            results  = resp_rules.json()  
            bindings = results.get("results", {}).get("bindings", [])  

            resp_rules_toon = 'id_chunk;texto_regra\n'  

            for index,item in enumerate(bindings,1):        

                score     = float( item.get("score", {}).get("value", 0) )
                score     = round(score, 3)
                id_chunk  = item.get("idChunk", {}).get("value", "")
                regra_uri = item.get("regraURI", {}).get("value", "")
                descricao = normalize(item.get("descricao", {}).get("value", ""))            

                resp_rules_toon += f'{id_chunk};{descricao}\n' 
            
                knowledge_base[regra_uri] = descricao
        
        else:

            error_msg = f"Graph DB ERROR ({resp_rules.status_code}): {resp_rules.text}"
            print(f"--> {error_msg}")     
    
            return {
                'status': 'ERROR', 
                'response': error_msg, 
                'dataset': {}
            }         

        # Buscar apenas Chunks

        resp_chunks = requests.post(
            url,
            data=query_chunks,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )

        if resp_chunks.status_code == 200:
            
            results   = resp_chunks.json()  
            bindings = results.get("results", {}).get("bindings", [])

            resp_chunks_toon = 'id_chunk;score;texto_chunk\n'

            for index,item in enumerate(bindings,1):        

                score    = float( item.get("score", {}).get("value", 0) )
                score    = round(score, 3)
                id_chunk = item.get("idChunk", {}).get("value", "")
                texto    = normalize(item.get("texto", {}).get("value", ""))

                resp_chunks_toon += f'{id_chunk};{score};{texto}\n' 
                knowledge_base[id_chunk] = texto
                
        else:
            
            error_msg = f"Graph DB ERROR ({resp_chunks.status_code}): {resp_chunks.text}"
            print(f"--> {error_msg}")     
    
            return {
                'status': 'ERROR', 
                'response': error_msg, 
                'dataset': {}
            } 
        
        resp_final = f'''Regras:
        {textwrap.dedent(resp_rules_toon)}
        Chunks:
        {textwrap.dedent(resp_chunks_toon) }
        ''' 

        return {
            'status':'OK',
            'response':resp_final,
            'dataset':knowledge_base
        }
    
    except Exception as e:

        error_msg = f"Graph ERROR: {str(e)}"
        print(f"--> {error_msg}")
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }
