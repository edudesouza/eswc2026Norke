
import sys, requests, re, textwrap, ast

from requests.auth  import HTTPBasicAuth

from src_en.config     import settings
from src_en.utils.text import normalize

'''Graph Reasoning Distance (GRD), pensar!'''

def graph_search(class_rules,expantion,keyword,question,named_graph,retrieval_size):

    print( f'\n--> search graph ({settings.repositorio})')

    status         = ''
    filter         = ''
    knowledge_base = {}
    resp_final     = resp_rules_toon = resp_chunks_toon  = ""
    filter_article = filter_chapter  = ""    
    keyword        = re.sub(r'[\\/]+', ' ', keyword)  
    keyword        = re.sub(r'([\\\'"])', r'\\\1', keyword)
    article        = expantion['query_expansion']['article']
    chapter        = expantion['query_expansion']['chapter']     

    if article and article.strip().lower() not in ('none', 'null'):
        #article = article.lower()
        #filter_article = f'CONTAINS(LCASE(STR(?bc)), "{article}")'

        articles = [
            a.strip().lower()
            for a in article.split('|')
            if a.strip()
        ]

        conditions = [
            f'CONTAINS(LCASE(STR(?bc)), "{a}")'
            for a in articles
        ]

        filter_article = " || ".join(conditions)
        
        retrieval_size = 50

    if chapter and chapter.strip().lower() not in ('none', 'null'):
        #chapter = chapter.lower()
        #filter_chapter = f'CONTAINS(LCASE(STR(?bc)), "{chapter}")' 

        chapter = [
            a.strip().lower()
            for a in chapter.split('|')
            if a.strip()
        ]

        conditions = [
            f'CONTAINS(LCASE(STR(?bc)), "{a}")'
            for a in chapter
        ]

        filter_chapter = " || ".join(conditions)
        
        retrieval_size = 50   

    if filter_article:
        filter = f'FILTER( {filter_article} )'

    if filter_chapter:
        filter = f'FILTER( {filter_chapter} )'   

    if filter_article and filter_chapter:
        filter = f'FILTER( {filter_article} && {filter_chapter} )' 

    try:

        query_class_rules = f'''
            PREFIX :     <https://omc.co/vocabulary/>
            PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT DISTINCT ?regraURI ?descricao ?texto ?breadcrumb ?tipo
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
            ?regraURI rdf:type/rdfs:subClassOf* :Rule .

            OPTIONAL {{
                ?regraURI :descricao ?descricao .
                #FILTER(LANGMATCHES(LANG(?descricao), "pt") || LANG(?descricao) = "")
            }}
            OPTIONAL {{ ?regraURI rdf:type ?tipo }}
            OPTIONAL {{ ?chunk :texto ?textoBruto .
                #FILTER(LANGMATCHES(LANG(?textoBruto), "pt") || LANG(?textoBruto) = "")
            }}
            OPTIONAL {{ ?chunk :idChunk ?breadcrumb }}

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
                SELECT ?rule (MAX(?s) AS ?score)
                WHERE {{
                ?q a luc-index:omc_full ;
                    # Usamos a query rica em palavras-chave que você forneceu
                    luc:query '{keyword}' ;
                    luc:entities ?rule .

                ?rule luc:score ?s .
                }}
                GROUP BY ?rule
                ORDER BY DESC(?score)
                LIMIT 500
            }}

            # 2. FILTRO DE TIPO COM REASONING
            # Garante que o item encontrado é uma Regra ou uma subclasse de Regra
            ?rule a/rdfs:subClassOf* :Rule .
            
            # Pega o tipo específico para exibir (ex: https://omc.co/vocabulary/Regra)
            OPTIONAL {{ ?rule a ?tipo }}

            # 3. RECUPERA A DESCRIÇÃO (O motivo do match)
            ?rule :descricao ?descricao . FILTER(LANG(?descricao) = "" || LANGMATCHES(LANG(?descricao), "pt"))

            # 4. EXPANSÃO: BUSCA O CHUNK CONECTADO (Para não vir vazio)
            OPTIONAL {{
                
                # Procura vizinhos em qualquer direção (Regra->Chunk OU Chunk->Regra)
                #{{ ?rule ?p ?chunkNode . }} UNION {{ ?chunkNode ?p ?rule . }}

                # O filtro é: esse vizinho é um Chunk e tem texto?
                #?chunkNode a :Chunk ;
                #    :texto ?textoBruto .

                {{
                    # Direção 1: Regra -> Chunk
                    ?rule ?p ?chunkNode .
                    ?chunkNode a :Chunk .
                    BIND(?p AS ?predicado)
                    BIND("regra→chunk" AS ?direcao)
                }} UNION {{
                    # Direção 2: Chunk -> Regra  
                    ?chunkNode ?p ?rule .
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

            BIND(?rule AS ?regraURI)
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

        query_chunks_v1 = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:        <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?point ?score ?texto ?documento ?tipo
            FROM <https://omc.co/graph/{named_graph}>
            WHERE {{  

            {{
                SELECT ?point 
                    (MAX(?s) AS ?score) 
                    (SAMPLE(?t) AS ?texto)
                WHERE {{

                #Busca Lucene (transversal)
                ?q a luc-index:omc_full ;
                    luc:query  '{keyword}' ;
                    luc:entities ?point .

                ?point luc:score ?s .

                #FILTRO COM SUBCLASSES (o ponto chave)
                ?point rdf:type/rdfs:subClassOf* :Point .

                FILTER EXISTS {{
                    ?point rdf:type/rdfs:subClassOf* :Point .
                }}

                OPTIONAL {{
                    ?point :fullText ?t .
                    #FILTER(LANG(?t) = "" || LANGMATCHES(LANG(?t), "en"))
                }}

                }}
                GROUP BY ?point
                ORDER BY DESC(?score)
                LIMIT 200
            }}

            # Documento pai (artigo, seção, etc.)
            OPTIONAL {{
                ?point :is_part_of ?documento .
                OPTIONAL {{ ?documento rdf:type ?tipo }}
            }}

            }}
        ORDER BY DESC(?score)
        LIMIT {retrieval_size}
        '''

        query_chunks = f'''
            PREFIX :           <https://omc.co/vocabulary/>
            PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
            PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:        <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?point ?breadcrumb ?score ?texto ?documento ?tipo
            FROM <https://omc.co/graph/{named_graph}>
            WHERE {{  

                {{
                    SELECT ?point 
                        (MAX(?s) AS     ?score) 
                        (SAMPLE(?t) AS  ?texto)
                        (SAMPLE(?bc) AS ?breadcrumb) 
                    WHERE {{

                        #Busca Lucene (transversal)
                        ?q a luc-index:eu_full ;
                            luc:query  '{keyword}' ;
                            luc:entities ?point .

                        ?point luc:score ?s .
                        #FILTRO COM SUBCLASSES (o ponto chave)
                        ?point rdf:type/rdfs:subClassOf* :Point .                    

                        OPTIONAL {{ ?point :fullText ?t . }}
                        OPTIONAL {{ ?point :breadcrumb ?bc . }} 
                        {filter}                     

                    }}
                    GROUP BY ?point
                    ORDER BY DESC(?score)
                    LIMIT 200
                }}

                # Documento pai (artigo, seção, etc.)
                OPTIONAL {{
                    ?point :is_part_of ?documento .
                    OPTIONAL {{ ?documento rdf:type ?tipo }}
                }}

            }}
            ORDER BY DESC(?score)
            LIMIT {retrieval_size}
        '''

        _query_chunks = f'''
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
                        ?chunk a/rdfs:subClassOf* :Rule ;
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
                VALUES ?tipoPermitido {{ :Chunk :Rule :Paragrafo }}
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
                ?search a inst:contexto ;
                        :searchTerm '{keyword}';
                        :documentResult ?result .

                ?result :value ?documentID ;   		
                        :score ?score .    		

                BIND(?chunkID AS ?chunk)

                OPTIONAL {{ ?chunk v:idChunk ?idChunk }}
                OPTIONAL {{ ?chunk v:texto   ?texto }}
                
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

            resp_rules_toon = 'breadcrumb;texto_regra\n'  

            for index,item in enumerate(bindings,1): 

                #print( '->',item.get("breadcrumb", {}).get("value", "") )       

                score     = float( item.get("score", {}).get("value", 0) )
                score     = round(score, 3)
                id_chunk  = item.get("breadcrumb", {}).get("value", "")
                regra_uri = item.get("regraURI", {}).get("value", "")
                texto     = normalize(item.get("texto", {}).get("value", ""))
                if texto=='':
                    texto = normalize(item.get("descricao", {}).get("value", "")) 

                resp_rules_toon += f'{id_chunk};{texto}\n' 
            
                knowledge_base[regra_uri] = texto
        
        else:

            error_msg = f"Graph DB ERROR ({resp_rules.status_code}): {resp_rules.text}"
            print(f"--> {error_msg}")
            print('---- rules ----') 
            print( query_class_rules )

            sys.exit()   
    
            return {
                'status': 'ERROR', 
                'response': error_msg, 
                'dataset': {}
            }         
        
        '''print( query_class_rules )
        print( '-'*100 )
        print( query_chunks )'''

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

            '''print(results)
            print('-'*100)
            print(bindings)'''            

            for index,item in enumerate(bindings,1): 

                #print( item.get("breadcrumb", {}).get("value", "") )       

                score       = float( item.get("score", {}).get("value", 0) )
                score       = round(score, 3)
                breadcrumb  = item.get("breadcrumb", {}).get("value", "")
                texto       = normalize(item.get("texto", {}).get("value", ""))
                if texto=='':
                    texto    = normalize(item.get("descricao", {}).get("value", ""))

                resp_chunks_toon += f'{breadcrumb};{score};{texto}\n' 
                knowledge_base[breadcrumb] = texto
                
        else:
            
            error_msg = f"Graph DB ERROR ({resp_chunks.status_code}): {resp_chunks.text}"
            print(f"--> {error_msg}")
            print('---- chunks ----') 
            print( query_chunks )

            sys.exit()   
    
            return {
                'status': 'ERROR', 
                'response': error_msg, 
                'dataset': {}
            } 
        
        resp_final = f'''Deontonic rules:
        {textwrap.dedent(resp_rules_toon)}
        General Context:
        {textwrap.dedent(resp_chunks_toon) }
        ''' 
        
        '''print(resp_rules_toon)
        print('-'*100)
        print(resp_chunks_toon)'''

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
