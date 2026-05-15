
import sys, requests, re, textwrap, ast, time

from langchain_openai                import ChatOpenAI, OpenAIEmbeddings
from langchain_together              import ChatTogether     
from langchain_ollama                import ChatOllama
from langchain_anthropic             import ChatAnthropic
from langchain_google_genai          import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk

from concurrent.futures import ThreadPoolExecutor
from requests.auth  import HTTPBasicAuth

from src_en.utils      import diff_time
from src_en.config     import settings
from src_en.utils.text import normalize

'''Graph Reasoning Distance (GRD), pensar!'''

def extract_class():

    prefixes = {
        "https://omc.co/vocabulary/": "omc:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    }

    def compact_uri(uri):
        for namespace, prefix in prefixes.items():
            if uri.startswith(namespace):
                return uri.replace(namespace, prefix, 1)
        return uri

    query='''PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?class (COUNT(DISTINCT ?s) AS ?count)
    FROM <https://omc.co/graph/5511993891773>
    WHERE {
    ?s rdf:type ?class .
    }
    GROUP BY ?class
    ORDER BY DESC(?count)'''

    url     = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

    resp = requests.post(
        url,
        data=query,
        headers=headers,
        auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME,settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
    )

    resp_toon = '''prefixes:
    omc: https://omc.co/vocabulary/
    rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#
    rdfs: http://www.w3.org/2000/01/rdf-schema#

    classes[class,count]:
    '''  

    if resp.status_code == 200:
            
        results  = resp.json()  
        bindings = results.get("results", {}).get("bindings", [])

        for index,item in enumerate(bindings,1): 

            class_uri = compact_uri(item.get("class", {}).get("value", ""))
            count     = item.get("count", {}).get("value", "0")

            resp_toon += f'  {class_uri},{count}\n'

    print('--> class OK')

    return resp_toon

def extract_property():

    prefixes = {
        "https://omc.co/vocabulary/": "omc:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    }

    def compact_uri(uri):
        for namespace, prefix in prefixes.items():
            if uri.startswith(namespace):
                return uri.replace(namespace, prefix, 1)
        return uri

    query='''SELECT ?property
       (COUNT(*) AS ?count)
       (SUM(IF(isIRI(?o) || isBlank(?o), 1, 0)) AS ?objectCount)
       (SUM(IF(isLiteral(?o), 1, 0)) AS ?literalCount)
    FROM <https://omc.co/graph/5511993891773>
    WHERE {
    ?s ?property ?o .
    }
    GROUP BY ?property
    ORDER BY DESC(?count)
    '''

    url     = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

    resp = requests.post(
        url,
        data=query,
        headers=headers,
        auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME,settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
    )

    resp_toon = '''prefixes:
    omc: https://omc.co/vocabulary/
    rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#
    rdfs: http://www.w3.org/2000/01/rdf-schema#

    properties[property,count,objectCount,literalCount,kind]:
    '''  

    if resp.status_code == 200:
            
        results  = resp.json()  
        bindings = results.get("results", {}).get("bindings", [])

        for index,item in enumerate(bindings,1): 

            property_uri  = compact_uri(item.get("property", {}).get("value", ""))
            count         = item.get("count", {}).get("value", "0")
            object_count  = int(item.get("objectCount", {}).get("value", "0"))
            literal_count = int(item.get("literalCount", {}).get("value", "0"))

            if literal_count == 0:
                kind = "object"
            elif object_count == 0:
                kind = "literal"
            else:
                kind = "mixed"

            resp_toon += f'  {property_uri},{count},{object_count},{literal_count},{kind}\n'

    print('--> props OK')

    return resp_toon

def extract_relations():

    prefixes = {
        "https://omc.co/vocabulary/": "omc:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    }

    def compact_uri(uri):
        for namespace, prefix in prefixes.items():
            if uri.startswith(namespace):
                return uri.replace(namespace, prefix, 1)
        return uri

    query='''PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?subjectClass ?property ?objectClass (COUNT(*) AS ?count)
    FROM <https://omc.co/graph/5511993891773>
    WHERE {
    ?s ?property ?o .
    ?s rdf:type ?subjectClass .
    ?o rdf:type ?objectClass .
    }
    GROUP BY ?subjectClass ?property ?objectClass
    ORDER BY DESC(?count)
    '''

    url     = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

    resp = requests.post(
        url,
        data=query,
        headers=headers,
        auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME,settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
    )

    resp_toon = '''prefixes:
    omc: https://omc.co/vocabulary/
    rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#
    rdfs: http://www.w3.org/2000/01/rdf-schema#

    relations[subjectClass,property,objectClass,count]:
    '''  

    if resp.status_code == 200:
            
        results  = resp.json()  
        bindings = results.get("results", {}).get("bindings", [])

        for index,item in enumerate(bindings,1): 

            subject_class = compact_uri(item.get("subjectClass", {}).get("value", ""))
            property_uri  = compact_uri(item.get("property", {}).get("value", ""))
            object_class  = compact_uri(item.get("objectClass", {}).get("value", ""))
            count         = item.get("count", {}).get("value", "0")

            resp_toon += f'  {subject_class},{property_uri},{object_class},{count}\n'

    print('--> rels OK')

    return resp_toon

def extract_instances():

    with ThreadPoolExecutor(max_workers=3) as executor:
        class_future    = executor.submit(extract_class)
        property_future = executor.submit(extract_property)
        relation_future = executor.submit(extract_relations)

        classes    = class_future.result()
        properties = property_future.result()
        relations  = relation_future.result()

    result = f'''schema:
    {classes}
    {properties}
    {relations}'''

    return result

def auto_query(model_provider,schema,user_question,extracted_classes,user_question_5w3h,filter): 

    inicio = time.time()  

    if model_provider=='maritaca':
        llm = ChatMaritalk(
            model='sabia-4', 
            api_key=settings.MARITACA_API_KEY,
            temperature=0.1, 
            max_tokens=1000, 
            model_kwargs={"response_format": {"type": "json_object"}}
        )     

    if model_provider=='gpt':
        llm = ChatOpenAI(
            model='gpt-4.1', 
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            #model_kwargs={"response_format": {"type": "json_object"}}
        )    

    if model_provider=='claude':
        llm = ChatAnthropic(
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
            model="claude-sonnet-4-6"
        )

    if model_provider=='together':
        llm = ChatTogether(
            together_api_key=settings.TOGETHER_API_KEY,
            temperature=0,
            #model="ServiceNow-AI/Apriel-1.5-15b-Thinker",
            model="Qwen/Qwen3-Coder-Next-FP8",
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

    if model_provider=='ollama':
        llm = ChatOllama(  

            #model="kimi-k2-thinking:cloud",
            #model="kimi-k2.6:cloud",            # 1º
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="deepseek-v4-pro:cloud",            
            #model="qwen3-next:80b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="gemma3:27b-cloud",

            #model="devstral-2:123b-cloud",
            #model="glm-5:cloud",               # 2º
            model="qwen3-coder-next:cloud",    # 3º
            #model="gemma4:31b-cloud",            
            #model="qwen3.5:397b-cloud",
            #model="mistral-large-3:675b-cloud",
            #model="gpt-oss:120b-cloud",                 

            #num_predict=1024,
            temperature=0,
            #format="json",
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

    if filter!='':
        filter = f'''## FILTER ##
        This query has a filter: {filter}
        '''

    system = '''You are a legal knowledge graph specialist, GraphDB expert, query and RDF data analyst.
        Your task is receive the schema used in the database and formulate a SPARQL query to extract normative context that
        ensure that the extractes information is aligned to the classes and properties represente in user question.

        To resolve that, you will analyse:
        1. the user original question
        2. the user question rewrited using 5W3H method
        3. the extracted classes related to the user question
        4. the database schema, including the classes, properties and relations between them.

        ## IMPORTANT ##
        - prefered classes to be use: :Point :Rule, never use :Chunk

        As a result we will deliver a valid SPARQL query
    '''
    user = f'''## USER QUESTION ##
        {user_question}

        ## 5W3H REWRITED QUESTION ##
        {user_question_5w3h}

        ## EXTRACTED CLASSES ##
        Use only this classes in your query:
        {extracted_classes}

        ## SCHEMA ##
        {schema}

        {filter}

        ## QUERY STRATEGY ##
        The retrieval strategy MUST execute two searches simultaneously and merge 
        the results using UNION:

        A structured semantic graph retrieval over ontology rules/classes/subclasses.
        A Lucene-based textual retrieval over normative passages.

        The structured retrieval MUST identify relevant :Rule nodes connected through relations such as :appliesTo and :refersTo, including subclass traversal using rdf:type/rdfs:subClassOf*.
        The Lucene retrieval MUST search textual evidence across normative fragments (:Point nodes), returning relevant passages, breadcrumbs, and textual relevance scores.
        The final result set MUST combine both semantic rule-based evidence and textual normative evidence into a unified ranked response, typically limited to the top 50 results.
        
        Below we have a template query:

        PREFIX : <https://omc.co/vocabulary/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX luc: <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index: <http://www.ontotext.com/connectors/lucene/instance#>

        CONSTRUCT {{
        ?uri rdf:type ?tipo .
        ?uri :fullText ?texto .
        ?uri :descricao ?descricao .
        ?uri :breadcrumb ?breadcrumb .
        ?uri :score ?score .
        }}
        FROM <https://omc.co/graph/5511993891773>
        WHERE {{
        {{
            VALUES ?classe {{
            :Board
            :Rule
            }}

            VALUES ?prop {{
            :appliesTo
            :refersTo
            }}

            ?uri rdf:type/rdfs:subClassOf* :Rule .
            ?uri ?prop ?obj .
            ?obj rdf:type/rdfs:subClassOf* ?classe .

            OPTIONAL {{ ?uri :descricao ?descricao . }}
            OPTIONAL {{ ?uri :fullText ?texto . }}
            OPTIONAL {{ ?uri :breadcrumb ?breadcrumb . }}

            BIND(:Rule AS ?tipo)
            BIND(100 AS ?score)
        }}
        UNION
        {{
            ?q a luc-index:eu_full ;
            luc:query "board majority rules of procedure" ;
            luc:entities ?uri .

            ?uri luc:score ?score .
            ?uri rdf:type/rdfs:subClassOf* :Point .

            OPTIONAL {{ ?uri :fullText ?texto . }}
            OPTIONAL {{ ?uri :breadcrumb ?breadcrumb . }}

            BIND(:Point AS ?tipo)
        }}
        }}
        ORDER BY DESC(?score)
        LIMIT 20

        ## IMPORTANT ##   
        1. Lucene index MUST use always: {extracted_classes}
        2. All queries MUST use CONSTRUCT (not SELECT).
        3. The response must be valid TTL (text/turtle).   
        4. Returns only the SPARQL query without any comments, or any extra info like ```sparql or ```
        5. Always return plain text

        6. Only change this part of the code, classes and properties
        VALUES ?classe {{
            :Board
            :Rule
        }}

        VALUES ?prop {{
            :appliesTo
            :refersTo
        }}

        The generated query MUST minimize the RDF output size.
        Prefer compact subgraphs.

        Use LIMIT 10-20 whenever possible.

        Avoid retrieving:
        - labels
        - metadata
        - owl:sameAs
        - unrelated predicates
        - redundant triples

        Return only the minimum triples required to answer the question.
        The final TTL response should remain under approximately 50 RDF triples whenever possible.
    '''
    try:
        msg = llm.invoke([("system", system), ("user", user)])

    except Exception as e:
        error_msg = f"LLM ERROR: {str(e)}"
        print( f"--> {error_msg}" )
        print( '-'*100 )
        print( f"--> {user}" )
        sys.exit()
    
    if( msg.content==''):
        error_msg = f"LLM ERROR, empty: {msg}"
        sys.exit()

    print('-'*100)
    print(f'tokens: {msg.usage_metadata}\n' )

    diff_time('--> auto query: ', inicio)

    return msg.content

def plain_query(model_provider,user_question,extracted_classes,user_question_5w3h,filter):

    inicio = time.time()  

    with open("src_en/ingest/grafo.ttl", encoding="utf-8") as f:
        graph = f.read()

    if model_provider=='maritaca':
        llm = ChatMaritalk(
            model='sabia-4', 
            api_key=settings.MARITACA_API_KEY,
            temperature=0.1, 
            max_tokens=1000, 
            model_kwargs={"response_format": {"type": "json_object"}}
        )     

    if model_provider=='gpt':
        llm = ChatOpenAI(
            model='gpt-4.1', 
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            #model_kwargs={"response_format": {"type": "json_object"}}
        )    

    if model_provider=='claude':
        llm = ChatAnthropic(
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
            model="claude-sonnet-4-6"
        )

    if model_provider=='together':
        llm = ChatTogether(
            together_api_key=settings.TOGETHER_API_KEY,
            temperature=0,
            #model="ServiceNow-AI/Apriel-1.5-15b-Thinker",
            model="Qwen/Qwen3-Coder-Next-FP8",
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

    if model_provider=='ollama':
        llm = ChatOllama(  

            #model="kimi-k2-thinking:cloud",
            #model="kimi-k2.6:cloud",            # 1º
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="deepseek-v4-pro:cloud",            
            #model="qwen3-next:80b-cloud",
            model="gemini-3-pro-preview:latest",
            #model="gemma3:27b-cloud",

            #model="devstral-2:123b-cloud",
            #model="glm-5:cloud",               # 2º
            #model="qwen3-coder-next:cloud",    # 3º
            #model="gemma4:31b-cloud",            
            #model="qwen3.5:397b-cloud",
            #model="mistral-large-3:675b-cloud",
            #model="gpt-oss:120b-cloud",                 

            #num_predict=1024,
            temperature=0,
            #format="json",
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

    if filter!='':
        filter = f'''## FILTER ##
        This query has a filter: {filter}
        '''

    system = '''You are a legal knowledge graph specialist, GraphDB expert, query and RDF data analyst.
        Your task is receive the schema used in the graph database to extract normative context that
        ensure that the extractes information is aligned to the classes and properties represented in user question.

        To resolve that, you will analyse:
        1. the user original question
        2. the user question rewrited using 5W3H method
        3. the extracted classes related to the user question
        4. the database schema, including the classes, properties and relations between them.

        ## IMPORTANT ##
        - prefered classes to be use: :Point :Rule, never use :Chunk

        As a result we will outpu a toon, like the example:
        uri,texto,tipo,bc,score
        https://omc.co/5511993891773/7492/Chapter_III/Article_13/Point_2,"In addition to the information referred to in paragraph 1, the controller shall, at the time when personal data are obtained, provide the data subject with the following further information necessary to ensure fair and transparent processing:",https://omc.co/vocabulary/Point,"chapter III, article 13, point 2",1.1658612251281738E1
        https://omc.co/5511993891773/7492/Chapter_III/Article_13/Point_1,"Where personal data relating to a data subject are collected from the data subject, the controller shall, at the time when personal data are obtained, provide the data subject with all of the following information:",https://omc.co/vocabulary/Point,"chapter III, article 13, point 1",1.0275516510009766E1
        https://omc.co/5511993891773/7492/Chapter_III/Article_13/Point_3,"Where the controller intends to further process the personal data for a purpose other than that for which the personal data were collected, the controller shall provide the data subject prior to that further processing with information on that other purpose and with any relevant further information as referred to in paragraph 2.",https://omc.co/vocabulary/Point,"chapter III, article 13, point 3",6.529461860656738E0

    '''

    user = f'''## USER QUESTION ##
        {user_question}

        ## 5W3H REWRITED QUESTION ##
        {user_question_5w3h}

        ## EXTRACTED CLASSES ##
        Use only this classes in your query:
        {extracted_classes}  

        ## GRAPH DATA ##
        {graph}
    '''

    try:
        msg = llm.invoke([("system", system), ("user", user)])

    except Exception as e:
        error_msg = f"LLM ERROR: {str(e)}"
        print( f"--> {error_msg}" )        
        sys.exit()
    
    if( msg.content==''):
        error_msg = f"LLM ERROR, empty: {msg}"
        sys.exit()

    print('-'*100)
    print(f'tokens: {msg.usage_metadata}\n' )

    diff_time('--> auto query: ', inicio)

    return msg.content

def _graph_search(class_rules,expantion,keyword,question,named_graph,retrieval_size):

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

        resp_rules_toon  = ''
        resp_chunks_toon  = plain_query('ollama',question,class_rules,keyword,filter)

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
        print(f"--> {error_msg}")

        sys.exit()
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }

def __graph_search(class_rules,expantion,keyword,question,named_graph,retrieval_size):

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

        if( filter!='' ):
            #query = query_chunks
            schema = extract_instances()
            query  = auto_query('ollama',schema,question,class_rules,keyword,filter)
            print( query )

        else:
            schema = extract_instances()
            query  = auto_query('ollama',schema,question,class_rules,keyword,filter)
            print( query )

        resp_chunks = requests.post(
            url,
            data=query,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )

        if resp_chunks.status_code == 200:
            
            results  = resp_chunks.json()  
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
        print(f"--> {error_msg}")

        sys.exit()
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }

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
            
            results  = resp_rules

            '''
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
        '''
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

        if( filter!='' ):
            #query = query_chunks
            schema = extract_instances()
            query  = auto_query('ollama',schema,question,class_rules,keyword,filter)
            print( query )

        else:
            schema = extract_instances()
            query  = auto_query('ollama',schema,question,class_rules,keyword,filter)
            print( query )
        
        headers = {"Content-Type": "application/sparql-query", "Accept": "text/turtle"}

        resp_chunks = requests.post(
            url,
            data=query,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )

        if resp_chunks.status_code == 200:
            
            results = resp_chunks 
            
            '''bindings = results.get("results", {}).get("bindings", [])
         
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
            '''    
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
        {textwrap.dedent(resp_rules.text)}
        General Context:
        {textwrap.dedent(resp_chunks.text) }
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
        print(f"--> {error_msg}")

        sys.exit()
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }
