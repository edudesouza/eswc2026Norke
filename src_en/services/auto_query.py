

import json,ast,requests

from langchain_openai                import ChatOpenAI, OpenAIEmbeddings
from langchain_together              import ChatTogether     
from langchain_ollama                import ChatOllama
from langchain_anthropic             import ChatAnthropic
from langchain_google_genai          import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk

from concurrent.futures import ThreadPoolExecutor
from requests.auth  import HTTPBasicAuth

from src_en.config     import settings
from src_en.utils.text import normalize

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

    print('-> class OK')

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

    print('-> props OK')

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

    print('-> rels OK')

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

def auto_query(model_provider,schema,user_question,extracted_classes,user_question_5w3h): 

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

    # instalar https://ollama.com/library/qwen3-coder-next
    if model_provider=='ollama':
        llm = ChatOllama(  

            #model="kimi-k2-thinking:cloud",
            model="kimi-k2.6:cloud",            # 1º
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="deepseek-v4-pro:cloud",            
            #model="qwen3-next:80b-cloud",
            #model="gemini-3-pro-preview:latest",
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
        {extracted_classes}

        ## SCHEMA ##
        {schema}

        Abaixo temos dois exemplos de queries:

        #1
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
        PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf:        <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?point ?breadcrumb ?score ?texto ?documento ?tipo
        FROM <https://omc.co/graph/5511993891773>
        WHERE {{  

            {{
                SELECT ?point 
                (MAX(?s) AS     ?score) 
                (SAMPLE(?t) AS  ?texto)
                (SAMPLE(?bc) AS ?breadcrumb) 
                WHERE {{

                    #Busca Lucene (transversal)
                    ?q a luc-index:eu_full ;
                    luc:query  'Refusal to provide biometric data fingerprint for time clock, To protect personal data and privacy, Company workplace, Not specified, Employee or data subject, By exercising data subject rights or withholding consent, employee refuse fingerprint time clock' ;
                    luc:entities ?point .

                    ?point luc:score ?s .
                    #FILTRO COM SUBCLASSES (o ponto chave)
                    ?point rdf:type/rdfs:subClassOf* :Point .                    

                    OPTIONAL {{ ?point :fullText ?t . }}
                    OPTIONAL {{ ?point :breadcrumb ?bc . }}                              

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
        ORDER BY ASC(?score)
        LIMIT 50

        #2
        PREFIX :     <https://omc.co/vocabulary/>
        PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?regraURI ?descricao ?texto ?breadcrumb ?tipo
        FROM <https://omc.co/graph/5511993891773>
        WHERE {{

            VALUES ?classe {{
                :Automatedprocessing 
                :Identificationofdatasubject 
                :Processingsensitivedata 
                :Securityofpersonaldata 
                :Controller 
                :Datasubject 
                :Rule 
                :Sensitivepersonaldata        
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
        LIMIT 50

        ## IMPORTANT ##
        Returns only the SPARQL query without any comments, or any extra info like ```sparql or ```
        Always return plain text
    '''

    msg = llm.invoke([("system", system), ("user", user)])

    print('-'*100)
    print(f'tokens: {msg.usage_metadata}\n' )

    return msg.content

#-----------------------------------------

schema              = extract_instances()
model               = 'ollama'
user_question       = 'My 15-year-old son s school requires facial recognition for entry and attendance. They say it is for security and asked for consent, but students who refuse cannot enter. The system is run by an external company that also uses the data to improve its technology. Since it is mandatory and for safety, I assume this is allowed under GDPR, correct?'
extracted_classes   = ':Entity :Controller :Processor :Datasubject :Processing :Sensitivepersonaldata :Processingsensitivedata :Consent :Givenconsent :Lawfulbasisforprocessing'
user_question_5w3h  = '''legality of mandatory facial recognition for school entry and attendance, school 
security and attendance; external company data improvement, school entry and attendance, ongoing, 
15-year-old son, student, school, external company, parent, facial recognition system, facial 
recognition legality school'''

resp = auto_query(model,schema,user_question,extracted_classes,user_question_5w3h)

print('-'*100)
print(resp)


