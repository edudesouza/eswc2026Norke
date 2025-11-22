import os, re, time, warnings, requests, json, datetime as dt

from rich           import print
from elasticsearch  import Elasticsearch, helpers
from typing         import Dict, List, Any

import openai
import logging

from urllib.parse               import quote_plus,urlparse

from requests.auth              import HTTPBasicAuth
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_openai           import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms   import Ollama

from ragas          import evaluate as ev_ragas
from ragas.metrics  import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from datasets       import Dataset
from ragas.llms     import llm_factory

from deepeval           import evaluate as ev_deep
from deepeval.test_case import LLMTestCase
from deepeval.metrics   import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models    import OllamaModel
from langchain_together import ChatTogether

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

import langextract as lx

from rich           import print
from dotenv         import load_dotenv
load_dotenv()

os.environ['CONFIDENT_METRIC_LOGGING_VERBOSE'] = '0'
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "200"

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

TOGETHER_API_KEY   = os.getenv('TOGETHER_API_KEY')
OPENAI_API_KEY     = os.getenv('OPENAI_API_KEY')
openai_client      = openai.OpenAI(api_key=OPENAI_API_KEY)

ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER')
ELASTICSEARCH_PASS = os.getenv('ELASTICSEARCH_PASS')

elastic_client = Elasticsearch( 
    ELASTICSEARCH_HOST
    ,basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
    verify_certs=False
)

def criar_resposta_v1(palavras_chave,pergunta,candidatos,modelo): 

    print('-> criar resposta')

    '''llm_model = os.getenv("LLM_MODEL",modelo)
    llm = ChatOpenAI(
        model=llm_model, 
        temperature=0.3,
        model_kwargs={"response_format": {"type": "json_object"}}
    ) ''' 

    llm = Ollama(
        model=modelo,
        temperature=0.3,
        format="json"
    )

    '''llm = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        temperature=0.3,
        model=resolvedor,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )''' 

    system = (
        "Você é um assistente jurídico de condomínios no Brasil. "
        "Responda APENAS com base nas evidências fornecidas. "
        "Se houver uma regra explícita, cite."
        "Se não houver base suficiente, diga isso."
        "Relacionamentos, mostre atributos que trazem o relacionamento entre os itens usados na resposta"
        "Saída OBRIGATÓRIA: JSON válido com EXATAMENTE três campos: "
        '{"resposta": string, "resposta_completa": string, "chunks": array de strings}. Sem texto extra.'
    )   

    user_1 = f"""
        Pergunta do usuário:
        {pergunta}

        Documentos encontrados:  
        {candidatos}   

        Tarefa:
        1) Leia os documentos acima
        2) Escolha o(s) trecho(s) que melhor respondem à pergunta
        3) Cite o trecho-chave: selecione até ~200 caracteres de “texto” que respondam à pergunta        
        4) Gere uma conclusão objetiva (sim/não/depende + condição), apenas com base nos chunks de maior score. Não invente fatos.
        5) Copie EXATAMENTE o ID do chunk usado, pois será usado no JSON de resposta  
          
        Regras de estilo:
        - Português claro e direto.
        - Não invente entidades/nós. Use apenas o que aparece nos chunks.
        - Priorize os 2-3 chunks com maior score que sejam realmente pertinentes.

        Retorne um objeto json como no exemplo abaixo:
        {{
            "resposta": "<texto curto usando apenas o contexto ou mensagem de insuficiência>",            
            "chunks": ["<1111_11_11>", "<1111_11_12>"]
        }}  

        ATENÇÃO: Você DEVE usar APENAS os IDs contidos nos documentos. 
        VALIDAÇÃO: Antes de responder, confirme cada ID em "idChunk" existe na lista "IDs DISPONÍVEIS".
    """
    
    user_2 = f'''    
    Pergunta do usuário:
    {pergunta}

    Entidades relevantes: 
    {candidatos}

    Tarefa:
    1) Agrupe por idChunk (dedupe). Para cada idChunk, mantenha:
    - score (o maior entre os repetidos)
    - trecho-chave: selecione até ~200 caracteres de “texto” que respondam à pergunta (se “snippet” ajudar, use-o para localizar a parte certa no “texto”, mas remova <em>…</em>).
    2) Gere uma conclusão objetiva (sim/não/depende + condição), apenas com base nos chunks de maior score. Evite inventar fatos.
    3) Formate exatamente assim (texto livre + lista):
    Não pode parar duas motos na mesma vaga. A regra é 1 veículo por vaga. Para ter duas motos, só com segunda vaga de motocicletas (via sorteio quando houver disponibilidade).

    Evidências (chunks)
    - idChunk: 5511993891773_749_52
    Trecho-chave: “Será designada uma motocicleta por apartamento e por vaga… Uma segunda vaga…”
    - idChunk: 5511993891773_749_53
    Trecho-chave: “Mantém-se a regra de um veículo por vaga…”

    Relacionamentos, obrigatório na resposta
    - Use o elemento pred, campo value 
    - Use o elemento neighbor, campo value
    - Use o elemento tipoNeighbor, campo value
    - exemplos: "pred": {{ "type": "uri", "value": "https://omc.co/vocabulary/estaContidoEm" }}, nesta caso o relacionamento é: estaContidoEm
        - https://omc.co/vocabulary/estaContidoEm  → estaContidoEm
        - https://omc.co/vocabulary/refereSeA      → refereSeA
    - idChunk (apresente o idChunk) → pred.value            (apresente apenas o localName)
    - idChunk (apresente o idChunk) → neighbor.value        (apresente apenas o localName)
    - idChunk (apresente o idChunk) → tipoNeighbor.value    (apresente apenas o localName)

    Formato final (copie esta estrutura):
    {{
        "resposta": "<TEXTO PRINCIPAL USADO NA RESPOSTA, COM OS TRECHOS CHAVES CONTIDOS NAS EVIDÊNCIAS>",
        "resposta_completa": "<TEXTO COMPLETO (incluindo a seção 'Evidências (chunks)', 'Relacionamentos: ...' e 'Conclusão: ...')>",
        "chunks": ["<id_1>", "<id_2>"]
    }}
    '''
    
    user_3 = f'''
        Você é um assistente jurídico especializado em direito condominial brasileiro.
        
        {pergunta}

        Contexto:
        {candidatos}

        # METODOLOGIA DE ANÁLISE JURÍDICA

        ## 1. CONCEITOS FUNDAMENTAIS
        - **Trecho-chave**: Excerto textual específico do documento que fundamenta juridicamente sua resposta
        - **Análise multi-chunk**: Capacidade de sintetizar informações de múltiplos documentos
        - **Hierarquia de relevância**: Score (_score) indica confiabilidade da fonte
        - Nunca use **depende** em suas respostas

        ## 2. PROCESSO ANALÍTICO (execute nesta ordem)
        ### Fase 1: Compreensão
        - Identifique a natureza jurídica da pergunta (normativa, interpretativa, consultiva)\n
        - Mapeie conceitos-chave e termos jurídicos relevantes
        - Considere implicações práticas e exceções possíveis

        ### Fase 2: Investigação nos Chunks
        - Priorize chunks com _score > 0.85 como fontes primárias
        - a partir do questionamento do usuário crie uma premissa, analise cuidadosamente o contexto, 
        - busque uma conclusão lógica para esta premissa, esta conclusão pode ser afirmativa, ou negativa
        - você pode encontrar no contexto, fatos e argumentos que justificam ou repudiam o que está sendo perguntado
        - Identifique chunks complementares (0.70-0.85) para contexto adicional
        - Fique atento às ambiguidades, pois em um chunk você poderá termos oum escrita que conflite com outro chunk
        -- como proceder à desambiguação: 
        -- analise a pergunta determinando qual é o real motivador do que está sendo questionado
        -- analise entre as opções candidatas qula melhor atende aos requisitos que respondem ao questionamento 
        - Busque relações lógicas entre diferentes chunks:
        * Regra geral + exceções
        * Direito + obrigação correspondente
        * Norma + penalidade
        * Permissão + requisitos

        ### Fase 3: Síntese Jurídica
        - Construa resposta que integre múltiplos chunks quando aplicável
        - Identifique o trecho mais relevante (até 200 caracteres)
        - Forneça conclusão clara: SIM / NÃO / DEPENDE + condições
        - Mantenha linguagem acessível sem perder precisão técnica

        ## 3. PRINCÍPIOS DE RACIOCÍNIO
        - **Literalidade controlada**: Base-se estritamente no texto, mas interprete com contexto jurídico
        - **Economia informacional**: Priorize qualidade sobre quantidade de chunks
        - **Transparência**: Explicite quando houver ambiguidade ou informação insuficiente
        - **Conexões inteligentes**: Relacione chunks que tratam do mesmo tema sob ângulos diferentes

        ## 4. VALIDAÇÕES OBRIGATÓRIAS
        - [ ] Todos os IDs de chunks existem no contexto fornecido
        - [ ] Percentuais somam 100% e refletem contribuição real de cada chunk
        - [ ] Trecho-chave é citação literal (não paráfrase)
        - [ ] Resposta tem consequência lógica baseada nos chunks
        - [ ] Nenhuma informação foi inventada ou inferida sem base textual

        ## 6. Evidências
        - idChunk: 5511993891773_749_52
        Trecho-chave: “Será designada uma motocicleta por apartamento e por vaga… Uma segunda vaga…”
        - idChunk: 5511993891773_749_53
        Trecho-chave: “Mantém-se a regra de um veículo por vaga…”

        ##7. Relacionamentos, obrigatório na resposta
        - Use o elemento pred, campo value 
        - Use o elemento neighbor, campo value
        - Use o elemento tipoNeighbor, campo value
        - exemplos: "pred": {{ "type": "uri", "value": "https://omc.co/vocabulary/estaContidoEm" }}, nesta caso o relacionamento é: estaContidoEm
            - https://omc.co/vocabulary/estaContidoEm  → estaContidoEm
            - https://omc.co/vocabulary/refereSeA      → refereSeA
        - idChunk (apresente o idChunk) → pred.value            (apresente apenas o localName)
        - idChunk (apresente o idChunk) → neighbor.value        (apresente apenas o localName)
        - idChunk (apresente o idChunk) → tipoNeighbor.value    (apresente apenas o localName)

        ## 5. FORMATO DE SAÍDA
        Sempre que possível use as palavras-chaves na sua resposta.
        É obrigatório trazer o trecho-chave na resposta, conforme o exemplo abaixo.
        Retorne exclusivamente JSON sem markdown, seguindo estrutura exata especificada no exemplo abaixo:
        {{
            "resposta": "<texto curto usando apenas o contexto ou mensagem de insuficiência, trecho-chave: caso exista colocar aqui o trecho encontrado no documento>",            
            "chunks": ["<1111_11_11>", "<1111_11_12>"],
            "trecho_chave": "<Trecho-chave é a transcrição completa do documento que valiada a resposta apresentada>",
            "percentual": ["<80>", "<20>"],
            "evidencias": ["<idChunk → pred.value>","<idChunk → tipoNeighbor.value>"]
        }}  

        ATENÇÃO: Você DEVE usar APENAS os IDs contidos nos documentos. 
        VALIDAÇÃO: Antes de responder, confirme que cada ID em "chunks" exista no contexto.
    '''

    user_4 = f'''
        Você é um assistente jurídico especializado em direito condominial brasileiro.

        Pergunta:
        {pergunta}

        Contexto:
        {candidatos}

        Palavras chave:
        {palavras_chave}

        # METODOLOGIA DE ANÁLISE JURÍDICA

        ## 1. CONCEITOS FUNDAMENTAIS
        - **Trecho-chave**: Excerto textual específico do documento que fundamenta juridicamente sua resposta
        - **Análise multi-chunk**: Capacidade de sintetizar informações de múltiplos documentos
        - **Hierarquia de relevância**: Score (_score) indica confiabilidade da fonte
        - Nunca use **depende** em suas respostas

        ## 2. PROCESSO ANALÍTICO (execute nesta ordem)
        ### Fase 1: Compreensão
        - Identifique a natureza jurídica da pergunta (normativa, interpretativa, consultiva)\n
        - Mapeie conceitos-chave e termos jurídicos relevantes
        - Considere implicações práticas e exceções possíveis

        ### Fase 2: Investigação nos Chunks
        - Priorize as regras e depois os chunks, tipo: 'Regra' ou 'Chunk'
        - Priorize chunks com _score > 0.85 como fontes primárias
        - A partir do questionamento do usuário crie uma premissa, analise cuidadosamente o contexto, 
        - Busque uma conclusão lógica para esta premissa, esta conclusão pode ser afirmativa, ou negativa
        - você pode encontrar no contexto, fatos e argumentos que justificam ou repudiam o que está sendo perguntado
        - Identifique regras ou chunks complementares (0.70-0.85) para contexto adicional
        - Fique atento às ambiguidades, pois em uma regra ou chunk você poderá termos oum escrita que conflite com outro
        -- como proceder à desambiguação: 
        -- analise a pergunta determinando qual é o real motivador do que está sendo questionado
        -- analise entre as opções candidatas qual melhor atende aos requisitos que respondem ao questionamento 
        - Busque relações lógicas entre diferentes chunks e regras:
        * Regra geral + exceções
        * Direito + obrigação correspondente
        * Norma + penalidade
        * Permissão + requisitos

        ### Fase 3: Síntese Jurídica
        - Construa resposta que integre múltiplos chunks quando aplicável
        - Identifique o trecho mais relevante (até 200 caracteres)
        - Forneça conclusão clara: SIM / NÃO / DEPENDE + condições
        - Mantenha linguagem acessível sem perder precisão técnica

        ## 3. PRINCÍPIOS DE RACIOCÍNIO
        - **Literalidade controlada**: Base-se estritamente no texto, mas interprete com contexto jurídico
        - **Economia informacional**: Priorize qualidade sobre quantidade de chunks
        - **Transparência**: Explicite quando houver ambiguidade ou informação insuficiente
        - **Conexões inteligentes**: Relacione chunks que tratam do mesmo tema sob ângulos diferentes

        ## 4. VALIDAÇÕES OBRIGATÓRIAS
        - [ ] Todos os IDs de chunks existem no contexto fornecido
        - [ ] Percentuais somam 100% e refletem contribuição real de cada chunk
        - [ ] Trecho-chave é citação literal (não paráfrase)
        - [ ] Resposta tem consequência lógica baseada nos chunks
        - [ ] Nenhuma informação foi inventada ou inferida sem base textual

        ## 5. FORMATO DE SAÍDA
        Sempre que possível use as palavras-chaves na sua resposta.
        É obrigatório trazer o trecho-chave na resposta, conforme o exemplo abaixo.
        Retorne exclusivamente JSON sem markdown, seguindo estrutura exata especificada no exemplo abaixo:
        {{
            "resposta": "<texto curto usando apenas o contexto ou mensagem de insuficiência, trecho-chave: caso exista colocar aqui o trecho encontrado no documento>",            
            "chunks": ["<1111_11_11>", "<1111_11_12>"],
            "trecho_chave": "<Trecho-chave é a transcrição do documento que valiada a resposta apresentada>",
            "percentual": ["<80>", "<20>"]
        }}  

        ATENÇÃO: Você DEVE usar APENAS os IDs contidos nos documentos. 
        VALIDAÇÃO: Antes de responder, confirme que cada ID em "chunks" exista no contexto.
    '''

    msg = llm.invoke([("system", system), ("user", user_4)])
    
    #return msg.content
  
    return msg   
  
def buscar_grafo(palavras_chave,pergunta,modelo):

    print('\n-> buscar grafo')

    pergunta       = re.sub(r'[\\/]+', ' ', pergunta)
    palavras_chave = re.sub(r'[\\/]+', ' ', palavras_chave)

    GRAPHDB_USERNAME = os.getenv('GRAPHDB_USERNAME')
    GRAPHDB_PASSWORD = os.getenv('GRAPHDB_PASSWORD')
    repositorio = 'omc_v1'

    # lucene basico
    query_luc = f'''
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>

        SELECT ?idChunk ?texto ?score
        WHERE {{
        
        # Busca no índice Lucene
        ?q a luc-index:omc_full_chunk ;
            luc:query "({pergunta})" ;
            luc:entities ?chunk .

        # Score do Lucene
        ?chunk luc:score ?score .

        # Propriedades do Chunk
        ?chunk a :Chunk ;
                :idChunk ?idChunk ;
                :texto   ?texto .
        }}

        ORDER BY DESC(?score)
        LIMIT 10
    '''        
    
    # lucene com reasoning
    query_res = f'''
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
        PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?idChunk ?score ?texto ?pred ?neighbor ?tipoNeighbor 
        (GROUP_CONCAT(DISTINCT ?superTipoNeighbor; separator=", ") AS ?superTipos)
        WHERE {{
            GRAPH <https://omc.co/graph/5511993891773> {{
                {{
                SELECT ?chunk (MAX(?s) AS ?score) (SAMPLE(?id) AS ?idChunk) (SAMPLE(?t)  AS ?texto)
                    WHERE {{
                        ?q a luc-index:omc_full_chunk ;
                        luc:query "{pergunta}" ;
                        luc:entities ?chunk .
                        ?chunk luc:score ?s .

                        OPTIONAL {{ ?chunk :idChunk ?id }}
                        OPTIONAL {{ ?chunk :texto ?t . FILTER(LANG(?t) = "" || LANGMATCHES(LANG(?t), "pt")) }}
                    }}
                GROUP BY ?chunk
                ORDER BY DESC(?score)
                LIMIT 10
                }}

                # limite de predicados relevantes para evitar explosão
                VALUES ?pred {{ :estaContidoEm :aplicaA :possuiRelacionamento :mencionaEntidade :relacionamento :refereSeA }}

                OPTIONAL {{
                ?chunk ?pred ?neighbor .
                OPTIONAL {{ ?neighbor a ?tipoNeighbor }}
                    OPTIONAL {{
                        ?tipoNeighbor rdfs:subClassOf* ?superTipoNeighbor . FILTER(?superTipoNeighbor != ?tipoNeighbor)
                    }}
                }}

                FILTER EXISTS {{ ?chunk a :Chunk }}
            }}
        }}
        GROUP BY ?idChunk ?score ?texto ?pred ?neighbor ?tipoNeighbor
        ORDER BY DESC(?score)
    '''
    
    query_1 = f'''
        PREFIX v:          <https://omc.co/vocabulary/>
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
        PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?score ?idChunk ?tipo ?entidade ?descricao ?texto ?documento
        FROM <https://omc.co/graph/5511993891773>
        WHERE {{
            ?search a luc-index:omc_full ;
                luc:query '{palavras_chave}' ;            
                luc:entities ?entidade .
            
            ?entidade luc:score ?score ;    
            a ?tipo .   
            
            # Regra - pega descrição e documento relacionado
            OPTIONAL {{
                ?entidade a/rdfs:subClassOf* :Regra ;
                    :descricao ?descricao .
                OPTIONAL {{ ?documento :refereSeA ?entidade . }}
            }}
            
            # Chunk - pega texto, idChunk e documento que o contém
            OPTIONAL {{
                ?entidade a/rdfs:subClassOf* :Chunk ;
                    :texto ?texto ;
                    :idChunk ?idChunk .
                OPTIONAL {{ ?entidade :estaContidoEm ?documento . }}
            }}
        }}
        ORDER BY DESC(?score)
        LIMIT 90
    '''

    url     = f"http://localhost:7200/repositories/{repositorio}"
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

    resp = requests.post(
        url,
        data=query_1,
        headers=headers,
        auth=HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD)  # ajuste usuário/senha
    )

    if resp.status_code == 200:
        
        results   = resp.json()  
        bindings = results.get("results", {}).get("bindings", [])

        result_json = []

        for index,item in enumerate(bindings,1):
            
            tipo = item.get("tipo", {}).get("value", "").rstrip("/").split("/")[-1]
            
            if( tipo=='Regra' ):
                score = float( item.get("score", {}).get("value", 0) )
                score = round(score, 3)
                #print( index, score, item.get("descricao", {}).get("value", "") )
                #print( '-'*100 )
            else:
                score = float( item.get("score", {}).get("value", 0) )
                score = round(score, 3)

            result_json.append({
                "score":            score,
                "id_chunk":         item.get("idChunk", {}).get("value", ""),
                "tipo":             tipo,                
                "descricao_regra":  item.get("descricao", {}).get("value", ""),
                "texto_chunk":      item.get("texto", {}).get("value", ""),
                "entidade":         item.get("entidade", {}).get("value", ""),
                "documento":        item.get("documento", {}).get("value", ""),
            })  

        #print( result_json ) 

        try:
            resposta = criar_resposta_v1(palavras_chave,pergunta,result_json,modelo) 
        except Exception as erro:
            print( f'ERRO: {erro}')
            
        return {"resposta":resposta,"contexto":result_json}

    else:
        print("Erro:", resp.status_code, resp.text)
        print( query_1 )

def criar_palavra_chave(pergunta):
    
    prompt = '''
        Analise com atenção para obter o principal item questionado.
        Para os principais temas sempre busque pelo menos 2 sinomimos.  
        Vamos usar o 5W3H que é uma metodologia estruturada de questionamento, voltada para organizar o pensamento e o planejamento de ações.
        Sigla	    Pergunta	        Função prática
        What	    O que será feito?	Define o objetivo ou tarefa.
        Why	Por     que será feito?	    Define o propósito ou justificativa.
        Where	    Onde será feito?	Define o local ou contexto.
        When	    Quando será feito?	Define o prazo ou cronograma.
        Who	        Quem fará?	        Define o responsável.
        How	        Como será feito?	Define o método ou processo.
        How much	Quanto custará?	    Define o custo ou recursos necessários.
        How many	Quantos recursos?	Define a quantidade ou escala.
        Caso necessário, crie mais de um conjunto
        Não traga stop words
        Não use aspas duplas, nem aspas simples
        Não use barras, pipes ou barra invertida: '/','\','|', ao invés seja textual, use: 'ou','e'
        Evite duplicatas
    '''

    examples = [
        lx.data.ExampleData(
            text="Quero saber até que horas posso usar a piscina",
            extractions=[
                lx.data.Extraction(
                    extraction_class = "triple",
                    extraction_text  = "Quero saber até que horas posso usar a piscina",
                    attributes       = 
                    {
                        "what":     "O horário permitido de uso da piscina", 
                        "why":      "Para utilizar o espaço comum dentro das regras do condomínio", 
                        "where":    "Na piscina do condomínio",
                        "when":     "Hoje / em um dia específico (implícito)",
                        "who":      "Um morador do condomínio",
                        "how":      "Através do uso comum do espaço (seguindo regras internas)",
                        "how_much": "Não aplicável (uso comum)",
                        "how_many": "Não informado (pode influenciar em regras de uso)"
                    },
                )
            ],
        )
    ]

    res_ex = lx.extract(
        text_or_documents=pergunta,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-flash", 
        api_key=os.environ["GEMINI_API_KEY"],
        #model_id="gpt-4.1",                
        #api_key=os.environ["OPENAI_API_KEY"],
        fence_output=False,
        use_schema_constraints=True,
    )

    triples = []

    for ext in getattr(res_ex, "extractions", []):
        if ext.extraction_class == "triple":
            triples.append(ext.attributes)

    palavras_chave = ''

    for t in triples:
        palavras_chave +=f"{t.get('what')}, {t.get('why')}, {t.get('where')}, {t.get('when')}, {t.get('who')}, {t.get('how')}, {t.get('how_much')}, {t.get('how_many')},"

    return palavras_chave

def atualizar_elastic(id,resposta,relevancy,faithfulness,versao,avaliador):

    status = 'OK'

    texto = (resposta or "").strip()

    '''body = {
        "doc": {            
            "avaliacao":[
                    {
                    "versao":versao,
                    "avaliador":avaliador,
                    "grafo": texto,
                    "relevancia":relevancy,
                    "confiabilidade":faithfulness,
                }
            ]
        },
        "doc_as_upsert": True
    }'''

    body = body = {
        "script": {
            "lang": "painless",
            "source": """
                if (ctx._source.containsKey('avaliacoes') == false || ctx._source.avaliacoes == null) {
                    ctx._source.avaliacoes = new ArrayList();
                }
                ctx._source.avaliacoes.add(params.nova_avaliacao);
            """,
            "params": {
                "nova_avaliacao": {
                    "versao": versao,
                    "avaliador": avaliador,
                    "grafo": texto,
                    "relevancia": relevancy,
                    "confiabilidade": faithfulness
                }
            }
        },
        "upsert": {
            "avaliacoes": [
                {
                    "versao": versao,
                    "avaliador": avaliador,
                    "grafo": texto,
                    "relevancia": relevancy,
                    "confiabilidade": faithfulness
                }
            ]
        }
    }

    res = elastic_client.update(index="perguntas", id=id, body=body)
 
    return res

def diff_time(legenda,inicio):

    fim = time.time()
    tpo  = fim - inicio

    print( f'{legenda}{tpo:.2f}s\n' )

    return

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def build_retrieval_context(items, top_k=100, max_chars=1500, prefix_ids=True):
    
    # ordena por score desc (se houver)
    items_sorted = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    seen = set()
    out = []

    for it in items_sorted:
        # escolha do texto: Chunk -> texto_chunk; Regra -> descricao_regra
        raw = it.get("texto_chunk") or it.get("descricao_regra") or ""
        txt = normalize_ws(raw)
        if not txt:
            continue

        # opcional: prefixar id_chunk p/ rastreabilidade
        if prefix_ids:
            ident = it.get("id_chunk") or it.get("entidade") or ""
            if ident:
                txt = f"[{ident}] {txt}"

        # corta muito longo (evita estourar prompt)
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "..."

        # dedup
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(txt)
        if len(out) >= top_k:
            break

    return out

#-------------------------------------------------------

query = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","model","chunks"],
    "query"     : {"match_all":{}}, 
    "size"      : 1500
}

query_unico = {
    "_source"   : ["file_url", "id_usuario", "id_externo","capitulo","tema_capitulo","pergunta","resposta","contexto","chunks","avaliacoes","score_grafo"],
    "query": {
        "ids": {
            "values": ["cumbb5oB89dtCZp8OR-z"]
        }
    },
    "size": 1
}

resp = elastic_client.search(index="perguntas", body=query_unico)

total = len(resp["hits"]["hits"])

for index, item in enumerate(resp["hits"]["hits"],start=1):     

    id       = item['_id']
    pergunta = item['_source']['pergunta']
    resposta = item['_source']['resposta']
    contexto = item['_source']['contexto']
    
    try:
        grafo = item['_source']['grafo']
    except Exception as erro:
        grafo = ''

    print(f"\n{id} - {pergunta}\n\nGT: {resposta}")

    inicio = time.time()

    '''
    qwen3:4b                não retornou 
    llama3.2:3b             302s
    gemma3:1b               94s
    gemma3:latest           336s
    gpt-oss:20b             não retornou

    kimi-k2:1t-cloud        muito bom e rápido
    minimax-m2:cloud        muito bom e rápido

    ServiceNow-AI/Apriel-1.5-15b-Thinker
    meta-llama/Llama-3.2-3B-Instruct-Turbo
    Qwen/Qwen3-Next-80B-A3B-Thinking
    Qwen/Qwen3-Next-80B-A3B-Instruct
    
    X google/gemma-3n-E4B-it
    '''

    resolvedor = "kimi-k2:1t-cloud"
    avaliador  = "kimi-k2:1t-cloud"

    print( f'\n-> resolvedor: {resolvedor}' )
    print( f'-> avaliador: {avaliador}' )

    palavras_chave  = criar_palavra_chave(pergunta)
    grafo           = buscar_grafo(palavras_chave,pergunta,resolvedor)   

    try:     
        grafo_json = json.loads(grafo["resposta"])   
        print( f"\nGrafo: {grafo_json["resposta"]}" )
        print("-"*100)
        diff_time( '-> Grafo:',inicio )
    except Exception as erro:
        print( f'ERRO: {erro}' )

    #break

    inicio = time.time() 
    
    '''print('-> RAGAS\n')
    llm = ChatOpenAI(model="gpt-4.1")

    dataset = Dataset.from_list([
        {
            "question"      : pergunta,
            "answer"        : grafo_json["resposta"],
            "ground_truth"  : resposta,
            "contexts"      : [contexto]
        }
    ])

    result = ev_ragas(
        dataset,   
        llm=llm,    
        run_config=None,
        show_progress=False,        
        metrics=[
            faithfulness,          # resposta é fiel ao(s) contexto(s)?
            answer_relevancy,      # resposta é relevante à pergunta?
            context_precision,     # quanto do contexto citado é realmente útil?
            context_recall,        # os contextos cobrem o que a resposta usa?
            answer_correctness     # proximidade com a ground truth
        ]
    )

    print( result )

    print("-"*100)
    diff_time('-> RAGAS: ',inicio)
    inicio = time.time()'''

    '''print('-> Deep Eval 1\n')

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1",
        include_reason=True,
        verbose_mode=1
    )

    tc = LLMTestCase(        
        input           = pergunta,
        actual_output   = grafo_json["resposta"],
        expected_output = resposta,
        context         = [contexto]
    )

    result = ev_deep(       
        test_cases=[tc],
        metrics=[metric],
    )

    print(result)

    print("-"*100)
    diff_time('-> DeepEval: ',inicio)

    inicio = time.time()
    print('-> Deep Eval 2\n')'''

    #precision = ContextualPrecisionMetric()
    #recall    = ContextualRecallMetric()
    #relevancy = AnswerRelevancyMetric()

    model = OllamaModel(
        model = avaliador,
        base_url = "http://localhost:11434",
        temperature=0
    )

    '''model = ChatTogether(
        together_api_key=TOGETHER_API_KEY,
        temperature=0,
        model=avaliador,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    ) '''   

    answer_relevancy = AnswerRelevancyMetric(model=model,include_reason=True)
    faithfulness     = FaithfulnessMetric(model=model,include_reason=True)

    retrieval_ctx = build_retrieval_context(grafo["contexto"], top_k=20)        

    test_case = LLMTestCase(
        input             = pergunta,
        actual_output     = grafo_json["resposta"],
        expected_output   = resposta,
        retrieval_context = retrieval_ctx,      
        context           = [normalize_ws(contexto)]
    )

    try:
        answer_relevancy.measure(test_case)
        print("- Relevancia: ", answer_relevancy.score)
        print("- Reason: ", answer_relevancy.reason)
        print('-'*100)
    except Exception as erro:
        print( f'ERRO relevancia: {erro}' )

    try:
        faithfulness.measure(test_case)
        print("- Confiabilidade: ", faithfulness.score)
        print("- Reason: ", faithfulness.reason)
        print('-'*100)
    except Exception as erro:
        print( f'ERRO confiabilidade: {erro}' )

    diff_time( '-> DeepEval: ',inicio )
    print("-"*100)       

    resp_elastic = atualizar_elastic(
        id, 
        grafo_json["resposta"],
        answer_relevancy.score,
        faithfulness.score,'v2',
        avaliador
    )

    try:   
        print( f'-> Elastic: {resp_elastic['result']} {index} de {total}' )
    except Exception as erro:
        print( f'ERRO: {resp_elastic}' )  

    inicio = 0

    time.sleep(3)

print('\n--- fim ---') 