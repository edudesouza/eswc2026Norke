import sys,os,warnings,json,requests,re,time, textwrap

import openai
import tiktoken
import logging

from urllib.parse               import quote_plus,urlparse

from requests.auth              import HTTPBasicAuth
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_openai           import ChatOpenAI, OpenAIEmbeddings
from langchain_together         import ChatTogether     
from langchain_ollama           import ChatOllama
from langchain_anthropic        import ChatAnthropic
from langchain_google_genai     import ChatGoogleGenerativeAI

import langextract as lx

from rich                       import print

import numpy as np, torch
from scipy.special  import softmax
from rich           import print

from transformers           import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.utils     import logging as hf_logging
from sentence_transformers  import SentenceTransformer, util

from bert_score import BERTScorer
from bert_score import score

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

GEMINI_API_KEY   = os.getenv('GEMINI_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
OPENAI_API_KEY   = os.getenv('OPENAI_API_KEY')
OLLAMA           = 'http://localhost:11434/api/generate'

GRAPHDB_BASE_URL = os.getenv("GRAPHDB_BASE_URL_PROD")
GRAPHDB_USERNAME = os.getenv('GRAPHDB_USERNAME')
GRAPHDB_PASSWORD = os.getenv('GRAPHDB_PASSWORD_PROD')
repositorio      = 'omc_v1'

client = openai.OpenAI(api_key=OPENAI_API_KEY)

EMB_MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large-ma-v3" 
NLI_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

sim_model     = SentenceTransformer(EMB_MODEL_NAME)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=False)
nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
config        = AutoConfig.from_pretrained(NLI_MODEL_NAME)

def criar_resposta_v1(palavras_chave,pergunta,candidatos,modelo): 

    print('-> criar resposta')

    if modelo=='gpt':
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL",modelo), 
            temperature=0.3,
            model_kwargs={"response_format": {"type": "json_object"}}
        )    

    if modelo=='together':
        llm = ChatTogether(
            together_api_key=TOGETHER_API_KEY,
            temperature=0.3,
            model="ServiceNow-AI/Apriel-1.5-15b-Thinker",
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    if modelo=='ollama':
        llm = ChatOllama(
            model="kimi-k2:1t-cloud",
            temperature=0.3,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    system = (
        "Você é um assistente jurídico de condomínios no Brasil. "
        "Responda APENAS com base nas evidências fornecidas. "
        "Se houver uma regra explícita, cite."
        "Se não houver base suficiente, diga isso."
        "Relacionamentos, mostre atributos que trazem o relacionamento entre os itens usados na resposta"
        "Saída OBRIGATÓRIA: JSON válido com EXATAMENTE três campos: "
        "Nunca começar a resposta com: sim, não,claro,com certeza, negativo, positivo"
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

    print(f'tokens: {msg.usage_metadata}\n' )
   
    return msg.content

def buscar_grafo(palavras_chave,pergunta):

    print('\n-> buscar grafo')

    chunks_set = set()
    resp_final = resp_chunks_regras = resp_chunks_md  = ""
    pergunta   = re.sub(r'[\\/]+', ' ', pergunta) 

    query_regras = f'''
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
        PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?score ?idChunk ?texto ?regraURI ?tipo ?descricao
        FROM <https://omc.co/graph/5511993891773>
        WHERE {{
        {{
            # 1. SUBQUERY: Busca focada na REGRA (onde a descrição é boa)
            SELECT ?regra (MAX(?s) AS ?score)
            WHERE {{
            ?q a luc-index:omc_full ;
                # Usamos a query rica em palavras-chave que você forneceu
                luc:query '{palavras_chave}' ;
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
            {{ ?regra ?p ?chunkNode . }} UNION {{ ?chunkNode ?p ?regra . }}

            # O filtro é: esse vizinho é um Chunk e tem texto?
            ?chunkNode a :Chunk ;
                :texto ?textoBruto .
            
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
        LIMIT 10
    '''   

    query_chunks = f'''
        PREFIX :           <https://omc.co/vocabulary/>
        PREFIX luc:        <http://www.ontotext.com/connectors/lucene#>
        PREFIX luc-index:  <http://www.ontotext.com/connectors/lucene/instance#>
        PREFIX rdfs:       <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?idChunk ?score ?texto ?descricao ?documento ?tipo
        FROM <https://omc.co/graph/5511993891773>
        WHERE {{  
        {{
            SELECT ?chunk (MAX(?s) AS ?score) (SAMPLE(?id) AS ?idChunk) (SAMPLE(?t)  AS ?texto) (SAMPLE(?descRegra) AS ?descricao)
            WHERE {{
                ?q a luc-index:omc_full ;
                luc:query '{palavras_chave}' ;
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
            # VALUES ?tipoPermitido {{ :Chunk :Regra }}
            VALUES ?tipoPermitido {{ :Chunk }}
            ?chunk a ?tipoPermitido .
        }}
    }}
    ORDER BY DESC(?score)
    LIMIT 10

    '''

    url     = f"http://localhost:7200/repositories/{repositorio}"
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}

    #print( query_regras )

    # Buscar apenas Regras

    resp_regras = requests.post(
        url,
        data=query_regras,
        headers=headers,
        auth=HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD)  # ajuste usuário/senha
    )

    if resp_regras.status_code == 200:
        
        results   = resp_regras.json()  
        bindings = results.get("results", {}).get("bindings", [])        

        for index,item in enumerate(bindings,1):        

            score     = float( item.get("score", {}).get("value", 0) )
            score     = round(score, 3)
            id_chunk  = item.get("idChunk", {}).get("value", "")
            descricao = normalizar(item.get("descricao", {}).get("value", ""))            

            resp_chunks_regras += f'''
            id_chunk relacionado: {id_chunk} 
            texto_regra: {descricao}
            ''' 

            chunks_set.add(descricao)
    
    else:
        print("Erro:", resp_regras.status_code, resp_regras.text)
        print( query_regras )

    # Buscar apenas Chunks

    resp_chunks = requests.post(
        url,
        data=query_chunks,
        headers=headers,
        auth=HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD)  # ajuste usuário/senha
    )

    if resp_chunks.status_code == 200:
        
        results   = resp_chunks.json()  
        bindings = results.get("results", {}).get("bindings", [])

        for index,item in enumerate(bindings,1):        

            score    = float( item.get("score", {}).get("value", 0) )
            score    = round(score, 3)
            id_chunk = item.get("idChunk", {}).get("value", "")
            texto    = normalizar(item.get("texto", {}).get("value", ""))

            resp_chunks_md += f'''            
            score: {score}  
            id_chunk: {id_chunk}               
            texto: {texto}
            ''' 

            chunks_set.add(texto)
            
    else:
        print("Erro:", resp_regras.status_code, resp_regras.text)
        print( query_regras )
    
    resp_final = f'''Regras:
    {textwrap.dedent(resp_chunks_regras)}
    {textwrap.dedent(resp_chunks_md) }
    ''' 
    return {'resposta':resp_final,'dataset':chunks_set}

def criar_gt(dataset,pergunta,modelo,size):

    print(f'-> criar ground truth {size} candidatos')

    referencia = ''

    for item in dataset:
        referencia += item

    system = '''Você é um assistente jurídico especializado em condomínios brasileiros.
        Deve criar perguntas e respostas baseadas em um documento de referência (como convenções ou regulamentos).
        Seu objetivo é gerar perguntas realistas e úteis para um chatbot de dúvidas condominiais.
        Nunca começar a resposta com: sim, não,claro,com certeza, negativo, positivo
        Siga rigorosamente o formato e a contagem solicitada.'''

    user = f'''Documento (texto de referência):
        {referencia}

        Pergunta do usuário:
        {pergunta}

        Tarefa:
        1. Leia com atenção o documento / regras.
        2. Leia com atenção a pergunta do usuário.
        3. Crie **{size} perguntas e respostas** com base SOMENTE no documento / regras.
        4. As perguntas devem ser escritas em tom coloquial (como em conversas de WhatsApp) e as respostas devem refletir fielmente o conteúdo no documento / regras.
        
        Regras de estilo:
        - {size} perguntas (ou mais, se houver vários temas no documento / regras).
        - Português claro, natural e direto.
        - Não invente perguntas ou respostas fora do texto.
        - Cada resposta deve ter de **180 a 220 caracteres**.

        Execução:
        - Gere {size} perguntas e respostas para a pergunta: {pergunta}

        Contexto usado na pergunta e resposta:
        - Gere um resumo do contexto que foi usado para gerar a pergunta e resposta.
        - Este texto deve trazer os argumentos que justificam a resposta, pois será usado para avaliar a acuracidade e precisão da resposta.
          
        Retorne um objeto json chamado "perguntas_respostas" como no exemplo abaixo:
        {{
            "pergunta": "<texto curto com a pergunta, máximo de 200 caracteres, não colocar o id do chunk aqui>",            
            "resposta": "<texto curto com a resposta, extamente 200 caracteres, não colocar o id do chunk aqui>",
            "contexto":"<texto contento o contexto usado para criar a pergunta, entre 800 e 1000 caracteres>",
        }} 

    '''  

    llm = modelo
    msg = llm.invoke([("system", system), ("user", user)])

    #print( msg.content )
    print('-> Perguntas e respostas OK')

    return msg.content 

    return

def sim(referencia,candidate):

    print("-> SIM")
    
    emb_gold = sim_model.encode(referencia)
    emb_cand = sim_model.encode(candidate)
    
    sim = util.cos_sim(emb_gold, emb_cand).item()   
    
    if sim >= 0.85:
        return {"status":"EXCELENTE (aprovada)", "score":float(sim)}
    elif sim >= 0.75:
        return {"status":"BOA (aprovada)", "score":float(sim)}
    elif sim >= 0.65:
        return {"status":"RAZOÁVEL (revisar)", "score":float(sim)}
    elif sim >= 0.50:
        return {"status":"RUIM (reprovar)", "score":float(sim)}
    else:
        return {"status":"PÉSSIMA (completamente errada)", "score":float(sim)}

def nli( referencia,candidato ):

    print("-> NLI")

    model_input = nli_tokenizer(
        *([referencia],[candidato]), 
        padding=True, 
        return_tensors="pt"
    )

    with torch.no_grad():

        output  = nli_model(**model_input)
        scores  = output[0][0].detach().numpy()
        scores  = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        score_por_label = {config.id2label[i]: scores[i] for i in range(len(scores))}

        entailment      = score_por_label.get('entailment', 0)
        neutral         = score_por_label.get('neutral', 0)
        contradiction   = score_por_label.get('contradiction', 0)     

        score_final = entailment - (contradiction * 2.5) - (neutral * 0.6)
        
        return {
            "score":float(score_final),
            "entailment":float(entailment), 
            "contradiction":float(contradiction), 
            "neutral":float(neutral)
        }

def bertscore( referencia,candidato ):

    print("-> Bert score")

    scorer = BERTScorer(
        #model_type="neuralmind/bert-base-portuguese-cased", num_layers=12,
        model_type="xlm-roberta-large", num_layers=24,
        #model_type="rufimelo/Legal-BERTimbau-sts-large-ma-v3", num_layers=12,
        lang="pt", 
        rescale_with_baseline=False,    
    )

    P, R, F1 = scorer.score(candidato, referencia, verbose=False)

    
    precision = P.item()
    recall    = R.item()
    f1        = F1.item()

    return {
        "precision":float(f1),
        "recall":float(recall),
        "f1":float(f1)
    }

def avaliar_resposta_completa(entailment, neutral, contradiction, f1):

    print('-> score combinado')

    # Score de fidelidade (NLI)
    fidelidade = entailment + (neutral * 0.5)  # Tolera neutral parcialmente
    
    # Score de relevância (BERT)
    relevancia = f1
    
    # Penalidade por contradição
    penalidade = contradiction
    
    # Score final: média ponderada com penalidade
    score_final = (
        (fidelidade * 0.4) +      # 40% fidelidade
        (relevancia * 0.6) -      # 60% relevância (mais importante)
        (penalidade * 2.0)        # Penalidade forte para contradição
    )
    
    # Garantir que fica entre 0 e 1
    score_final = max(0, min(1, score_final))
    
    # Classificação
    if score_final >= 0.8:
        categoria = "EXCELENTE"
    elif score_final >= 0.65:
        categoria = "BOA"
    elif score_final >= 0.5:
        categoria = "ACEITÁVEL"
    else:
        categoria = "INADEQUADA"
    
    return {
        'score_final': score_final,
        'categoria': categoria,
        'detalhes': {
            'fidelidade': fidelidade,
            'relevancia': relevancia,
            'penalidade': penalidade
        }
    }

def normalizar(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def diff_time(legenda,inicio):

    fim = time.time()
    tpo  = fim - inicio

    print( f'{legenda}{tpo:.2f}s\n' )

    return

pergunta1  = 'posso parar duas motos na minha vaga?'
pergunta2  = 'meu carro é pequeno e vi que se eu parar a minha moto dentro da minha vaga, cabe e não atrabalha ninguem, blz?'
pergunta3  = 'oi, bom dia, eu preciso fazer uma apresentação para uns clientes e pensei em fazer no salão de festas é algo pequeno só 20 pessoas, posso?' 
pergunta4  = 'Poderia me ajudar com uma dúvida sobre o fundo de obra ... para que eu preciso pagar se não está acontecendo nenhuma obra?' 
pergunta5  = 'estou recebendo uns parentes aqui no meu apartamento e hoje está muito quente a gente pode ir para a piscina rapidinho?'
pergunta6  = 'sou médico e só tenho o domingo livre, não existe forma alguma de fazer a minha mudança no próximo domingo? o síndico não pode aprovar essa exceção, ele me falou no elevador que por ele OK'
pergunta7  = 'estou com a perna quebrada e é a segunda vez que vcs impendem meu ifood de ser entregue, eu não tenho como descer da próxima vez vou chamar a polícia!'
pergunta8  = 'estou com a perna quebrada e vcs impendem meu ifood subir, não temo como abrir uma exceção?'
pergunta9  = 'minha arquiteta sugeriu a aplicação de um sobre piso, disse que é rápido não afeta a carga e não precisa de ART, posso fazer?'
pergunta10 = 'quero fazer um churrasco mas vi que a churrasqueira está ocupada, posso fazer um churrasco com um churrasqueira portátil lá perto do jardim, vi que o regulamento não proibe, concorda?'
pergunta11 = 'vou passar 3 meses fora trabalhando em um outro estado e nesse período vou fazer um AirBnB aqui, vi o regulamento e a convenção e nenhum probe então entendo que está OK, blz?'
pergunta12 = 'roubaram minha bike dentro do condomínio, isso é um absurso, o condomínio deve me reembolsar?'
pergunta13 = 'roubaram o carro do meu filho em frente ao condomínio, quero as imagens agora e o síndico não quer me fornecer, pode isso?'
pergunta14 = 'oi, bom dia, eu preciso fazer uma demonstração de produtos para meus clientes e pensei em alugar o salão de festas, serão umas 20 pessoas, OK?'
pergunta15 = 'oi, bom dia, quero fazer um culto com os irmãos da igreja no próximo dia 10 e quero alugar o salão, OK?' 
pergunta16 = 'porque meus convidados que estão no meu aniversário não podem fumar aqui na area de fora, perto da churrasqueira'

# gpt-4.1
gpt = ChatOpenAI(
    model="gpt-4.1", 
    temperature=0.2,
    model_kwargs={"response_format": {"type": "json_object"}}
) 

# claude-3-7-sonnet-20250219 | claude-sonnet-4-5-20250929 | claude-haiku-4-5-20251001
claude = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    temperature=0.2,
    max_tokens=1024,
    timeout=None,
    max_retries=3,   
)

# gemini-3-pro-preview | gemini-2.5-pro | gemini-2.5-flash
google = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY,
    response_mime_type="application/json",
    model_kwargs={"max_output_tokens": 8192},
    request_timeout=30000,
    #se quiser garantir system como system:
    #convert_system_message_to_human=False,
)

#--------------------------------------------------------------------------------------

print( '\n--- inicio ---\n')
inicio = time.time()

if len(sys.argv) < 2:
    print("Uso: digite a perguta, exemplo pergunta1 \"nr da pergunta aqui\"")
    sys.exit(1)

chave    = sys.argv[1].strip()
pergunta = globals().get(chave)

#--------------------------------------------------------------------------------------

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
    Não traga palavras soltas, crire sempre frases
    Use no máximo 300 caracteres
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
                    "when":     "Hoje ou um dia específico (implícito)",
                    "who":      "Um morador do condomínio",
                    "how":      "Através do uso comum do espaço (seguindo regras internas)",
                    "how_much": "Não aplicável (uso comum)",
                    "how_many": "Não informado (pode influenciar em regras de uso)"
                },
            )
        ],
    )
]

# gemini-2.5-flash | gemini-3-pro-preview
res_ex = lx.extract(
    text_or_documents=pergunta,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash", 
    #api_key=os.environ["GEMINI_API_KEY"],
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

remove_chars = ['|', '\\', '/', "'", '"']

for item in remove_chars:
    palavras_chave = palavras_chave.replace(item, '')

diff_time('--> palavras chaves OK: ', inicio)

print('-'*100)
print( pergunta )
print('-'*100)
print( palavras_chave )
print('-'*100)

inicio = time.time()
grafo_md = buscar_grafo(palavras_chave,pergunta)
diff_time('--> grafo OK: ', inicio)
#print(grafo_md)

inicio = time.time()
gt = criar_gt(grafo_md['dataset'],pergunta,gpt,5)
diff_time('--> GT OK: ', inicio)
#print( gt )

inicio = time.time()
resposta   = criar_resposta_v1(palavras_chave,pergunta,grafo_md['resposta'],'ollama') 
diff_time('--> resposta OK: ', inicio)

resp_json = json.loads(resposta)

print( f'{resp_json}' )
print( f'\nLLM: {resp_json["resposta"]}\n' ) 

#print( grafo_md )

gt_dict = json.loads(gt)
inicio  = time.time()

print('--- calcular score nli e sim ---')

for item in gt_dict['perguntas_respostas']:

    print( item['resposta'] )
    
    score_sim = sim( item['resposta'], resp_json["resposta"] )      
    print( score_sim['score'] )

    score_nli = nli( item['resposta'], resp_json["resposta"] )      
    print( score_nli['score'] )

    print( '-'*100 )

diff_time('\n--> calcular score OK: ', inicio)
