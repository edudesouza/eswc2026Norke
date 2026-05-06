
import json,ast

from langchain_openai                import ChatOpenAI, OpenAIEmbeddings
from langchain_together              import ChatTogether     
from langchain_ollama                import ChatOllama
from langchain_anthropic             import ChatAnthropic
from langchain_google_genai          import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk

from src_en.config import settings

def class_extraction(keyword,question,query_canonical,model_provider):

    if model_provider=='maritaca':
        llm = ChatMaritalk(
            model='sabia-4', 
            api_key=settings.MARITACA_API_KEY,
            temperature=0,
            max_tokens=10000,
            model_kwargs={"response_format": {"type": "json_object"}}
        )     

    if model_provider=='gpt':
        llm = ChatOpenAI(
            model='o4-mini', 
            api_key=settings.OPENAI_API_KEY
        )    

    if model_provider=='together':
        llm = ChatTogether(
            together_api_key=settings.TOGETHER_API_KEY,
            temperature=0,
            #model="ServiceNow-AI/Apriel-1.5-15b-Thinker",
            model="openai/gpt-oss-120b",
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    if model_provider=='ollama':
        llm = ChatOllama(
            #model="kimi-k2:1t-cloud",
            #model="kimi-k2-thinking:cloud",
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            model="GLM-4.7:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="gpt-oss:120b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="mistral-large-3:675b-cloud",
            #model="qwen3-next:80b-cloud",
            #model="gemma3:27b-cloud",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    print(f'--> extract classes')

    with open(settings.ontology, encoding="utf-8") as f:
        ontology_txt = f.read()

    system = (f'''
        You are a legal graph extractor in en-US.
        With this information, extract the classes present in the ontology below.

        Follow this ontology:
        {ontology_txt}        

        Example of extracting classes present in the ontology:

        1. If the text says 'I want to hold a religious service in the party room', the entities are:

        - Commonareause, for party room
        - Religiouspurpose, for a religious service
        - Partyroom, for party room
        - Resident, for I want to hold

        Return, as in the example below, only the classes, preceded by a colon, with the first letter capitalized and the other letters lowercase:
        :Commonareause :Religiouspurpose :Partyroom :Resident
    ''')

    user = (f''' 
        Analyze the user's question: {question}
        Aanalyze the rewritten and expanded question with a broader context: {keyword}
        Analyze the canonical fact that represents the main legal point to be answered: {query_canonical}
    ''')

    msg = llm.invoke([("system", system), ("user", user)])

    print('--> llm extraction OK')

    resp_json = msg.content

    print( resp_json )
   
    return resp_json

def response_judge(keyword,question,context_g,context_v,llm_g,llm_v,model_provider): 
    
    if model_provider=='maritaca':
        llm = ChatMaritalk(
            model='sabia-3.1', 
            api_key=settings.MARITACA_API_KEY,
            temperature=0.1,
            max_tokens=10000,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
    if model_provider=='gpt':
        llm = ChatOpenAI(
            model='gpt-4.1', 
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )    

    if model_provider=='together':
        llm = ChatTogether(
            together_api_key=settings.TOGETHER_API_KEY,
            temperature=0,
            #model="ServiceNow-AI/Apriel-1.5-15b-Thinker",
            model="openai/gpt-oss-120b",
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    if model_provider=='ollama':
        llm = ChatOllama(
            #model="kimi-k2:1t-cloud",
            #model="kimi-k2-thinking:cloud",
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="gpt-oss:120b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="mistral-large-3:675b-cloud",
            model="qwen3-next:80b-cloud",
            #model="gemma3:27b-cloud",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    print(f'--> create judge response, {getattr(llm, "model", None)}, temp.: {getattr(llm, "temperature", None)}')

    system = ('''Você é um assistente jurídico de condomínios no Brasil.
        Responda APENAS com base nas evidências fornecidas. 
        Se houver uma regra explícita, cite.
        Se não houver base suficiente, diga isso.
        Relacionamentos, mostre atributos que trazem o relacionamento entre os itens usados na resposta
        Saída OBRIGATÓRIA: JSON válido com EXATAMENTE três campos: 
        Nunca começar a resposta com: sim, não, claro, com certeza, negativo, positivo
        A resposta deve ter de 220 a 300 caracteres, nunca mais de 300 caracteres.
        A saida deve ser um json puro e válido, sem ``` aspas triplas ou ```json
        {"resposta": string, "trecho_chave": string,"triples": array}. Sem texto extra.'''
    )   

    user = f'''
        Você é um assistente jurídico especializado em direito condominial brasileiro.

        Questionamento ou dúvida:
        {question}

        Fatos que devem ser respondidos:
        {keyword}

        Contexto busca vetorial:
        {context_v}     

        Contexto busca knowedge graph:
        {context_g}  

        Resposta vetorial 
        {llm_g}

        Resposta knowledge graph     
        {llm_v}  

        ## 1 .CONCEITOS FUNDAMENTAIS   
        - Você possui dados para analisar as duas respostas, e se necessário criar sua própria resposta, usando os fatos apresentados.
        - As respostas *OBRIGATÓRIAMENTE* devem ter de **220 a 300 caracteres**, você deve reescrever se necessário.

        ## 2. REGRA DE POLARIDADE NORMATIVA (OBRIGATÓRIA):
        - Toda resposta deve explicitar claramente se a conduta é PERMITIDA ou PROIBIDA,
        - Utilizando linguagem categórica e definitiva.
        - É vedado o uso de linguagem condicional, procedimental ou hipotética
        - Quando houver proibição expressa no documento. 

        ## 3. MECANISMO DE CONTROLE DE RESPOSTA
        - Conte os caracteres da resposta. Se < 220 ou > 300, REESCREVA antes de retornar.     

        ## 4. FORMATO DE SAÍDA
        É obrigatório trazer o trecho-chave na resposta, conforme o exemplo abaixo.
        Retorne exclusivamente JSON sem markdown, seguindo estrutura exata especificada no exemplo abaixo:
        {{
            "resposta": "Texto da resposta fundamentada na regra (220-300 chars), não colocar o id do chunk aqui",            
            "trecho_chave": "Trecho-chave é a transcrição do documento que valiada a resposta apresentada"
        }}    
    '''

    msg = llm.invoke([("system", system), ("user", user)])

    #print(f'tokens: {msg.usage_metadata}\n' )
    print('--> judge response OK')

    try:

        texto_limpo = msg.content.replace('```json', '').replace('```', '').strip()
        resp_json   = json.loads(texto_limpo)
   
        return resp_json
    
    except Exception as e:

        return {
            "resposta": f"Erro ao processar resposta do modelo: {e}"
         }

def response_create(keyword,question,context,model_provider): 

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
            model_kwargs={"response_format": {"type": "json_object"}}
        )    

    if model_provider=='claude':
        llm = ChatAnthropic(
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
            model="claude-sonnet-4-6"
        )

    if model_provider=='ollama':
        llm = ChatOllama(  

            #model="kimi-k2-thinking:cloud",
            #model="kimi-k2.6:cloud",
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="deepseek-v4-pro:cloud",            
            #model="qwen3-next:80b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="gemma3:27b-cloud",
            
            model="gemma4:31b-cloud",            
            #model="qwen3.5:397b-cloud",
            #model="mistral-large-3:675b-cloud",
            #model="gpt-oss:120b-cloud",                 

            num_predict=1024,
            temperature=0,
            format="json",
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

    print(f'--> create llm response, {model_provider} {getattr(llm, "model", None)}, temp.: {getattr(llm, "temperature", None)}')

    system_v1 = ('''You are a legal assistant for law interpretation and explanations
        Answer ONLY based on the evidence provided.
        If there is an explicit rule, cite it.
        If there is not sufficient basis, state so.

        Relationships: show attributes that demonstrate the relationship between the items used in the answer.

        **ABSOLUTE RULES:**
        1. Never start the answer with: yes, no, of course, for sure, negative, positive
        2. The "answer" field MUST have maximum of 900 characters, prioritizing minimal and precise answers, while preserving completeness. Avoid verbosity and do not add information beyond what is necessary to answer the question.
        3. The "full_answer" field has a limit of 2000 characters
        4. OUTPUT: Pure, valid JSON, without markdown (no ```json or ```)

        **IMPORTANT**:   
        - Answer only using the provided graph evidence.
        - If the answer is not explicitly supported by the graph nodes, say: "I could not find sufficient evidence in the graph."
        - Do not use prior knowledge.
        - Preserve modal verbs exactly: shall, may, must.
        - Cite the node IDs used.
                 
        **GROUNDING**:  
        Do NOT infer a law name, topic, legal concept, or prior knowledge, if not present in current context
        The "breadcrumb" is the legal path that leads to the answer, it is the "trail" of legal concepts and rules that support the answer. It is mandatory to include the breadcrumb in the response, as it will be used to evaluate the accuracy and precision of the answer.

        **Required format**:
         {
            "answer": "Direct and grounded response (maximum of 900 characters). Cite the rule or norm. Do not include chunk IDs here.",
            "full_answer": "Detailed legal analysis (up to 2000 characters). Expand reasoning, exceptions, and implications.",
            "key_snippet": "Verbatim excerpt from the document that legally supports the answer (up to 100 characters).",
            "chunks": ["breadcrumb_1", "breadcrumb_1"],
            "percentage": ["80", "20"]
        } 
        '''
    )   

    user_v1 = f'''
        You are a legal assistant specializing in law interpretation and explanations

        Question or doubt:
        {question}

        Facts that must be answered:
        {keyword}

        {context}

        # LEGAL ANALYSIS METHODOLOGY

        ## 1. FUNDAMENTAL CONCEPTS
        - **Key passage**: Specific textual excerpt from the document that legally supports your answer
        - **Multi-chunk analysis**: Ability to synthesize information from multiple documents
        - **Hierarchy of relevance**: Score indicates source reliability
        - Never use **depends** in your answers
        - The answer MUST have **maximum of 900 characters**, you must rewrite if necessary

        ## 2. ANALYTICAL PROCESS (execute in this order)

        ### Phase 1: Understanding
        - Identify the legal nature of the question (normative, interpretative, consultative)
        - Map key concepts and legal terms Relevant
        - Consider practical implications and possible exceptions

        ### Phase 2: Investigation in the Chunks
        - Prioritize the rules and then the chunks, type: 'Rule' or 'Chunk'
        - Prioritize chunks with a _score > 0.85 as primary sources
        - From the user's question, create a premise, carefully analyze the context,
        - Seek a logical conclusion for this premise; this conclusion can be affirmative or negative
        - You may find facts and arguments in the context that justify or refute what is being asked
        - Identify complementary rules or chunks (0.60-0.85) for additional context
        - Be aware of ambiguities, as in one rule or chunk you may have wording that conflicts with another
        - How to proceed with disambiguation:
        -- Analyze the question, determining the real motivator of what is being questioned
        -- Analyze which of the candidate options best meets the requirements that answer the question
        -- Look for logical relationships between different chunks and rules:

        * General rule + exceptions
        * Right + corresponding obligation
        * Norm + penalty
        * Permission + requirements

        ### Phase 3: Legal Synthesis

        - Construct a response that integrates multiple chunks when applicable
        - Identify the most relevant excerpt (up to 100 characters)
        - Provide a clear conclusion: YES / NO / DEPENDS + conditions
        - Maintain accessible language without losing technical precision

        ## 3. REASONING PRINCIPLES

        - **Controlled Literalness**: Base yourself strictly on the text, but interpret it within a legal context
        - **Information Economy**: Prioritize quality over quantity of chunks
        - **Transparency**: Explicitly state when there is ambiguity or insufficient information
        - **Intelligent Connections**: Relate chunks that address the same topic from different angles

        ## 4. MANDATORY VALIDATIONS
        - [ ] All chunk IDs exist in the provided context
        - [ ] Percentages add up to 100% and reflect the real contribution of each chunk
        - [ ] Key passage is a verbatim quote (not a paraphrase)
        - [ ] Answer has a logical consequence based on the chunks
        - [ ] No information was invented or inferred without a textual basis
        - [ ] The answer MUST have maximum of 900 characters

        ## 5. Control Mechanism:
        - If any of the validations fail, rewrite the answer correcting the error.
        - The answer MUST have maximum of 900 characters; if the answer has fewer or more characters, discard it and generate a new one.

        ## 6. OUTPUT FORMAT
        - Your answer MUST have maximum of 900 characters, never more than 900 characters.
        - Whenever possible, use keywords in your answer. - It is mandatory to include the key segment in the response, as shown in the example below.
        - Return only JSON without markdown, following the exact structure specified in the example below:

        ## 7. Summary Mechanism
        - If the response has more than 900 characters, rewrite it to meet the *MANDATORY* character limit (maximum of 900), while maintaining the essence of the response.

        {{
            "answer": "Text of the answer based on the rule (maximum of 900 characters), do not put the chunk ID here>",
            "full_answer": "Detailed legal analysis (up to 2000 characters). Expand reasoning, exceptions, and implications.",
            "key_snippet": "<Key snippet is the transcription of the document that validates the presented answer>",
            "chunks": ["<breadcrumb_1>", "<breadcrumb_2>"],
            "percentage": ["<80>", "<20>"]
        }}        

        **IMPORTANT**:        
        GROUNDING: You can only use information to respond, present in this context
        VALIDATION: if the answer has more than 900 characters, rewrite it to meet the *MANDATORY* character limit (maximum of 900), maintaining the essence.
    '''

    system_v2 = ('''
        You are a legal assistant specialized in law interpretation and explanations.

        **ABSOLUTE RULES:**
        1. Never start the answer with: yes, no, of course, for sure, negative, positive.
        2. The "answer" field MUST have have between 200 and 900 characters, prioritizing minimal and precise answers, while preserving completeness. Avoid verbosity and do not add information beyond what is necessary to answer the question.
        3. The "full_answer" field has a limit of 2000 characters.
        4. Preserve modal verbs exactly: shall, may, must.
        5. Answer ONLY based on the evidence provided in the context.
        6. If there is an explicit rule, cite it.
        7. If there is not sufficient basis, state: "I could not find sufficient evidence in the graph."
        8. Do not use prior knowledge.
        9. OUTPUT: Pure, valid JSON, without markdown (no ```json or ```).

        **Required output format:**
        {
            "answer": "Direct and grounded response (500–900 characters). Cite the rule or norm. Do not include chunk IDs here.",
            "full_answer": "Detailed legal analysis (up to 2000 characters). Expand reasoning, exceptions, and implications.",
            "key_snippet": "Verbatim excerpt from the document that legally supports the answer (up to 100 characters).",
            "chunks": ["id_chunk_1", "id_chunk_2"],
            "percentage": ["80", "20"]
        }

        **Relationships:** 
        When applicable, show attributes that demonstrate the relationship between the items used in the answer.
        
    ''')

    user_v2 = f'''
        **Question:**
        {question}

        **Facts to be addressed:**
        {keyword}

        **Context (graph evidence):**
        {context}

        ---

        # LEGAL ANALYSIS METHODOLOGY

        ## Phase 1 — Understanding
        - Identify the legal nature of the question: normative, interpretative, or consultative.
        - Map key legal concepts and terms.
        - Consider practical implications and possible exceptions.

        ## Phase 2 — Investigation
        - Prioritize chunks of type "Rule" over type "Chunk".
        - From the question, build a premise and seek a logical conclusion — affirmative or negative.
        - Identify complementary chunks (score 0.70–0.85) for additional context.
        - When ambiguity exists:
        - Determine the real motivator behind the question.
        - Select the option that best satisfies the legal requirements.
        - Look for logical relationships:
            * General rule + exceptions
            * Right + corresponding obligation
            * Norm + penalty
            * Permission + requirements

        ## Phase 3 — Legal Synthesis
        - Integrate multiple chunks when applicable.
        - Identify the most relevant verbatim excerpt (key_snippet, up to 100 characters).
        - Construct a clear conclusion grounded strictly in the provided context.
        - Maintain accessible language without losing technical precision.

        ---

        # REASONING PRINCIPLES
        - **Controlled Literalness**: Base yourself strictly on the text, interpreted within legal context.
        - **Information Economy**: Prioritize quality over quantity of chunks.
        - **Transparency**: Explicitly state ambiguity or insufficient information.
        - **Intelligent Connections**: Relate chunks that address the same topic from different angles.
        - Never use the word **depends** in your answers.
    '''

    msg = llm.invoke([("system", system_v1), ("user", user_v1)])

    #print(f'tokens: {msg.usage_metadata}\n' )
    '''print('-'*100)
    print(user_v1)   
    print('-'*100) '''

    try:

        print('--> llm response OK')

        texto_limpo = msg.content.replace('```json', '').replace('```', '').strip()
        resp_json   = json.loads(texto_limpo) 

        return resp_json 
    
    except Exception as e:

        print('--> llm response ERRO')

        return {
            "resposta": f"Erro ao processar resposta do modelo: {e} {msg}"
         }

def ground_truth(dataset,question,keywords,query_canonical,model_provider,size):

    if model_provider=='maritaca':
        llm = ChatMaritalk(
            model='sabia-4', 
            api_key=settings.MARITACA_API_KEY,
            temperature=0.1,
            max_tokens=10000,
            model_kwargs={"response_format": {"type": "json_object"}}
        ) 
    
    if model_provider=='gpt':
        llm = ChatOpenAI(
            model='gpt-4.1', 
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )    

    if model_provider=='together':
        llm = ChatTogether(
            together_api_key=settings.TOGETHER_API_KEY,
            temperature=0,
            #model="ServiceNow-AI/Apriel-1.5-15b-Thinker",
            model="openai/gpt-oss-120b",
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    if model_provider=='ollama':
        llm = ChatOllama(
            #model="kimi-k2.5:cloud",
            #model="glm-4.7:cloud",
            #model="kimi-k2-thinking:cloud",
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",           #bom
            #model="deepseek-v3.1:671b-cloud",      #bom
            #model="gpt-oss:120b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="mistral-large-3:675b-cloud",    #bom
            #model="qwen3-next:80b-cloud",
            #model="gemma3:27b-cloud",
            model="gemma4:31b-cloud",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    print(f'--> create ground truth {size} candidates, {getattr(llm, "model", None)}, temp.:{getattr(llm, "temperature", None)}')

    reference = ''

    for item in dataset:
        reference += item

    system = '''You are a legal assistant legal assistant for law interpretation and explanations.
        You must create questions and answers based on a reference document (such as bylaws or regulations).
        Your goal is to generate realistic and useful questions for a condominium-related chatbot.
        Never start the answer with: yes, no, of course, certainly, negative, positive.

        Strictly follow the requested format and character count.
        The answer must be between 500 and 900 characters, never more than 900 characters.
        The output must be a pure and valid JSON, without triple quotes or JSON.

        LEGAL SECURITY GUIDELINES:
        1. Priority of Prohibition: If one rule says "X is prohibited" and another rule says "General meetings are allowed," the SPECIFIC PROHIBITION prevails.
        2. Don't create exceptions: If the text says "Religious events prohibited," don't respond "Allowed if only among residents," unless the text explicitly says "except if among residents."
        3. Be careful with lists: If the text prohibits "commercial, religious AND political events," this means that religious events are prohibited EVEN IF they are not commercial.
        4. In case of apparent conflict between two rules, generate a question that addresses this conflict and answer by pointing out the most severe restriction.

        REQUIRED JSON SCHEMA:

        {
            "questions_answers": [{
            "question": "Question text (max 900 chars), do not put the chunk ID here",
            "answer": "Answer text based on the rule (500-900 chars), do not put the chunk ID here",
            "context": "Legal explanation citing the conflict or the rule used (max 900 chars)",
            }]
        }
    '''
     
    user = f'''Document (reference text):
        {reference}

        Question or doubt:
        {question}

        Facts that must be answered:
        {keywords}

        Main fact to be addressed:
        {query_canonical}

        # LEGAL METHODOLOGY FOR BENCHMARK CREATION

        ## 1. FUNDAMENTAL CONCEPTS
        - Analyze the document in search of SPECIFIC RULES (prohibitions, fees, schedules).
        - If there is an explicit prohibition on the topic of the question (e.g., worship/religion), the answer MUST reflect this prohibition, even if there are general rules of coexistence.
        - Create **{size} questions and answers** based ONLY on the document/rules, and that test the limits of the rules (what is allowed and what is NOT allowed).
        - Questions should be written in a colloquial tone (like in WhatsApp conversations) and answers should faithfully reflect the content in the document/rules.

        ## 2. STYLE RULES:
        - {size} questions (or more, if there are multiple topics in the document/rules).
        - Clear, natural, and direct Portuguese.
        - Do not invent questions or answers outside the text.
        - Each answer should have **220 to 300 characters**.

        ## 3. NORMATIVE POLARITY RULE (MANDATORY):
        - Every answer must clearly state whether the conduct is ALLOWED or PROHIBITED,
        - Using categorical and definitive language.
        - The use of conditional, procedural, or hypothetical language is prohibited.
        - When there is an express prohibition in the document.

        # 4. CREATING QUESTIONS AND ANSWERS:
        - Generate {size} questions and answers (answers between 500 and 900 characters) for the question or doubt: {question}, remembering that the main fact to be addressed is: {query_canonical}

        # 5. CONTROL MECHANISM:
        - Answers must be between 500 and 900 characters long. If the answer has fewer or more characters, discard it and generate a new one.

        # 6. CREATING THE CONTEXT:
        - Generate a summary of the context used to generate the question and answer.
        - This text should contain the arguments that justify the answer, as it will be used to evaluate the accuracy and precision of the answer.

        IMPORTANT:
        Return ONLY valid JSON, strictly following the SCHEMA defined in the system instructions.
        Make sure that "questions_answers" is a LIST.   
    '''  

    msg = llm.invoke([("system", system ), ("user", user )])

    print('--> ground truth OK')

    #return msg.content

    try:

        if model_provider=='ollama':
            
            texto_limpo = msg.content.replace('```json', '').replace('```', '').strip()
            obj = ast.literal_eval(texto_limpo)  # str python-like -> dict
            resp_json = json.dumps(obj, ensure_ascii=False, indent=2)  # dict -> JSON string
           
        else:

            texto_limpo = msg.content.replace('```json', '').replace('```', '').strip()
            #resp_json   = json.loads(texto_limpo)
            resp_json   = texto_limpo
  
        return resp_json
    
    except Exception as e:

        return {
            "resposta": f"Erro ao processar resposta do modelo: {e}"
         }
