
import json,ast

from langchain_openai                import ChatOpenAI, OpenAIEmbeddings
from langchain_together              import ChatTogether     
from langchain_ollama                import ChatOllama
from langchain_anthropic             import ChatAnthropic
from langchain_google_genai          import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk

from src.config import settings

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

    with open("src/ingest/_owl_tbox_v4.ttl", encoding="utf-8") as f:
        ontology_txt = f.read()

    system = (f'''
        Você é um extrator de grafos jurídicos em pt-BR.        
        Com essas informações faça a extração de classes presentes na ontologia abaixo.
        Siga esta ontologia: 
        {ontology_txt}
        Exemplo de extração de classes presentes na ontologia::
        1. Se o texto disser 'Quero fazer um culto religioso no salão de festas', as entidades são:
        - UsoAreaComum, para salão de festas
        - FinalidadeReligiosa, para um culto religioso
        - SalaoDeFestas, para salão de festas
        - Morador, para quero fazer
        retorne conforme o exemplo abaixo, apenas as classes, precedida por dois pontos, com a primeira letra maiúscula e as outras letras minúsculas:
        :Usoareacomum :Finalidadereligiosa :Salaodefestas :Morador
    ''')

    user = (f''' 
        Analise a pergunta do usuário: {question} 
        Analise a pergunta reescrita e expandida com um contexto mais amplo: {keyword}
        Analise o fato canônico que representa o principal ponto jurídico a ser respondido: {query_canonical}
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
            model="kimi-k2:1t-cloud",
            #model="kimi-k2-thinking:cloud",
            #model="minimax-m2:cloud",
            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="gpt-oss:120b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="mistral-large-3:675b-cloud",
            #model="qwen3-next:80b-cloud",
            #model="gemma3:27b-cloud",
            num_predict=200,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    print(f'--> create llm response, {getattr(llm, "model", None)}, temp.: {getattr(llm, "temperature", None)}')

    system = ('''Você é um assistente jurídico de condomínios no Brasil.
        Responda APENAS com base nas evidências fornecidas. 
        Se houver uma regra explícita, cite.
        Se não houver base suficiente, diga isso.
        Relacionamentos, mostre atributos que trazem o relacionamento entre os itens usados na resposta.
        
        **REGRAS ABSOLUTAS:**       
        1. Nunca começar a resposta com: sim, não, claro, com certeza, negativo, positivo
        2. O campo "resposta" deve ter *OBRIGATÓRIAMENTE* entre 220 e 300 caracteres (contando letras, espaços, números, pontuação)
        3. O campo "resposta_completa" tem limite de 2000 caracteres
        4. SAÍDA: JSON puro, válido, sem markdown (sem ```json ou ```) 
        
        **Formato obrigatório:**
        {"resposta": "texto *OBRIGATÓRIAMENTE* entre 220 e 300 caracteres aqui", "resposta_completa": "texto mais detalhado aqui", "chunks": ["id_chunk_1", "id_chunk_2"]}'''
    )   

    user = f'''
        Você é um assistente jurídico especializado em direito condominial brasileiro.

        Questionamento ou dúvida:
        {question}

        Fatos que devem ser respondidos:
        {keyword}

        Contexto:
        {context}        

        # METODOLOGIA DE ANÁLISE JURÍDICA

        ## 1. CONCEITOS FUNDAMENTAIS
        - **Trecho-chave**: Excerto textual específico do documento que fundamenta juridicamente sua resposta
        - **Análise multi-chunk**: Capacidade de sintetizar informações de múltiplos documentos
        - **Hierarquia de relevância**: Score indica confiabilidade da fonte
        - Nunca use **depende** em suas respostas
        - A resposta deve ter *OBRIGATÓRIAMENTE* entre **220 a 300 caracteres**, você deve reescrever se necessário

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
            - Identifique o trecho mais relevante (até 100 caracteres)
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
        - [ ] A resposta deve possuir *OBRIGATÓRIAMENTE* entre 220 e 300 caracteres

        ## 5. Mecanismo de controle:
        - Caso alguma das validações falhe, reescreva a resposta corrigindo o erro.
        - A resposta deve ter *OBRIGATÓRIAMENTE* 220 e 300 caracteres, caso a resposta tenha menos ou mais caracteres, descarte-a e gere uma nova.

        ## 6. FORMATO DE SAÍDA
        - Sua resposta deve ter *OBRIGATÓRIAMENTE* de 220 a 300 caracteres, nunca mais de 300 caracteres.
        - Sempre que possível use as palavras-chaves na sua resposta.
        - É obrigatório trazer o trecho-chave na resposta, conforme o exemplo abaixo.
        - Retorne exclusivamente JSON sem markdown, seguindo estrutura exata especificada no exemplo abaixo:
        
        ## 7. Mecanismo de resumo
        -  Caso a resposta tenha mais de 300 cracteres, reescreva de modo a atender o limite *OBRIGATÓRIO* de caracteres (entre 220 e 300), mantendo a essência da resposta.

        {{
            "resposta": "Texto da resposta fundamentada na regra (220-300 chars), não colocar o id do chunk aqui>",            
            "chunks": ["<1111_11_11>", "<1111_11_12>"],
            "trecho_chave": "<Trecho-chave é a transcrição do documento que valiada a resposta apresentada>",
            "percentual": ["<80>", "<20>"]
        }}  

        **IMPORTANTE**:
        VALIDAÇÃO: caso a resposta tenha mais de 300 cracteres, reescreva de modo a atender o limite *OBRIGATÓRIO* de caracteres (entre 220 e 300), mantendo a essência da resposta.
    '''

    msg = llm.invoke([("system", system), ("user", user)])

    #print(f'tokens: {msg.usage_metadata}\n' )
    print('--> llm response OK')

    try:

        texto_limpo = msg.content.replace('```json', '').replace('```', '').strip()
        resp_json   = json.loads(texto_limpo)
   
        return resp_json
    
    except Exception as e:

        return {
            "resposta": f"Erro ao processar resposta do modelo: {e}"
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
            #model="kimi-k2:1t-cloud",
            #model="glm-4.7:cloud",
            #model="kimi-k2-thinking:cloud",
            #model="minimax-m2:cloud",
            model="deepseek-v3.2:cloud",           #bom
            #model="deepseek-v3.1:671b-cloud",      #bom
            #model="gpt-oss:120b-cloud",
            #model="gemini-3-pro-preview:latest",
            #model="mistral-large-3:675b-cloud",    #bom
            #model="qwen3-next:80b-cloud",
            #model="gemma3:27b-cloud",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    print(f'--> create ground truth {size} candidates, {getattr(llm, "model", None)}, temp.:{getattr(llm, "temperature", None)}')

    reference = ''

    for item in dataset:
        reference += item

    system = '''Você é um assistente jurídico especializado em condomínios brasileiros.
        Deve criar perguntas e respostas baseadas em um documento de referência (como convenções ou regulamentos).
        Seu objetivo é gerar perguntas realistas e úteis para um chatbot de dúvidas condominiais.
        Nunca começar a resposta com: sim, não,claro,com certeza, negativo, positivo
        Siga rigorosamente o formato e a contagem solicitada.
        A resposta deve ter de 220 a 300 caracteres, nunca mais de 300 caracteres.
        A saida deve ser um json puro e válido, sem ``` aspas triplas ou ```json
        
        DIRETRIZES DE SEGURANÇA JURÍDICA:
        1. Prioridade de Proibição: Se uma regra diz "É proibido X" e outra regra diz "É permitido reuniões em geral", a PROIBIÇÃO ESPECÍFICA prevalece.
        2. Não crie exceções: Se o texto diz "Proibido eventos religiosos", não responda "Pode se for só entre moradores", a menos que o texto diga explicitamente "exceto se for entre moradores".
        3. Cuidado com listas: Se o texto proíbe "eventos comerciais, religiosos E políticos", isso significa que eventos religiosos são proibidos MESMO QUE não sejam comerciais.
        4. Em caso de conflito aparente entre duas regras, gere uma pergunta que aborde esse conflito e responda apontando a restrição mais severa.
        
        SCHEMA JSON OBRIGATÓRIO:
        {
            "perguntas_respostas": [
                {
                    "pergunta": "Texto da pergunta (max 300 chars), não colocar o id do chunk aqui",
                    "resposta": "Texto da resposta fundamentada na regra (220-300 chars), não colocar o id do chunk aqui",
                    "contexto": "Explicação jurídica citando o conflito ou a regra usada (max 500 chars)",
                }
            ]
        }
        '''
     
    user = f'''Documento (texto de referência):
        {reference}

        Questionamento ou dúvida:
        {question}

        Fatos que devem ser respondidos:
        {keywords}   

        Principal fato para ser abordado:
        {query_canonical} 

        # METODOLOGIA JURÍDICA PARA CRIALÇÃO DO BENCHMARK

        ## 1. CONCEITOS FUNDAMENTAIS  
        - Analise o documento em busca de REGRAS ESPECÍFICAS (proibições, taxas, horários).
        - Se houver uma proibição explícita para o tema da pergunta (ex: culto/religião), a resposta DEVE refletir essa proibição, mesmo que existam regras gerais de convivência.
        - Crie **{size} perguntas e respostas** com base SOMENTE no documento / regras, e que testem os limites das regras (o que pode e o que NÃO pode).
        - As perguntas devem ser escritas em tom coloquial (como em conversas de WhatsApp) e as respostas devem refletir fielmente o conteúdo no documento / regras.
        
        ## 2. REGRAS DE ESTILO:
        - {size} perguntas (ou mais, se houver vários temas no documento / regras).
        - Português claro, natural e direto.
        - Não invente perguntas ou respostas fora do texto.
        - Cada resposta deve ter de **220 a 300 caracteres**.

        ## 3. REGRA DE POLARIDADE NORMATIVA (OBRIGATÓRIA):
        - Toda resposta deve explicitar claramente se a conduta é PERMITIDA ou PROIBIDA,
        - Utilizando linguagem categórica e definitiva.
        - É vedado o uso de linguagem condicional, procedimental ou hipotética
        - Quando houver proibição expressa no documento.

        # 4. CRIAÇÃO DAS PERGUNTAS E RESPOSTAS:
        - Gere {size} perguntas e respostas (respostas entre 220 e 300 caracteres) para a pergunta ou dúvida: {question}, lembando que o principal fato para ser abordado: {query_canonical}
        
        # 5. MECANISMO DE CONTROLE:
        - As respostas devem ter entre 220 e 300 caracteres, caso a resposta tenha menos ou mais caracteres, descarte-a e gere uma nova.

        # 6. CRIAÇÃO DO CONTEXTO:
        - Gere um resumo do contexto que foi usado para gerar a pergunta e resposta.
        - Este texto deve trazer os argumentos que justificam a resposta, pois será usado para avaliar a acuracidade e precisão da resposta.

        IMPORTANTE:
        Retorne APENAS o JSON válido seguindo estritamente o SCHEMA definido nas instruções do sistema.
        Certifique-se de que "perguntas_respostas" seja uma LISTA.    
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
