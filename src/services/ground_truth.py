
def criar_gt(dataset,question,model,size):

    print(f'-> create ground truth {size} candidates')

    reference = ''

    for item in dataset:
        reference += item

    system = '''Você é um assistente jurídico especializado em condomínios brasileiros.
        Deve criar perguntas e respostas baseadas em um documento de referência (como convenções ou regulamentos).
        Seu objetivo é gerar perguntas realistas e úteis para um chatbot de dúvidas condominiais.
        Nunca começar a resposta com: sim, não,claro,com certeza, negativo, positivo
        Siga rigorosamente o formato e a contagem solicitada.'''

    user = f'''Documento (texto de referência):
        {reference}

        Pergunta do usuário:
        {question}

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
        - Gere {size} perguntas e respostas para a pergunta: {question}

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

    llm = model
    msg = llm.invoke([("system", system), ("user", user)])

    #print( msg.content )
    print('-> ground truth OK')

    return msg.content
