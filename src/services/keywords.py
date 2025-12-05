
import langextract as lx

from src.config import settings

def keywords_create(question):
    
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

    try:

        res_ex = lx.extract(
            text_or_documents=question,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash", 
            api_key=settings.GEMINI_API_KEY,
            #model_id="gpt-4.1",                
            #api_key=settings.OPENAI_API_KEY,
            fence_output=False,
            use_schema_constraints=True,
        )
    
    except Exception as erro:
        
        print( f'ERRO: {erro}' )
        exit()
        
        return

    triples = []

    for ext in getattr(res_ex, "extractions", []):
        if ext.extraction_class == "triple":
            triples.append(ext.attributes)

    palavras_chave = ''

    for t in triples:
        palavras_chave +=f"{t.get('what')}, {t.get('why')}, {t.get('where')}, {t.get('when')}, {t.get('who')}, {t.get('how')}, {t.get('how_much')}, {t.get('how_many')},"

    return palavras_chave
