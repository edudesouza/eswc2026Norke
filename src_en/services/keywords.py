
import langextract as lx

from sentence_transformers import util

from src_en.evaluation import keyword_complexity
from src_en.config import settings

from src_en.models import embedding_model

def keywords_create(question,model,api,url):

    ontology_txt=''
   
    with open(settings.ontology, encoding="utf-8") as f:
        ontology_txt = f.read()
    
    prompt = f'''
        ## 1. FUNDAMENTAL CONCEPTS
        You are an expert in Ontologies and Intent Extraction for condominium management systems.
        Your task is to analyze the user's question and extract structured metadata based on the provided Ontology.
        
        ## 2. REFERENCE ONTOLOGY
        {ontology_txt}
        
        ## 3. EXTRACTION INSTRUCTIONS (5W3H METHODOLOGY)
        Analyze carefully to identify the main item being asked about.
        For the main topics, always look for at least 2 synonyms.
        We will use 5W3H, which is a structured questioning methodology designed to organize thinking and action planning.
        Acronym    Question               Practical function
        What       What will be done?     Defines the objective or task.
        Why        Why will it be done?   Defines the purpose or justification.
        Where      Where will it be done? Defines the place or context.
        When       When will it be done?  Defines the deadline or schedule.
        Who        Who will do it?        Defines the responsible party.
        How        How will it be done?   Defines the method or process.
        How much   How much will it cost? Defines the cost or required resources.
        How many   How many resources?    Defines the quantity or scale.
        If necessary, create more than one set.
        If you do not find all 5W3H items, return only NULL in the item where you did not find the facts.
        Do not include stop words.
        Do not use double quotes or single quotes.
        Do not use slashes, pipes, or backslashes: '/','\','|'; instead, write them textually using: 'or','and'.
        Also generate a canonical query with a maximum of 12 terms, using only nouns and main verbs, without generic terms such as resident, condominium, requester.
        Extract article and chapter ONLY if they are explicitly mentioned in the user's question otherwise mark as none.
        Do NOT infer article or chapter from the law name, topic, legal concept, or prior knowledge, if not present mark as none.
        Identify if the question has writen a specific article, if note mark as none
        Identify if the question has writen a specific chapter, if note mark as none
        In cases that you find more than one article or chapter, separate them with pipe, for example: 'article 2|article 5' or 'chapter i|chapter ii'.
        Avoid duplicates.
    '''

    examples = [
        lx.data.ExampleData(
            text="I want to know until what time I can use the pool.",
            extractions=[
                lx.data.Extraction(
                    extraction_class = "triple",
                    extraction_text  = "I want to know until what time I can use the pool.",
                    attributes       = 
                    {
                        "what": "The permitted hours for using the pool",
                        "why": "To use the space within the condominium rules",
                        "where": "At the condominium pool",
                        "when": "Today or on a specific day (implied)",
                        "who": "A resident of the condominium",
                        "how": "Through common use of the space (following internal rules)",
                        "how_much": "Not applicable (common use)",
                        "how_many": "Not informed (may influence usage rules)",
                        "canonical": "pool operating hours",
                        "article": "article 2",
                        "chapter": "chapter (i,1)"
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
            model_id=model, 
            api_key=api,
            model_url=url,  
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

    #complexity = keyword_complexity(triples)
    #print( f'--> complexidade {complexity}')

    components = []
    expansion  = {}

    valid_parts = [v for t in triples for v in t.values() if str(v).upper() != 'NULL']
    keywords    = ", ".join(valid_parts)

    for t in triples:  

        components = [
            t.get('what'),
            #t.get('why'),
            #t.get('where'),
            #t.get('when'),
            #t.get('who'),
            t.get('how'),
            #t.get('how_much'),
            #t.get('how_many')
        ]

        expansion = {
            "what":t.get('what'),
            "why":t.get('why'),
            "where":t.get('where'),
            "when":t.get('when'),
            "who":t.get('who'),
            "how":t.get('how'),
            "how_much":t.get('how_much'),
            "how_many":t.get('how_many'),
            "canonical":t.get('canonical'),
            "article":t.get('article'),
            "chapter":t.get('chapter')
        }

        '''print( f' what:     {t.get('what')}' )
        print( f' why:      {t.get('why')}' )
        print( f' where:    {t.get('where')}' )
        print( f' when:     {t.get('when')}' )
        print( f' who:      {t.get('who')}' )
        print( f' how:      {t.get('how')}' )
        print( f' how_much: {t.get('how_much')}' )
        print( f' how_many: {t.get('how_many')}' )'''

    # Calcular embeddings
    embeddings = embedding_model.encode(components)

    # Matriz de similaridade
    similarities = util.cos_sim(embeddings, embeddings)

    # Análise
    avg_similarity = similarities.mean().item()
 
    return {
        "keywords":keywords,
        "complexity_score":avg_similarity,
        "query_expansion":expansion
    }
