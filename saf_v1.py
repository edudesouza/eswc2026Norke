import asyncio
import os
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Load environment variables
load_dotenv()

async def main():
    
    # Initialize configuration
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        },
        retrieval_config={"top_k": 3}
    )
    
    # Initialize components
    llm = OpenAILLM(
        model="gpt-4.1", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a simple knowledge base for this example
    knowledge_base = {
        "regra": 'Proibição de utilização de qualquer espaço das áreas comuns do condomínio para eventos ou entrevistas comerciais, religiosos, profissionais, políticos ou de divulgação de produtos ou serviços',
        "chunk": '''82º - Os banheiros internos são de utilização do salão de festas e espaço gourmet, sendo que os banheiros externos servirão de apoio à área de churrasqueira e piscinas. Página 25 de 43 Regulamento Interno CONDOMÍNIO ALTO DO IPIRANGA NOUVEAUX...''',
        "regra": 'Somente uma reserva por dia por local (Salões de Festas, Espaço Gourmet ou Churrasqueira)',
        "regra": 'Banheiros internos são de utilização do salão de festas e espaço gourmet; banheiros externos servirão de apoio à área de churrasqueira e piscinas',
        "regra": '82º - Movimentação de mesas e cadeiras entre Salões de Festa e Espaço Gourmet pelo mesmo condômino, com devolução ao local original até o término do evento.',
        "regra": 'Artigo 81 - A identificação fornecida pelo CONDOMÍNIO deve estar visível no veículo, sempre que o mesmo estiver na Garagem do CONDOMÍNIO.',
        "chunk": '''CAPÍTULO VIII - DOS ESPAÇOS DE FESTAS Artigo 78º - A área dos salões de festas, espaço gourmet e churrasqueira/forno de pizza assim como os equipamentos nele contidos... destinam-se exclusivamente à realização de eventos, de caráter privado e reservado...''',
        "regra": 'A reserva terá validade de 4 horas, sendo que o morador poderá seguir na sala até a próxima reserva registrada',
        "regra": 'Capacidade de 40 pessoas por salão de festas e espaço gourmet; 30 pessoas na churrasqueira/forno de pizza',
        "chunk": '''Artigo 80º - Os interessados deverão efetuar a reserva com antecedência de pelo menos 1 (uma) semana antes da data da reunião...''',
        "chunk": '''$1º - As cadeiras, mesas, e demais móveis e utensílios dos Salões de Festa e Espaço Gourmet e Churrasqueira/Forno de Pizza não poderão ser retirados...''',
        "regra": 'Desconto de 10% no valor total da reserva se for feita a reserva dos Salão de Festas, Espaço Gourmet e Churrasqueira no mesmo dia, para o mesmo evento e pelo mesmo morador',
        "chunk": '''$5º - Somente será permitida uma reserva por dia por Local... Artigo 81º - Será cobrada uma taxa no valor de 22%...''',
        "regra": 'Mensalmente será transferido 25% do valor arrecadado com o aluguel dos espaços da cota dos Salões de Festas para a Cota Condominial Ordinária'
    }
        
    extractor   = FactExtractor(config=config, llm=llm)
    retriever   = SimpleRetriever(config=config, knowledge_base=knowledge_base)
    classifier  = FactClassifier(config=config, llm=llm)
    scorer      = FactualityScorer(config=config)
    
    # Create the evaluation pipeline
    pipeline = EvaluationPipeline(
        config=config,
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
        scorer=scorer
    )
    
    # Sample AI response to evaluate
    response = """
    Não é permitido. O regulamento interno veda expressamente a utilização de espaços comuns para eventos religiosos.
    """

    context = '''O regulamento interno do condomínio proíbe expressamente a utilização de qualquer espaço das áreas comuns para eventos ou entrevistas de caráter religioso, comercial, profissional, político ou de 
    divulgação de produtos ou serviços. Os espaços, como salão de festas, espaço gourmet e churrasqueira, destinam-se exclusivamente à realização de eventos privados e reservados promovidos pelos condôminos, sob sua 
    responsabilidade, e pequenas recepções. Portanto, não é permitido reservar o salão para cultos religiosos, mesmo que sejam promovidos por moradores, pois isso se enquadra na vedação prevista no documento.'''
    
    # Run the evaluation
    result = await pipeline.run(response, context)
    
    # Print the results
    print(f"Factuality Score: {result.factuality_score:.2f}")
    print("\nEvaluated Facts:")
    
    for i, evaluation in enumerate(result.evaluations):
        print(f"\n[{i+1}] Fact: {evaluation.fact.text}")
        print(f"    Confidence: {evaluation.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
