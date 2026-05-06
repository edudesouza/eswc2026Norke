
from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

from saf_eval.config import Config, LoggingConfig
from saf_eval.utils.logging import get_logger

from src_en.config import settings

from dotenv import load_dotenv
load_dotenv()

# https://github.com/dowhiledev/saf-eval

async def saf(knowledge_base,response,context,debug=False):

    print( '--> saf score ')
    
    # Initialize configuration
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        },
        retrieval_config={"top_k": 3},
        logging=LoggingConfig( 
            console=False
        )       
    )
    
    # Initialize components
    llm = OpenAILLM(
        model="gpt-4.1", 
        api_key=settings.OPENAI_API_KEY
    )
      
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
        
    # Run the evaluation
    result = await pipeline.run(response, context)

    #print( result )

    if debug==True:
        for i, evaluation in enumerate(result.evaluations):
            print(f"Fact: {evaluation.fact.text}")
            print(f"Confidence: {evaluation.confidence:.2f}")

    return result.factuality_score
