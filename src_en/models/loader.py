
import torch
import warnings, logging

from sentence_transformers import SentenceTransformer
from transformers          import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers.utils    import logging as hf_logging

from src_en.config import settings

hf_logging.set_verbosity_error()
error1 = logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
error2 = logging.getLogger("transformers").setLevel(logging.ERROR)
error3 = logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

if error1 or error2 or error3:
    print( f'error 1: {error1}\nerror 2: {error2}\nerror 3: {error3} ' )

print("\n *** INICIANDO CARREGAMENTO DOS MODELOS US-EN***")

embedding_model = SentenceTransformer(settings.EMB_MODEL_NAME)
nli_tokenizer   = AutoTokenizer.from_pretrained(settings.NLI_MODEL_NAME, use_fast=False)
nli_model       = AutoModelForSequenceClassification.from_pretrained(settings.NLI_MODEL_NAME)
nli_config      = AutoConfig.from_pretrained(settings.NLI_MODEL_NAME)
