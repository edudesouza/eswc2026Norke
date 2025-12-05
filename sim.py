import nltk
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

corpus = [
    'Não, a regra é clasar voc~e só pode usar um veículo por vaga',
    'O uso da vaga é retrito para moradores do condomínio, sendo que o uso deve ser feito por apenas um veículo por unidade habitacional',
]

token_text1 = nltk.word_tokenize(corpus[0].lower())
token_text2 = nltk.word_tokenize(corpus[1].lower())

# Vetorização dos textos
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Calcula a similaridade do cosseno
similaridade = cosine_similarity(X[0], X[1])

print(f"Similaridade por cosseno: {similaridade[0][0]:.4f}")





