import nltk
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')

corpus = [
    'Não, a regra é clasar voc~e só pode usar um veículo por vaga',
    'O uso da vaga é retrito para moradores do condomínio, sendo que o uso deve ser feito por apenas um veículo por unidade habitacional',
]

token_text1 = nltk.word_tokenize(corpus[0].lower())
token_text2 = nltk.word_tokenize(corpus[1].lower())

# Cria um conjunto de palavras únicas (vocabulário)
vocab = set(token_text1).union(set(token_text2))

# Cria vetores de frequência
vec1 = [1 if word in token_text1 else 0 for word in vocab]
vec2 = [1 if word in token_text2 else 0 for word in vocab]

# Calcula a similaridade por cosseno
# Redimensiona os vetores para arrays 2D como esperado pelo cosine_similarity
similarity = cosine_similarity([vec1], [vec2])

print(f"Similaridade: {similarity[0][0]}")





