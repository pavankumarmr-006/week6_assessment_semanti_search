
import time
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
from openai import OpenAI


# ---------------------------------
# STEP 1: CREATE 10 SENTENCES
# ---------------------------------

sentences = [
    "I love learning Python programming",
    "Python is a popular coding language",
    "Machine learning is a part of AI",
    "Artificial intelligence is growing fast",
    "I love eating pizza",
    "Pizza is my favourite food",
    "The weather is very nice today",
    "It is raining outside",
    "I enjoy playing football",
    "Football is a popular sport"
]


# ---------------------------------
# STEP 2: LOCAL EMBEDDING MODEL
# ---------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

start = time.time()

embeddings = model.encode(sentences)

local_time = time.time() - start

print("\nLOCAL EMBEDDINGS")

print("Number of sentences:", len(embeddings))
print("Embedding dimensions:", embeddings.shape)
print("Time taken:", local_time)


# ---------------------------------
# STEP 3: API EMBEDDINGS
# ---------------------------------

print("\nAPI EMBEDDINGS")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:

    client = OpenAI(api_key=api_key)

    start = time.time()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences
    )

    api_embeddings = [
        item.embedding for item in response.data
    ]

    api_time = time.time() - start

    print("Number of sentences:", len(api_embeddings))
    print("Embedding dimensions:", len(api_embeddings[0]))
    print("Time taken:", api_time)

else:

    print("OPENAI_API_KEY not found. Skipping API test.")


# ---------------------------------
# STEP 4: OWN COSINE SIMILARITY
# ---------------------------------

def cosine_similarity(a, b):

    dot_product = 0

    for i in range(len(a)):
        dot_product += a[i] * b[i]

    magnitude_a = 0
    magnitude_b = 0

    for i in range(len(a)):
        magnitude_a += a[i] ** 2
        magnitude_b += b[i] ** 2

    magnitude_a = magnitude_a ** 0.5
    magnitude_b = magnitude_b ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    result = dot_product / (magnitude_a * magnitude_b)

    return result


# ---------------------------------
# STEP 5: TEST COSINE SIMILARITY
# ---------------------------------

print("\nCOSINE SIMILARITY TESTS")

print(
    cosine_similarity([1, 2, 3], [2, 4, 6])
)

print(
    cosine_similarity([1, 0], [0, 1])
)

print(
    cosine_similarity([1, 0], [-1, 0])
)


# Compare with sklearn

print("\nSKLEARN COMPARISON")

print(
    sklearn_cosine(
        [[1, 2, 3]],
        [[2, 4, 6]]
    )[0][0]
)


# ---------------------------------
# STEP 6: SIMILARITY MATRIX
# ---------------------------------

similarity_matrix = sklearn_cosine(embeddings)

print("\nSIMILARITY MATRIX")

print(np.round(similarity_matrix, 2))


# ---------------------------------
# STEP 7: TOP 3 SIMILAR PAIRS
# ---------------------------------

pairs = []

for i in range(len(sentences)):

    for j in range(i + 1, len(sentences)):

        score = similarity_matrix[i][j]

        pairs.append(
            (score, sentences[i], sentences[j])
        )

pairs.sort(reverse=True)

print("\nTOP 3 SIMILAR PAIRS")

for score, sentence1, sentence2 in pairs[:3]:

    print("\nSentence 1:", sentence1)
    print("Sentence 2:", sentence2)
    print("Similarity:", round(score, 3))


# ---------------------------------
# STEP 8: NO COMMON WORDS EXAMPLE
# ---------------------------------

sentence1 = "The physician treated the patient"

sentence2 = "A doctor helped someone who was ill"

word1 = set(sentence1.lower().split())
word2 = set(sentence2.lower().split())

common_words = word1.intersection(word2)

embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

score = cosine_similarity(
    embedding1[0],
    embedding2[0]
)

print("\nNO COMMON WORDS EXAMPLE")

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Common words:", common_words)
print("Semantic similarity:", round(score, 3))


# ---------------------------------
# STEP 9: SEMANTIC SEARCH
# ---------------------------------

queries = [
    "I want to learn coding",
    "What food do you like?",
    "Tell me about sports"
]

print("\nSEMANTIC SEARCH")

for query in queries:

    query_embedding = model.encode([query])

    scores = sklearn_cosine(
        query_embedding,
        embeddings
    )[0]

    best_index = np.argmax(scores)

    print("\nQuery:", query)

    print("Best result:", sentences[best_index])

    print("Score:", round(scores[best_index], 3))


# ---------------------------------
# STEP 10: DUPLICATE DETECTION
# ---------------------------------

print("\nDUPLICATE DETECTION")

threshold = 0.70

for i in range(len(sentences)):

    for j in range(i + 1, len(sentences)):

        score = similarity_matrix[i][j]

        if score >= threshold:

            print("Possible duplicate:")
            print(sentences[i])
            print(sentences[j])
            print("Score:", round(score, 3))


# ---------------------------------
# STEP 11: K-MEANS CLUSTERING
# ---------------------------------

print("\nK-MEANS CLUSTERING")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(embeddings)

for cluster_number in range(3):

    print("\nCluster:", cluster_number)

    for i in range(len(sentences)):

        if labels[i] == cluster_number:

            print(sentences[i])


# Add meaningful names after inspecting the results.

print("\nCluster names can be assigned after checking the sentences.")
