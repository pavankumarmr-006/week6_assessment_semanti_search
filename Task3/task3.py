
import time
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    FilterSelector
)

 
# STEP 1: EXPLAIN ANN AND HNSW 

print("ANN AND HNSW")

print("1. ANN means Approximate Nearest Neighbor.")
print("2. ANN finds similar vectors quickly.")
print("3. HNSW is an ANN search algorithm.")
print("4. HNSW uses a graph structure.")
print("5. It provides fast search with good accuracy.")

 
# STEP 2: FAISS EXPERIMENT 

print("\nFAISS EXPERIMENT")

# Create 50000 random vectors

number_of_vectors = 50000

dimension = 128

vectors = np.random.random(
    (number_of_vectors, dimension)
).astype("float32")

# Normalize vectors

faiss.normalize_L2(vectors)

# Create 100 query vectors

queries = np.random.random(
    (100, dimension)
).astype("float32")

faiss.normalize_L2(queries)

 
# STEP 3: EXACT SEARCH 

exact_index = faiss.IndexFlatIP(dimension)

exact_index.add(vectors)

start = time.time()

exact_scores, exact_results = exact_index.search(
    queries,
    10
)

exact_time = time.time() - start

print("Exact search time:", exact_time)

 
# STEP 4: HNSW SEARCH 

hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    32,
    faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efSearch = 64

hnsw_index.add(vectors)

start = time.time()

hnsw_scores, hnsw_results = hnsw_index.search(
    queries,
    10
)

hnsw_time = time.time() - start

 
# STEP 5: RECALL@10 

correct = 0

total = 0

for i in range(len(queries)):

    exact_set = set(exact_results[i])

    hnsw_set = set(hnsw_results[i])

    correct += len(exact_set.intersection(hnsw_set))

    total += 10


recall = correct / total

print("\nSEARCH RESULTS")

print("Exact search time:", exact_time)

print("HNSW search time:", hnsw_time)

print("Recall@10:", recall)

 
# STEP 6: LOAD EMBEDDING MODEL 

print("\nQDRANT VECTOR DATABASE")

model = SentenceTransformer("all-MiniLM-L6-v2")

 
# STEP 7: SAMPLE DOCUMENTS 

documents = [

    {
        "source": "leave.txt",
        "page": 1,
        "doc_type": "TXT",
        "text": "Employees get 20 days of annual leave."
    },

    {
        "source": "attendance.pdf",
        "page": 3,
        "doc_type": "PDF",
        "text": "Employees must arrive at the office on time."
    },

    {
        "source": "security.pdf",
        "page": 4,
        "doc_type": "PDF",
        "text": "Employees must protect company passwords."
    },

    {
        "source": "benefits.docx",
        "page": 2,
        "doc_type": "DOCX",
        "text": "Employees receive medical benefits."
    },

    {
        "source": "office.txt",
        "page": 5,
        "doc_type": "TXT",
        "text": "Visitors must register at reception."
    }

]

 
# STEP 8: GENERATE EMBEDDINGS 
texts = []

for document in documents:

    texts.append(document["text"])


embeddings = model.encode(texts)

print("Embedding dimensions:", embeddings.shape)

 
# STEP 9: CONNECT TO QDRANT 

client = QdrantClient(
    path="./qdrant_data"
)

collection_name = "company_documents"

# Remove old collection when running the demo again

if client.collection_exists(collection_name):

    client.delete_collection(collection_name)

 
# STEP 10: CREATE COLLECTION 

client.create_collection(

    collection_name=collection_name,

    vectors_config=VectorParams(

        size=384,

        distance=Distance.COSINE
    )
)
 
# STEP 11: INSERT DOCUMENTS 

points = []

for i in range(len(documents)):

    point = PointStruct(

        id=i + 1,

        vector=embeddings[i].tolist(),

        payload={

            "source": documents[i]["source"],

            "page": documents[i]["page"],

            "doc_type": documents[i]["doc_type"],

            "text": documents[i]["text"],

            "section": "Company Policy"
        }
    )

    points.append(point)


client.upsert(

    collection_name=collection_name,

    points=points
)


print("\nDOCUMENTS INSERTED")
 
# STEP 12: CHECK VECTOR COUNT 

count = client.count(

    collection_name=collection_name,

    exact=True
)

print("Vector count:", count.count)

 
# STEP 13: SEMANTIC SEARCH 

query = "How many annual leave days?"

query_embedding = model.encode(query)

results = client.query_points(

    collection_name=collection_name,

    query=query_embedding.tolist(),

    limit=3
).points


print("\nSEARCH RESULTS")

for result in results:

    print("Source:", result.payload["source"])

    print("Page:", result.payload["page"])

    print("Text:", result.payload["text"])

    print("Score:", result.score)

    print()

 
# STEP 14: FILTERED SEARCH 

print("\nFILTERED SEARCH")

results = client.query_points(

    collection_name=collection_name,

    query=query_embedding.tolist(),

    query_filter=Filter(

        must=[

            FieldCondition(

                key="doc_type",

                match=MatchValue(value="PDF")
            ),

            FieldCondition(

                key="page",

                range=Range(gte=3)
            )
        ]
    ),

    limit=5
).points


for result in results:

    print(result.payload["source"])

    print(result.payload["text"])

    print("Score:", result.score)

 
# STEP 15: TEST UPSERT 

print("\nUPSERT TEST")

client.upsert(

    collection_name=collection_name,

    points=points
)

count = client.count(

    collection_name=collection_name,

    exact=True
)

print("Vector count after upsert:", count.count)

 
# STEP 16: CHECK COLLECTION STATUS

print("\nCOLLECTION STATUS")

collection_info = client.get_collection(
    collection_name
)

print("Collection:", collection_name)

print("Points:", collection_info.points_count)

print("Indexed vectors:", collection_info.indexed_vectors_count)


# STEP 17: DELETE ONE SOURCE

print("\nDELETE DOCUMENTS")

client.delete(

    collection_name=collection_name,

    points_selector=FilterSelector(

        filter=Filter(

            must=[

                FieldCondition(

                    key="source",

                    match=MatchValue(value="leave.txt")
                )
            ]
        )
    )
)

count = client.count(

    collection_name=collection_name,

    exact=True
)

print("Count after deletion:", count.count)


client.close()

print("\nTASK 3 COMPLETED")
