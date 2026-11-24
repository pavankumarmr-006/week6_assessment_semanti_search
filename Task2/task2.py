
import os
import uuid
import fitz

from datetime import date
from docx import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

 
# STEP 1: READ TXT DOCUMENTS 

folder = "documents"

documents = []

for filename in os.listdir(folder):

    path = os.path.join(folder, filename)

    text = ""

    # Read TXT files
    if filename.endswith(".txt"):

        with open(path, "r", encoding="utf-8") as file:

            text = file.read()

    # Read PDF files
    elif filename.endswith(".pdf"):

        pdf = fitz.open(path)

        for page in pdf:

            text += page.get_text() + "\n"

    # Read Word files
    elif filename.endswith(".docx"):

        word = Document(path)

        for paragraph in word.paragraphs:

            text += paragraph.text + "\n"

    if text.strip() != "":

        documents.append({

            "source": filename,

            "text": text
        })


print("Total documents:", len(documents))

 
# STEP 2: FIXED SIZE CHUNKING 

def fixed_chunking(text, chunk_size=512, overlap=50):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start = end - overlap

    return chunks

 
# STEP 3: SENTENCE CHUNKING 

def sentence_chunking(text):

    sentences = text.replace("?", ".").replace("!", ".").split(".")

    chunks = []

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence != "":

            chunks.append(sentence)

    return chunks

 
# STEP 4: RECURSIVE CHUNKING 

splitter = RecursiveCharacterTextSplitter(

    chunk_size=512,

    chunk_overlap=50
)

 
# STEP 5: COMPARE CHUNK COUNTS 

print("\nCHUNK COUNT COMPARISON")

for document in documents:

    text = document["text"]

    fixed_chunks = fixed_chunking(text, 512, 50)

    sentence_chunks = sentence_chunking(text)

    recursive_chunks = splitter.split_text(text)

    print("\nDocument:", document["source"])

    print("Fixed chunks:", len(fixed_chunks))

    print("Sentence chunks:", len(sentence_chunks))

    print("Recursive chunks:", len(recursive_chunks))

 
# STEP 6: SHOW SENTENCE CUT 

sample_text = (
    "Employees must request annual leave "
    "from their manager. Employees must "
    "also inform their team before taking leave."
)

print("\nSENTENCE CHUNKING EXAMPLE")

fixed_result = fixed_chunking(sample_text, 7, 2)

for chunk in fixed_result:

    print(chunk)

 
# STEP 7: CHUNK SIZE EXPERIMENT 

chunk_sizes = [128, 256, 512, 1024]

questions = [

    {
        "question": "How many annual leave days?",
        "answer": "20 days",
        "source": "doc1.txt",
        "page": 1
    },

    {
        "question": "Who approves leave?",
        "answer": "manager",
        "source": "doc1.txt",
        "page": 1
    },

    {
        "question": "Is sick leave available?",
        "answer": "sick leave",
        "source": "doc1.txt",
        "page": 1
    },

    {
        "question": "When should employees arrive?",
        "answer": "on time",
        "source": "doc2.txt",
        "page": 1
    },

    {
        "question": "How is attendance recorded?",
        "answer": "every day",
        "source": "doc2.txt",
        "page": 1
    },

    {
        "question": "How should passwords be kept?",
        "answer": "confidential",
        "source": "doc3.txt",
        "page": 1
    },

    {
        "question": "How should company information be protected?",
        "answer": "protect company information",
        "source": "doc3.txt",
        "page": 1
    },

    {
        "question": "What benefits do employees receive?",
        "answer": "medical benefits",
        "source": "doc4.txt",
        "page": 1
    },

    {
        "question": "Does the company provide bonuses?",
        "answer": "performance bonuses",
        "source": "doc4.txt",
        "page": 1
    },

    {
        "question": "Where should visitors register?",
        "answer": "reception",
        "source": "doc5.txt",
        "page": 1
    }

]


print("\nCHUNK SIZE EXPERIMENT")

for size in chunk_sizes:

    total_hits = 0

    for question in questions:

        source_document = None

        for document in documents:

            if document["source"] == question["source"]:

                source_document = document

        if source_document is None:
            continue

        chunks = fixed_chunking(
            source_document["text"],
            size,
            50
        )

        found = False

        for chunk in chunks:

            if question["answer"].lower() in chunk.lower():

                found = True
                break

        if found:

            total_hits += 1

    hit_rate = total_hits / len(questions)

    print(
        "Chunk size:", size,
        "Hit rate:", round(hit_rate, 2)
    )

 
# STEP 8: ADD METADATA 

all_chunks = []

for document in documents:

    chunks = splitter.split_text(document["text"])

    for i, chunk in enumerate(chunks):

        chunk_data = {

            "source": document["source"],

            "page": 1,

            "section": "Company Policy",

            "created_date": str(date.today()),

            "chunk_index": i,

            "doc_type": document["source"].split(".")[-1].upper(),

            "text": chunk
        }

        all_chunks.append(chunk_data)


print("\nTOTAL CHUNKS:", len(all_chunks))

 
# STEP 9: CITATION FUNCTION 

def describe_chunk(chunk):

    citation = (
        "Source: " + chunk["source"] +
        ", Page: " + str(chunk["page"]) +
        ", Section: " + chunk["section"]
    )

    return citation


print("\nCHUNK CITATION")

if len(all_chunks) > 0:

    print(describe_chunk(all_chunks[0]))

 
# STEP 10: CREATE STABLE IDS 

def create_id(chunk):

    text = (
        chunk["source"] +
        str(chunk["chunk_index"]) +
        chunk["text"]
    )

    chunk_id = uuid.uuid5(
        uuid.NAMESPACE_DNS,
        text
    )

    return str(chunk_id)


for chunk in all_chunks:

    chunk["id"] = create_id(chunk)


print("\nFIRST CHUNK ID")

if len(all_chunks) > 0:

    print(all_chunks[0]["id"])
 
# STEP 11: INDEX TWICE 

database = {}

# First indexing
for chunk in all_chunks:

    database[chunk["id"]] = chunk

print("\nFIRST RUN COUNT:", len(database))


# Second indexing
for chunk in all_chunks:

    database[chunk["id"]] = chunk

print("SECOND RUN COUNT:", len(database))

 
# STEP 12: CHANGE ONE PARAGRAPH 

old_chunks = all_chunks.copy()

changed_document = documents[0].copy()

changed_document["text"] = (
    changed_document["text"] +
    "\nEmployees can also apply for unpaid leave."
)

new_chunks = splitter.split_text(
    changed_document["text"]
)

new_chunk_data = []

for i, chunk in enumerate(new_chunks):

    new_chunk_data.append({

        "source": changed_document["source"],

        "page": 1,

        "section": "Company Policy",

        "created_date": str(date.today()),

        "chunk_index": i,

        "doc_type": "TXT",

        "text": chunk
    })


# Compare old and new IDs

old_ids = []

for chunk in old_chunks:

    if chunk["source"] == "doc1.txt":

        old_ids.append(create_id(chunk))


new_ids = []

for chunk in new_chunk_data:

    new_ids.append(create_id(chunk))


changed_ids = set(old_ids).symmetric_difference(set(new_ids))

print("\nINCREMENTAL INDEXING")

print("Old chunk count:", len(old_ids))

print("New chunk count:", len(new_ids))

print("Changed chunk IDs:", len(changed_ids))


# Update changed document in database

for chunk_id in old_ids:

    if chunk_id not in new_ids:

        database.pop(chunk_id, None)


for chunk in new_chunk_data:

    chunk["id"] = create_id(chunk)

    database[chunk["id"]] = chunk


print("Final database count:", len(database))
