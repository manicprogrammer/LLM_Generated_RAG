# # PDF Extraction: The extract_text_from_pdf function uses PyPDF2 to extract all the text from the PDF file.
# Document Splitting: The extracted text is split into manageable chunks.
# Embedding: Each chunk is embedded using OpenAI's embedding model.
# Retrieval: The query is embedded and used to search for the most relevant chunk using FAISS.
# Generation: The most relevant chunk is passed along with the query to OpenAI’s text-davinci-003 model to generate a response.
# Usage:
# Replace pdf_path with the path to your PDF file and query with your question. The rag function will read the PDF, retrieve the relevant chunk, and generate a response based on that chunk.
# This setup allows you to use a PDF as the source document for your RAG model, making it more versatile for different types of content.
# pip install openai faiss-cpu PyPDF
# Simple prompt: You are a computer programmer that specializes in writing functioning AI software in python. 
#    You use simple code to do powerful things. 
#    Write me some python that does a single document RAG for OpenAI API.
# then had to clarify prompt to add this: have it read the document from a PDF instead of plain text input

import openai
import faiss
import numpy as np
import re
import PyPDF2

# Initialize OpenAI API key
openai.api_key = "your-openai-api-key"

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Split a document into chunks
def split_document(doc, chunk_size=500):
    doc = re.sub(r'\s+', ' ', doc)  # Normalize whitespaces
    return [doc[i:i + chunk_size] for i in range(0, len(doc), chunk_size)]

# Embed chunks using OpenAI
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings).astype('float32')

# Search for the most relevant chunk using FAISS
def search_chunks(query, chunks, chunk_embeddings):
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']

    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    
    _, I = index.search(query_embedding, 1)
    
    return chunks[I[0][0]]

# Generate response using the relevant chunk
def generate_response(query, chunk):
    prompt = f"Query: {query}\nContext: {chunk}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

# Main function to perform RAG
def rag(query, pdf_path):
    document = extract_text_from_pdf(pdf_path)
    chunks = split_document(document)
    chunk_embeddings = embed_chunks(chunks)
    relevant_chunk = search_chunks(query, chunks, chunk_embeddings)
    response = generate_response(query, relevant_chunk)
    return response

# Example usage
pdf_path = "example.pdf"
query = "What is Self-Sovereign Identity?"

# Perform RAG
response = rag(query, pdf_path)
print("Response:", response)
