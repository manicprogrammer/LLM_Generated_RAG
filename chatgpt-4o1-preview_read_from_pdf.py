# from ChatGPT-4o1-preview
# Prompt:
# You are a computer programmer that specializes in writing functioning AI software in python. 
# You use simple code to do powerful things.  
# Write me some python that does a single document RAG reading from a PDF for OpenAI API.
#
# there was the following note as part of the output:
# Note: This script is designed for simplicity and may not handle very large documents efficiently. 
# For more advanced use cases, consider integrating a vector database 
# or optimizing the text splitting and embedding process.
#
# has not yet been edited to be executable with the desired results

import os
import openai
import PyPDF2
import numpy as np

# Install required packages if not already installed:
# pip install openai PyPDF2 numpy tiktoken

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def read_pdf(file_path):
    """Read and extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(open(file_path, 'rb'))
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def split_text(text, max_tokens=500):
    """Split text into chunks of a specified token length."""
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    """Get the embedding of a text chunk using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    embedding = response['data'][0]['embedding']
    return embedding

def compute_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar_chunks(question_embedding, chunk_embeddings, top_k=3):
    """Find the most similar text chunks to the user's question."""
    similarities = [compute_similarity(question_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices

def answer_question(question, chunks, chunk_embeddings):
    """Generate an answer to the user's question using relevant text chunks."""
    # Get embedding for the question
    question_embedding = get_embedding(question)
    # Find most similar chunks
    top_indices = find_most_similar_chunks(question_embedding, chunk_embeddings)
    # Retrieve relevant chunks
    relevant_chunks = [chunks[i] for i in top_indices]
    # Construct the prompt
    context = "\n\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Get the answer from OpenAI
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=200,
        temperature=0
    )
    answer = response['choices'][0]['text'].strip()
    return answer

def main():
    # Path to your PDF file
    file_path = 'document.pdf'  # Replace with your PDF file path
    # Read the PDF
    text = read_pdf(file_path)
    # Split the text into chunks
    chunks = split_text(text)
    # Get embeddings for the chunks
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    # Get user question
    question = input("Enter your question: ")
    # Get the answer
    answer = answer_question(question, chunks, chunk_embeddings)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
