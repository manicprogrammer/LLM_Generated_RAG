# from ChatGPT-4o1-preview
# Prompt:
# You are a computer programmer that specializes in writing functioning AI software in python. 
# You use simple code to do powerful things.  
# Write me some python that does a single document RAG reading from a PDF for OpenAI API.

import openai
import PyPDF2
import numpy as np

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text(text, max_chunk_size=1000):
    """Split text into chunks of a specified maximum size."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ''
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += '\n\n' + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_embedding(text, model='text-embedding-ada-002'):
    """Generate an embedding for a given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

def create_embedding_index(chunks):
    """Create an index of embeddings for text chunks."""
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append({'chunk': chunk, 'embedding': embedding})
    return embeddings

def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query_pdf(query, embeddings_index):
    """Find the most relevant text chunk for a given query."""
    query_embedding = get_embedding(query)
    similarities = []
    for item in embeddings_index:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        similarities.append({'chunk': item['chunk'], 'similarity': similarity})
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    most_similar_chunk = similarities[0]['chunk']
    return most_similar_chunk

def generate_answer(query, context, model='gpt-3.5-turbo'):
    """Generate an answer using the context and query."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that uses the provided context to answer the question."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    answer = response['choices'][0]['message']['content']
    return answer

def main():
    pdf_path = 'document.pdf'  # Replace with your PDF file path
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    embeddings_index = create_embedding_index(chunks)
    
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        context = query_pdf(query, embeddings_index)
        answer = generate_answer(query, context)
        print("Answer:", answer)

if __name__ == '__main__':
    main()
