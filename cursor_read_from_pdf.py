# Simple prompt: provide code for using retrieval augmented generation on a single PDF using the OpenAI API

import os
import PyPDF2
import openai
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Function to read PDF content
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to generate a response using OpenAI API
def generate_response(prompt):
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Main function to use RAG on a PDF
def rag_on_pdf(pdf_path, query):
    pdf_content = read_pdf(pdf_path)
    prompt = f"Based on the following content from a PDF:\n\n{pdf_content}\n\nAnswer the question: {query}"
    answer = generate_response(prompt)
    return answer

# Example usage
pdf_path = 'data\Self-Sovereign Identity A Systematic Review Mapping and Taxonomy.pdf'
query = 'What is the main topic of the document?'
response = rag_on_pdf(pdf_path, query)
print(response)