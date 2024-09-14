# Simple Prompt: provide me python code to use OpenAI API to perform retrieval augmented generation on a pdf
##
# I tried to start editing this but it was so far off base in the generated code that I just stopped. It's doing some weird stuff like
# chunking the document then taking each chunk of the doc and passing it to the LLM with the prompt to extract up to 150 tokens from that chunk
# no embeddings or anything. A very weird proposed solution and entirely unworkable.
#
# It's funny - CoPilot is in use as I type these comments and suggesting comments to add. One suggestion was:
# "I think it would be better to just read the PDF and pass the whole thing to the LLM with the prompt to answer the question."
# How insightful of it! 
# Why didn't it just do that in the first place? It would have provided a simple solution to the simple prompt. It's exactly what Cursor did
##

# Step 1: Install Required Libraries
# pip install pymupdf openai

import fitz  # PyMuPDF
import openai
import os
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Step 2: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Step 3: Chunk Text
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Step 4: Retrieve Relevant Information
def retrieve_information(chunks, query, api_key):
    openai.api_key = api_key
    retrieved_info = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="davinci",
            prompt=f"Extract relevant information for the query: '{query}' from the following text:\n\n{chunk}",
            max_tokens=150
        )
        retrieved_info.append(response.choices[0].text.strip())
    return " ".join(retrieved_info)

# Step 5: Generate Response
def generate_response(retrieved_info, query, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Based on the following information, generate a response to the query: '{query}'\n\n{retrieved_info}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Main Function
def main(pdf_path, query, api_key):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    retrieved_info = retrieve_information(chunks, query, api_key)
    response = generate_response(retrieved_info, query, api_key)
    return response

# Example Usage
if __name__ == "__main__":
    pdf_path = 'data\Self-Sovereign Identity A Systematic Review Mapping and Taxonomy.pdf'
    query = "What is the main topic of the document?"
    api_key = openai.api_key  # Assuming the API key is already loaded
    response = main(pdf_path, query, api_key)
    print(response)