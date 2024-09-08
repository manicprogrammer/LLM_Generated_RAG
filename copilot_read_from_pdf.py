# Simple Prompt: provide me python code to use OpenAI API to perform retrieval augmented generation on a pdf

# Step 1: Install Required Libraries
# pip install pymupdf openai

import fitz  # PyMuPDF
import openai


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
    pdf_path = "example.pdf"
    query = "What is the main topic of the document?"
    api_key = "your_openai_api_key"
    response = main(pdf_path, query, api_key)
    print(response)