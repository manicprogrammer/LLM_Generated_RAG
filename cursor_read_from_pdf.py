# Simple prompt: provide code for using retrieval augmented generation on a single PDF using the OpenAI API
# This version of the script has been edited from what was generated to be able to run.
# edited to run with the OpenAI Python V1.43.0
##
# response:
#    The main topic of the document is Self-Sovereign Identity (SSI), which is an identity model that emphasizes 
#    user control over personal data. The document presents a systematic review of the literature on SSI,
#    mapping its theoretical and practical advancements, and proposing a taxonomy to categorize the research.
#    It discusses concepts, challenges, and solutions related to SSI while addressing issues such as privacy, 
#    identity management, and the evolution of identity models. 
#    The study aims to provide an organized understanding of SSI literature and to highlight areas for future research.
##
# Due to how Cursor decided to create the solution the response is the best response of the set of responses I could get.
# This makes sense in that it is using the full document for context rather than an identified chunk. 

import os
import PyPDF2
import openai
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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