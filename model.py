import os
import io
import sys
import glob
import fitz
import getpass
import warnings
import pytesseract
from PIL import Image
from typing import List, Union
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader
)
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from pdf2image import convert_from_path

warnings.filterwarnings("ignore")



load_dotenv()

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
  GEMINI_API_KEY = getpass.getpass("Enter you Google Gemini API key: ")



def load_model():
  """
  Func loads the model and embeddings
  """
  model = ChatGoogleGenerativeAI(
      model="models/gemini-2.5-flash",
      google_api_key=GEMINI_API_KEY,
      temperature=0.4,
      convert_system_message_to_human=True
  )
  embeddings = GoogleGenerativeAIEmbeddings(
      # model="models/embedding-004",
      model="models/text-embedding-004",
      google_api_key=GEMINI_API_KEY
  )
  return model, embeddings


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def extract_images_from_pdf(pdf_path: str, image_output_dir: str):
    os.makedirs(image_output_dir, exist_ok=True)
    image_paths = []
    doc = fitz.open(pdf_path)

    keywords = ['zoning', 'plan', 'layout', 'structure', 'development', 'site']

    for page_index in range(len(doc)):
        page_text = doc[page_index].get_text().lower()
        if not any(word in page_text for word in keywords):
            continue  # skip irrelevant pages

        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            # Check size using PIL
            image = Image.open(io.BytesIO(image_bytes))
            if image.width < 500 or image.height < 500:
                continue  # Skip small images

            aspect_ratio = image.width / image.height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue  # Skip weirdly shaped images

            image_path = os.path.join(
                image_output_dir, f"{os.path.basename(pdf_path).replace('.pdf','')}_filtered_{page_index}_{img_index}.{ext}"
            )
            image.save(image_path)
            image_paths.append(image_path)

    return image_paths


def load_documents(source_dir: str, extract_images: bool = True, image_output_dir: str = "pdf_images"):
    """
    Load documents from multiple sources including image extraction from PDFs
    """
    documents = []
    images = []

    file_types = {
        "*.pdf": PyPDFLoader,
        "*.csv": CSVLoader
    }

    if os.path.isfile(source_dir):
        ext = os.path.splitext(source_dir)[1].lower()
        if ext == ".pdf":
            documents.extend(PyPDFLoader(source_dir).load())
            if extract_images:
                images.extend(extract_images_from_pdf(source_dir, image_output_dir))
        elif ext == ".csv":
            documents.extend(CSVLoader(source_dir).load())
    else:
        for pattern, loader in file_types.items():
            for file_path in glob.glob(os.path.join(source_dir, pattern)):
                documents.extend(loader(file_path).load())
                if pattern == "*.pdf" and extract_images:
                    images.extend(extract_images_from_pdf(file_path, image_output_dir))

    return documents, images



def ocr_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"OCR failed for {image_path}: {str(e)}"



def images_to_documents(image_paths: List[str]) -> List[Document]:
    image_docs = []
    for path in image_paths:
        ocr_text = ocr_image(path)
        if ocr_text:
            image_docs.append(Document(
                page_content=ocr_text,
                metadata={"source": path}
            ))
    return image_docs

# ++++++++++++++++++++++++++++++++++++++++++++++




def create_vector_store(docs: List[Document], embeddings, chunk_size: int = 10000, chunk_overlap: int = 200):
  """
  Create vector store from documents
  """
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
  )
  splits = text_splitter.split_documents(docs)
  return FAISS.from_documents(splits, embeddings).as_retriever(search_kwargs={"k": 5})




PROMPT_TEMPLATE = """
  Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  {context}

  Question: {question}
  Answer:"""



def get_qa_chain(source_dir):
  """
  Create QA chain with proper error handling
  """

  try:
      docs, image_paths = load_documents(source_dir, extract_images=True)
      image_docs = images_to_documents(image_paths)

      all_docs = docs + image_docs  

      llm, embeddings = load_model()
      retriever = create_vector_store(all_docs, embeddings)

      prompt = PromptTemplate(
          template=PROMPT_TEMPLATE,
          input_variables=["context", "question"]
      )

      response = RetrievalQA.from_chain_type(
          llm=llm,
          chain_type="stuff",
          retriever=retriever,
          return_source_documents=True,
          chain_type_kwargs={"prompt": prompt}
      )

      return response

  except Exception as e:
      return f"Error initializing QA system: {e}"



def query_system(query: str, qa_chain):
  if not qa_chain:
    return "System not initialized properly"

  try:
    result = qa_chain({"query": query})
    if not result["result"] or "don't know" in result["result"].lower():
      return "The answer could not be found in the provided documents"
    return f"{result['result']}" #\nSources: {[s.metadata['source'] for s in result['source_documents']]}"
  except Exception as e:
    return f"Error processing query: {e}"




