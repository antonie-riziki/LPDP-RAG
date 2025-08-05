import os
import sys
import io
import glob
import fitz 
import getpass
import warnings
import base64
from PIL import Image
from typing import List, Union, Tuple
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
import google.generativeai as genai

warnings.filterwarnings("ignore")

sys.path.insert(1, './src')

load_dotenv()

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    GEMINI_API_KEY = getpass.getpass("Enter you Google Gemini API key: ")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

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
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    return model, embeddings

def load_vision_model():
    """
    Load Gemini Vision model for image analysis
    """
    return genai.GenerativeModel('gemini-2.5-flash')

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_gemini(image_path: str, context_prompt: str = None) -> str:
    """
    Analyze image using Gemini Vision and return detailed description
    """
    try:
        model = load_vision_model()
        
        # Load the image
        image = Image.open(image_path)
        
        # Create a comprehensive prompt for image analysis
        if context_prompt:
            prompt = f"""
            {context_prompt}
            
            Please analyze this image in detail. Focus on:
            1. Visual elements and layout
            2. Any text, labels, or annotations visible
            3. Spatial relationships and organization
            4. Key features that would be relevant for document understanding
            5. Any technical diagrams, maps, or plans shown
            
            Provide a comprehensive description that would help in answering questions about this document.
            """
        else:
            prompt = """
            Analyze this image in detail. Describe:
            1. What type of document or diagram this appears to be
            2. Key visual elements, text, and layout
            3. Any technical information, measurements, or annotations
            4. Spatial relationships and organizational structure
            5. Important details that would help understand the content
            
            Provide a comprehensive description suitable for document analysis and question answering.
            """
        
        response = model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return f"Image analysis failed for {os.path.basename(image_path)}: {str(e)}"

def extract_images_from_pdf(pdf_path: str, image_output_dir: str):
    """
    Extracts embedded images and page previews from a PDF.
    Saves them in the image_output_dir and returns a list of paths.
    """
    os.makedirs(image_output_dir, exist_ok=True)
    image_paths = []
    doc = fitz.open(pdf_path)

    keywords = ['zoning', 'plan', 'layout', 'structure', 'development', 'site']

    for page_index in range(len(doc)):
        page_text = doc[page_index].get_text().lower()
        if not any(word in page_text for word in keywords):
            continue  # skip these pages

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

def process_images_to_documents(image_paths: List[str], pdf_path: str) -> List[Document]:
    """
    Process images using Gemini Vision and convert descriptions to LangChain Documents
    """
    image_documents = []
    
    print(f"Processing {len(image_paths)} images with Gemini Vision...")
    
    for i, image_path in enumerate(image_paths):
        print(f"Analyzing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Extract page information from filename
        filename = os.path.basename(image_path)
        page_info = filename.split('_')
        page_num = page_info[2] if len(page_info) > 2 else "unknown"
        
        # Create context-aware prompt
        context_prompt = f"""
        This image is extracted from a PDF document about development plans and maps.
        The image comes from page {page_num} of the document.
        This appears to be related to zoning, planning, layout, structure, or development content.
        """
        
        # Analyze the image
        description = analyze_image_with_gemini(image_path, context_prompt)
        
        # Create a Document with the image description
        doc = Document(
            page_content=f"IMAGE ANALYSIS - {filename}:\n{description}",
            metadata={
                "source": pdf_path,
                "image_path": image_path,
                "page": page_num,
                "type": "image_analysis",
                "filename": filename
            }
        )
        image_documents.append(doc)
    
    return image_documents

def load_documents(source_dir: str, extract_images: bool = True, image_output_dir: str = "pdf_images") -> Tuple[List[Document], List[str]]:
    """
    Load documents from multiple sources including image extraction and analysis from PDFs
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
            # Load text documents
            text_docs = PyPDFLoader(source_dir).load()
            documents.extend(text_docs)
            
            if extract_images:
                # Extract images
                image_paths = extract_images_from_pdf(source_dir, image_output_dir)
                images.extend(image_paths)
                
                # Process images and add their descriptions as documents
                if image_paths:
                    image_documents = process_images_to_documents(image_paths, source_dir)
                    documents.extend(image_documents)
                    print(f"Added {len(image_documents)} image analysis documents")
                
        elif ext == ".csv":
            documents.extend(CSVLoader(source_dir).load())
    else:
        for pattern, loader in file_types.items():
            for file_path in glob.glob(os.path.join(source_dir, pattern)):
                if pattern == "*.pdf":
                    # Load text documents
                    text_docs = loader(file_path).load()
                    documents.extend(text_docs)
                    
                    if extract_images:
                        # Extract and process images
                        image_paths = extract_images_from_pdf(file_path, image_output_dir)
                        images.extend(image_paths)
                        
                        if image_paths:
                            image_documents = process_images_to_documents(image_paths, file_path)
                            documents.extend(image_documents)
                else:
                    documents.extend(loader(file_path).load())

    return documents, images

def create_vector_store(docs: List[Document], embeddings, chunk_size: int = 10000, chunk_overlap: int = 200):
    """
    Create vector store from documents (both text and image descriptions)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents, but handle image analysis documents differently
    all_splits = []
    for doc in docs:
        if doc.metadata.get("type") == "image_analysis":
            # Don't split image analysis documents to maintain context
            all_splits.append(doc)
        else:
            # Split regular text documents
            splits = text_splitter.split_documents([doc])
            all_splits.extend(splits)
    
    print(f"Creating vector store with {len(all_splits)} document chunks")
    return FAISS.from_documents(all_splits, embeddings).as_retriever(search_kwargs={"k": 7})

# Enhanced prompt template that handles both text and image content
MULTIMODAL_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. The context may include both text content and image analysis descriptions.

When image analysis is provided, it will be clearly marked as "IMAGE ANALYSIS" and will contain detailed descriptions of visual elements, diagrams, maps, or plans.

Pay special attention to:
- Visual elements described in image analyses
- Spatial relationships and layouts mentioned
- Technical diagrams, maps, or planning documents
- Integration between text content and visual information

If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""

def get_qa_chain(source_dir):
    """Create QA chain with multimodal support and proper error handling"""
    try:
        docs, images = load_documents(source_dir, extract_images=True)
        if not docs:
            raise ValueError("No documents found in the specified sources")

        print(f"Loaded {len(docs)} total documents (including image analyses)")
        print(f"Extracted {len(images)} images")

        llm, embeddings = load_model()
        retriever = create_vector_store(docs, embeddings)

        prompt = PromptTemplate(
            template=MULTIMODAL_PROMPT_TEMPLATE,
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
        print(f"Error initializing QA system: {e}")
        return f"Error initializing QA system: {e}"

def query_system(query: str, qa_chain):
    """Enhanced query system that can handle multimodal context"""
    if not qa_chain:
        return "System not initialized properly"

    try:
        result = qa_chain({"query": query})
        if not result["result"] or "don't know" in result["result"].lower():
            return "The answer could not be found in the provided documents"
        
        # Enhanced response with source information
        response = f"{result['result']}"
        
        # Add source information
        sources = []
        for doc in result['source_documents']:
            source_info = f"{doc.metadata.get('source', 'Unknown')}"
            if doc.metadata.get('type') == 'image_analysis':
                source_info += f" (Image: {doc.metadata.get('filename', 'Unknown')})"
            elif 'page' in doc.metadata:
                source_info += f" (Page: {doc.metadata.get('page', 'Unknown')})"
            sources.append(source_info)
        
        if sources:
            response += f"\n\nSources: {list(set(sources))}"
        
        return response
        
    except Exception as e:
        return f"Error processing query: {e}"

# Test the enhanced system
if __name__ == "__main__":
    qa_chain = get_qa_chain("91bf7702-development-plans-maps_compressed.pdf")
    
    query = "can you analyze the plans in the Mithuri Centre Base Map alone, provide the status of the plan"
    print(query_system(query, qa_chain))
