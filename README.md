# Multimodal RAG System for PDF Analysis

This project implements a **multimodal Retrieval-Augmented Generation (RAG) system** that can process both text content and images from PDFs using Google's Gemini models. The system extracts images from PDFs, analyzes them with Gemini Vision, and combines the visual analysis with text content for comprehensive question answering.

## 🚀 Features

- **Text Processing**: Extracts and chunks text from PDFs using LangChain
- **Image Extraction**: Intelligently extracts relevant images from PDFs based on keywords
- **Vision Analysis**: Uses Gemini Vision (gemini-2.5-flash) to analyze extracted images
- **Multimodal Embedding**: Combines text and image descriptions in a single vector store
- **Enhanced Retrieval**: Retrieves both textual and visual information for comprehensive answers
- **Contextual Understanding**: Provides context-aware responses using both text and visual elements

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## 📖 How It Works

### 1. **Image Extraction**
```python
def extract_images_from_pdf(pdf_path: str, image_output_dir: str):
```
- Scans PDF pages for keywords: 'zoning', 'plan', 'layout', 'structure', 'development', 'site'
- Extracts embedded images that meet size and aspect ratio criteria
- Saves images to local directory for analysis

### 2. **Vision Analysis**
```python
def analyze_image_with_gemini(image_path: str, context_prompt: str = None):
```
- Uses Gemini Vision model to analyze each extracted image
- Generates detailed descriptions focusing on:
  - Visual elements and layout
  - Text, labels, and annotations
  - Spatial relationships
  - Technical diagrams and plans
  - Relevant details for document understanding

### 3. **Document Processing**
```python
def process_images_to_documents(image_paths: List[str], pdf_path: str):
```
- Converts image analysis results into LangChain Document objects
- Adds metadata including source, page number, and image path
- Creates searchable text descriptions of visual content

### 4. **Multimodal Vector Store**
```python
def create_vector_store(docs: List[Document], embeddings):
```
- Combines text chunks and image descriptions
- Handles image analysis documents without chunking to preserve context
- Creates FAISS vector store for efficient retrieval

### 5. **Enhanced Retrieval**
- Uses specialized prompt template for multimodal content
- Retrieves relevant text and image descriptions together
- Provides source attribution for both text and images

## 🔧 Usage

### Basic Usage
```python
from model import get_qa_chain, query_system

# Initialize the multimodal QA system
qa_chain = get_qa_chain("your_pdf_file.pdf")

# Ask questions that can benefit from both text and images
query = "Analyze the development plans and describe the layout shown"
result = query_system(query, qa_chain)
print(result)
```

### Running the Test Script
```bash
python test_multimodal.py
```

This will:
1. Process your PDF with both text and image analysis
2. Run several test queries
3. Enter interactive mode for custom questions

## 🎯 Key Improvements Over Standard RAG

### Standard RAG Limitations:
- ❌ Only processes text content
- ❌ Misses visual information in diagrams, maps, and plans
- ❌ Cannot understand spatial relationships
- ❌ Limited context for technical documents

### Multimodal RAG Benefits:
- ✅ **Visual Understanding**: Analyzes diagrams, maps, and technical drawings
- ✅ **Spatial Awareness**: Understands layouts and spatial relationships
- ✅ **Complete Context**: Combines textual and visual information
- ✅ **Better Accuracy**: More comprehensive answers for technical documents
- ✅ **Rich Descriptions**: Detailed analysis of visual elements

## 📊 Example Use Cases

### Development Plans Analysis
```python
query = "What is the status of the Mithuri Centre Base Map plan?"
# Returns: Combined analysis of text descriptions AND visual plan layouts
```

### Zoning Information
```python
query = "Describe the zoning layout and structure shown in the documents"
# Returns: Text-based zoning info PLUS visual analysis of zoning maps
```

### Technical Diagrams
```python
query = "What technical features are visible in the site plans?"
# Returns: Detailed description of visual elements, measurements, and annotations
```

## 🔄 System Architecture

```
PDF Input
    ↓
┌─────────────────┬─────────────────┐
│   Text Extract  │  Image Extract  │
│   (PyPDFLoader) │   (PyMuPDF)     │
└─────────────────┼─────────────────┘
    ↓             ↓
Text Chunks    Gemini Vision
    ↓             ↓
    └──→ Vector Store ←──┘
            ↓
      FAISS Retrieval
            ↓
      Gemini LLM Response
```

## ⚙️ Configuration

### Image Filtering
- Minimum size: 500x500 pixels
- Aspect ratio: 0.5 to 2.0
- Keyword filtering for relevant pages

### Vector Store Settings
- Chunk size: 10,000 characters
- Chunk overlap: 200 characters
- Retrieval: Top 7 results
- Image descriptions: No chunking (preserve context)

### Model Configuration
- **LLM**: gemini-2.5-flash
- **Vision**: gemini-2.5-flash
- **Embeddings**: text-embedding-004

## 🚨 Important Notes

1. **API Costs**: Image analysis uses Gemini Vision API calls - monitor your usage
2. **Processing Time**: Initial setup takes longer due to image analysis
3. **Image Quality**: Better image quality = better analysis results
4. **Keywords**: Adjust keywords in `extract_images_from_pdf()` for your domain

## 📝 Customization

### Adding New File Types
```python
file_types = {
    "*.pdf": PyPDFLoader,
    "*.csv": CSVLoader,
    "*.docx": DocxLoader,  # Add new types here
}
```

### Custom Vision Prompts
```python
context_prompt = f"""
Custom context for your domain...
Focus on specific elements relevant to your use case...
"""
```

### Adjusting Image Filters
```python
keywords = ['your', 'domain', 'specific', 'keywords']
min_size = 300  # Adjust minimum image size
aspect_ratio_range = (0.3, 3.0)  # Adjust aspect ratio limits
```

## 🔮 Future Enhancements

- [ ] Support for multiple PDF processing
- [ ] Image similarity search
- [ ] OCR for text within images
- [ ] Table extraction and analysis
- [ ] Multi-language support
- [ ] Caching for processed images

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
