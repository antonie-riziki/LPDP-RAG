#!/usr/bin/env python3
"""
Test script for multimodal RAG system that processes both PDF text and images
"""

import os
from model import get_qa_chain, query_system, load_documents
from dotenv import load_dotenv

def main():
    """Test the multimodal RAG system"""
    load_dotenv()
    
    # Check if API key is available
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Please set your GOOGLE_API_KEY in a .env file")
        return
    
    print("🚀 Testing Multimodal RAG System")
    print("=" * 50)
    
    # Path to your PDF
    pdf_path = "91bf7702-development-plans-maps_compressed.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Processing PDF: {pdf_path}")
    print("⏳ This may take a few minutes as images are analyzed with Gemini Vision...")
    
    # Initialize the QA chain
    qa_chain = get_qa_chain(pdf_path)
    
    if isinstance(qa_chain, str):  # Error occurred
        print(f"❌ Error initializing system: {qa_chain}")
        return
    
    print("✅ Multimodal RAG system initialized successfully!")
    print("\n" + "=" * 50)
    
    # Test queries that should benefit from both text and image analysis
    test_queries = [
        "Can you analyze the plans in the Mithuri Centre Base Map alone, provide the status of the plan",
        "What visual elements or layouts are shown in the development plans?",
        "Describe any zoning information or site layouts mentioned in the documents",
        "What are the key features of the development plans shown in the images?",
        "Analyze the spatial organization and structure shown in the planning documents"
    ]
    
    print("🔍 Testing queries on multimodal content:")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("   Answer:")
        result = query_system(query, qa_chain)
        print(f"   {result}")
        print("-" * 50)
    
    print("\n✨ Multimodal RAG testing completed!")
    
    # Interactive mode
    print("\n💬 Interactive mode - Ask your own questions (type 'quit' to exit):")
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        if user_query:
            result = query_system(user_query, qa_chain)
            print(f"\nAnswer: {result}")
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()