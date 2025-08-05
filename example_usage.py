#!/usr/bin/env python3
"""
Example usage of the multimodal RAG system for development plans analysis
"""

import os
from model import get_qa_chain, query_system, load_documents
from dotenv import load_dotenv

def analyze_development_plans():
    """Example of how to analyze development plans with multimodal RAG"""
    
    # Load environment variables
    load_dotenv()
    
    print("🏗️ Development Plans Analysis with Multimodal RAG")
    print("=" * 60)
    
    # Path to your PDF document
    pdf_path = "91bf7702-development-plans-maps_compressed.pdf"
    
    # Step 1: Initialize the multimodal QA system
    print("📊 Initializing multimodal analysis system...")
    print("   • Extracting text from PDF")
    print("   • Identifying and extracting relevant images")
    print("   • Analyzing images with Gemini Vision")
    print("   • Creating unified vector store")
    print()
    
    qa_chain = get_qa_chain(pdf_path)
    
    if isinstance(qa_chain, str):  # Error case
        print(f"❌ Error: {qa_chain}")
        return
    
    print("✅ System ready! Now you can ask questions that leverage both text and visual content.")
    print()
    
    # Step 2: Example queries for development plans
    print("🔍 Example Analysis Queries:")
    print("-" * 40)
    
    # Original query from your code
    original_query = "can you analyze the plans in the Mithuri Centre Base Map alone, provide the status of the plan"
    print(f"📍 Query: {original_query}")
    result = query_system(original_query, qa_chain)
    print(f"💡 Answer: {result}")
    print()
    
    # Additional example queries that showcase multimodal capabilities
    additional_queries = [
        "What visual elements and spatial layouts are shown in the development plans?",
        "Describe the zoning structure and organization visible in the planning documents",
        "What technical details or measurements can be seen in the site plans?"
    ]
    
    for query in additional_queries:
        print(f"📍 Query: {query}")
        result = query_system(query, qa_chain)
        print(f"💡 Answer: {result}")
        print()
    
    print("🎯 Key Benefits of Multimodal Analysis:")
    print("   • Combines textual descriptions with visual analysis")
    print("   • Understands spatial relationships and layouts")
    print("   • Analyzes technical diagrams and maps")
    print("   • Provides comprehensive context-aware responses")

def inspect_extracted_content():
    """Show what content was extracted for analysis"""
    
    print("\n🔍 Content Inspection:")
    print("=" * 30)
    
    # Load and inspect the documents
    docs, images = load_documents(
        "91bf7702-development-plans-maps_compressed.pdf", 
        extract_images=True
    )
    
    text_docs = [doc for doc in docs if doc.metadata.get("type") != "image_analysis"]
    image_docs = [doc for doc in docs if doc.metadata.get("type") == "image_analysis"]
    
    print(f"📄 Text documents: {len(text_docs)}")
    print(f"🖼️  Image analyses: {len(image_docs)}")
    print(f"📁 Extracted images: {len(images)}")
    
    if image_docs:
        print("\n🖼️  Sample image analysis:")
        sample_image_doc = image_docs[0]
        print(f"   Filename: {sample_image_doc.metadata.get('filename', 'Unknown')}")
        print(f"   Page: {sample_image_doc.metadata.get('page', 'Unknown')}")
        print(f"   Content preview: {sample_image_doc.page_content[:200]}...")

if __name__ == "__main__":
    # Main analysis
    analyze_development_plans()
    
    # Optional: Show extracted content details
    # inspect_extracted_content()