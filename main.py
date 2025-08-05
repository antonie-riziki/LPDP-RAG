import sys

from model import get_qa_chain, query_system, load_documents
from image_analysis import load_images_from_folder, get_image_details


if __name__ == "__main__":

    # Run the image extraction model first
    qa_chain = get_qa_chain("91bf7702-development-plans-maps_compressed.pdf")

    query = "summarize the LPDP Dcoument plan and provide the status of the plan"
    print(query_system(query, qa_chain))

    docs, imgs = load_documents("91bf7702-development-plans-maps_compressed.pdf", extract_images=True)

    # After Extraction now generate the contextual output
    image_folder = "pdf_images"
    images = load_images_from_folder(image_folder)

    for idx, image in enumerate(images):
        print(f"\n--- Analyzing Image {idx + 1}/{len(images)} ---\n")
        get_image_details(image)