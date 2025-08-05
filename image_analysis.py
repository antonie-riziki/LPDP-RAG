import os
import google.generativeai as genai
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# def initialize_gemini(api_key: str):
#     genai.configure(api_key=api_key)
#     return genai.GenerativeModel(model_name="models/gemini-pro-vision")





def load_images_from_folder(folder_path: str, limit: int = 10):
    images = []
    supported_ext = ['.png', '.jpg', '.jpeg', '.webp']
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_ext):
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                if len(images) >= limit:
                    break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images



def get_image_details(image):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""

            You are an expert in urban planning, architecture, and infrastructure development. 
            Given an image of a Local Physical Development Plan (LPDP), technical site layout, or 
            zoning map:

            Carefully analyze the visual contents of the image, including legends, zoning codes, 
            color coding, infrastructure markings, and annotations.

            Identify and explain key features such as roads, building zones, green spaces, 
            utilities, public spaces, and land use zones.

            Determine the status or stage of the development (e.g., proposed, under review, 
            approved, implemented).

            Infer the intended purpose of the plan and its impact on the surrounding area 
            (e.g., population distribution, commercial growth, transportation improvements).

            Highlight any missing components, inconsistencies, or areas requiring further attention.

            Provide a summary of the development plan using technical and contextual insight.

            Output format:

            - Plan Type & Purpose: [e.g., Urban residential layout, zoning adjustment]
            - Key Observations: [e.g., Major road networks, utility placements, high-density zones]
            - Development Status: [e.g., Final draft under review]
            - Potential Implications: [e.g., Increase in traffic, commercial expansion]
            - Recommendations (if any): [e.g., Consider adding public transport access]

            If the image does not clearly indicate these aspects, say so with reasoning.

            """
        
        # Generate content with temperature set to 1.5
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = model.generate_content(
            [prompt, image],
            generation_config=generation_config
        )
        print(response.text)
    
    except Exception as e:
        return f"Error generating information: {str(e)}"


if __name__ == "__main__":
    image_folder = "pdf_images"
    images = load_images_from_folder(image_folder)

    for idx, image in enumerate(images):
        print(f"\n--- Analyzing Image {idx + 1}/{len(images)} ---\n")
        get_image_details(image)
