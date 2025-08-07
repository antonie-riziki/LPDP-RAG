

import os
import google.generativeai as genai
from PIL import Image
from typing import BinaryIO

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))



def load_image_file(file_obj: BinaryIO):
    try:
        image = Image.open(file_obj).convert("RGB")
        return image
    except Exception as e:
        raise OSError(f"Failed to load image from file object: {e}")




def get_image_details(image):
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
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

            Use OCR to extract text annotations, legends, and metadata in the image. 
            Use your spatial reasoning to interpret color zones, line work, and layout hierarchy.

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
    with open("", "rb") as f:
        image_input = load_image_file(f)
        get_image_details(image_input)

