import fitz
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def extract_images_from_pdf(pdf_path, output_folder="images"):

    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    # open pdf
    doc = fitz.open(pdf_path)

    image_paths = []

    for page_index in range(len(doc)):

        page = doc[page_index]

        images = page.get_images(full=True)

        for image_index, img in enumerate(images):

            xref = img[0]

            # extract image
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image_ext = base_image["ext"]

            # image filename
            image_filename = (
                f"page_{page_index+1}_image_{image_index+1}.{image_ext}"
            )

            image_path = os.path.join(
                output_folder,
                image_filename
            )

            # save image
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

            print(f"Saved: {image_path}")

    doc.close()

    return image_paths


def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


def generate_image_captions(image_paths):

    image_descriptions = []

    system_prompt = """
Describe PDF images for retrieval in a multimodal RAG system.
Be precise and information-dense. Include any text in the image (OCR), labels, numbers, trends, and entities.
Do not start with "The image shows" or "This appears to be" — state facts directly.
"""

    for image_path in image_paths:

        print(f"\nProcessing: {image_path}")

        file_id = create_file(image_path)

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe this image in detail for a multimodal RAG system.",
                        },
                        {
                            "type": "input_image",
                            "file_id": file_id,
                        },
                    ],
                }
            ],
        )

        description = response.output_text

        image_descriptions.append({
            "image_path": image_path,
            "description": description
        })

        print(description)

    return image_descriptions


def extract_and_caption_images(pdf_path, output_folder="images"):

    image_paths = extract_images_from_pdf(pdf_path, output_folder)

    return generate_image_captions(image_paths)