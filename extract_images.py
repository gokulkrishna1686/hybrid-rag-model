import fitz
import os
from pydantic import BaseModel
from typing import List, Optional, Literal
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

ImageType = Literal[
    "chart", "diagram", "table", "photo", "screenshot",
    "illustration", "logo", "map", "icon", "text_block", "mixed", "other"
]

EntityType = Literal[
    "logo", "chart", "table", "photo", "icon", "text_block",
    "headline", "label", "shape", "arrow", "ui_element", "other"
]

class Entity(BaseModel):
    type: EntityType
    colors: List[str]
    shape: Optional[str] = None
    text: Optional[str] = None
    description: str

class ImageSchema(BaseModel):
    overall_type: ImageType
    summary: str
    entities: List[Entity]
    keywords: List[str]

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

def schema_to_text(schema: ImageSchema) -> str:
    parts = [f"TYPE: {schema.overall_type}", f"SUMMARY: {schema.summary}"]
    for e in schema.entities:
        bits = [f"entity={e.type}", f"colors={', '.join(e.colors)}"]
        if e.shape: bits.append(f"shape={e.shape}")
        if e.text:  bits.append(f"text={e.text}")
        bits.append(f"desc={e.description}")
        parts.append(" | ".join(bits))
    parts.append("KEYWORDS: " + ", ".join(schema.keywords))
    return "\n".join(parts)


def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


def generate_image_captions(image_paths):
    descriptions = []
    for image_path in image_paths:
        file_id = create_file(image_path)
        response = client.responses.parse(
            model="gpt-4o-mini",
            instructions=(
                "Extract a structured description of this image for retrieval. "
                "List every distinct entity (logo, chart, text block, icon, etc.) separately. "
                "Be precise about colors and shapes. Capture all visible text verbatim."
            ),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the schema."},
                    {"type": "input_image", "file_id": file_id},
                ],
            }],
            text_format=ImageSchema,
        )
        schema = response.output_parsed
        descriptions.append({
            "image_path": image_path,
            "schema": schema,
            "description": schema_to_text(schema),
        })
        print(f"Captioned: {image_path}")
    return descriptions


def extract_and_caption_images(pdf_path, output_folder="images"):

    image_paths = extract_images_from_pdf(pdf_path, output_folder)
    return generate_image_captions(image_paths)


def image_to_documents(item):
    """One image -> parent summary doc + N entity docs, linked by image_id."""
    from langchain_core.documents import Document  # or move to top of file

    schema = item["schema"]
    image_path = item["image_path"]
    image_id = os.path.basename(image_path)

    docs = []

    parent_text = (
        f"TYPE: {schema.overall_type}\n"
        f"SUMMARY: {schema.summary}\n"
        f"KEYWORDS: {', '.join(schema.keywords)}"
    )
    docs.append(Document(
        page_content=parent_text,
        metadata={
            "source": image_path,
            "type": "image_summary",
            "image_id": image_id,
        },
    ))

    for i, e in enumerate(schema.entities):
        bits = [f"entity={e.type}", f"colors={', '.join(e.colors)}"]
        if e.shape: bits.append(f"shape={e.shape}")
        if e.text:  bits.append(f"text={e.text}")
        bits.append(f"desc={e.description}")
        entity_text = f"From image: {schema.summary}\n" + " | ".join(bits)

        docs.append(Document(
            page_content=entity_text,
            metadata={
                "source": image_path,
                "type": "image_entity",
                "image_id": image_id,
                "entity_index": i,
            },
        ))

    return docs