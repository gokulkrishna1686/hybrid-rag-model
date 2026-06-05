import fitz
import os
import zipfile
from pydantic import BaseModel, Field
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
    colors: List[str] = Field(
        description=(
            "Descriptive color names actually visible on this entity, ordered by "
            "prominence. Use common names like 'navy blue', 'light gray', 'pale yellow', "
            "'forest green', 'orange', 'white', 'black'. NEVER return hex codes. "
            "For text entities, give the color of the text itself AND its background "
            "(e.g. ['white text', 'red background']). If unsure, omit the color "
            "rather than guessing — do NOT default to 'black'."
        )
    )
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
            model="gpt-4o",
            instructions=(
                """Extract a structured description of this image for retrieval.
                List every distinct entity (logo, chart, text block, icon, etc.) separately.
                For colors: use descriptive names ('navy blue', 'light gray', 'orange', 'pale yellow'), NEVER hex codes.
                Report the actual perceived color — do not default to black/white when the entity is clearly colored.
                For text entities, report both text color and background color.
                Be precise about shapes.
                Capture all visible text verbatim."""
            ),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the schema."},
                    {"type": "input_image", "file_id": file_id, "detail": "high"},
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


def extract_images_from_zip(file_path, media_prefix, output_folder="images"):

    # docx and pptx are just zip files with images under a media folder
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []

    keep_ext = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

    with zipfile.ZipFile(file_path) as z:

        for name in z.namelist():

            if not name.startswith(media_prefix):
                continue

            # skip vector formats the vision model can't read (emf/wmf)
            if os.path.splitext(name)[1].lower() not in keep_ext:
                continue

            image_bytes = z.read(name)

            image_path = os.path.join(
                output_folder,
                os.path.basename(name)
            )

            # save image
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

            print(f"Saved: {image_path}")

    return image_paths


def extract_and_caption_images(file_path, output_folder="images"):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        image_paths = extract_images_from_pdf(file_path, output_folder)
    elif ext == ".docx":
        image_paths = extract_images_from_zip(file_path, "word/media/", output_folder)
    elif ext == ".pptx":
        image_paths = extract_images_from_zip(file_path, "ppt/media/", output_folder)
    else:
        image_paths = []

    return generate_image_captions(image_paths)


def descriptions_to_serializable(descriptions):
    """Caption results -> plain JSON-able dicts (unpacks the pydantic schema)."""
    return [
        {
            "image_path": d["image_path"],
            "schema": d["schema"].model_dump(),
            "description": d["description"],
        }
        for d in descriptions
    ]


def descriptions_from_serializable(data):
    """Cached JSON -> caption results (rebuilds the pydantic schema)."""
    return [
        {
            "image_path": d["image_path"],
            "schema": ImageSchema(**d["schema"]),
            "description": d["description"],
        }
        for d in data
    ]


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