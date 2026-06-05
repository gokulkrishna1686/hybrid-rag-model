import spacy
_SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(_SPACY_MODEL)

text = """
This section is intended for internal circulation among managers and the people team. The engineering review panel was chaired by Rajesh Menon, with support from Anita Desai and Karthik Iyer, who coordinated calibration across the Bangalore and Chennai offices. The product team review was led by Sandra Pereira, working alongside Vikram Nair at the Pune office, while the design team feedback was consolidated by Meera Krishnan in the Hyderabad office. 
"""


def extract_entities(text):
    """spaCy NER -> a de-duplicated list of entity strings."""
    doc = nlp(text)
    print("Docs: ", doc)
    print("DOC Ents", doc.ents)
    entities = []
    for ent in doc.ents:
        if ent.text not in entities:
            entities.append(ent.text)
    print("RETURN")
    return entities

print(extract_entities(text))