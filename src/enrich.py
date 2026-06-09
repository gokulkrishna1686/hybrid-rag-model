"""
Chunk enrichment: entity extraction (spaCy) + PII / sensitivity (Presidio).

Two local libraries, two jobs:
  - spaCy    -> general named entities (people, orgs, dates, money...) used for
                filtered ("must mention X") search.
  - Presidio -> detect PII, which we map to a sensitivity tier used for
                role-based access control (RBAC) and redaction.

Everything runs locally and is free (no API calls).
"""

import spacy

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


# ---------------------------------------------------------------------------
# Load the models ONCE at import time and reuse them for every chunk.
# ---------------------------------------------------------------------------

# Small English model (~12 MB). Used for BOTH spaCy NER and Presidio so we
# only ever download one model.
_SPACY_MODEL = "en_core_web_sm"

nlp = spacy.load(_SPACY_MODEL)

# Tell Presidio to reuse the same small model instead of its default large one.
_provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": _SPACY_MODEL}],
})
analyzer = AnalyzerEngine(
    nlp_engine=_provider.create_engine(),
    supported_languages=["en"],
)

# Presidio's default recognizers (with the small model) miss dashed SSN / ID numbers
# like 123-45-6789, which would otherwise survive redaction. Add an explicit pattern so
# they are both classified (-> confidential) AND masked by redact().
analyzer.registry.add_recognizer(PatternRecognizer(
    supported_entity="US_SSN",
    name="dashed_ssn_recognizer",
    patterns=[Pattern(name="ssn_dashed", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.85)],
    context=["ssn", "social security", "identification number"],
))

anonymizer = AnonymizerEngine()


# ---------------------------------------------------------------------------
# Sensitivity tiers + RBAC role mapping
# ---------------------------------------------------------------------------

# Ordered tiers: a higher clearance can see its own tier AND everything below.
SENSITIVITY_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
}

# RBAC: a user's role decides which tier they are cleared to see.
# One role per tier, mapped 1:1, so the UI shows exactly three levels.
ROLE_CLEARANCE = {
    "guest": "public",
    "employee": "internal",
    "manager": "confidential",
}

# Which Presidio PII types push a chunk into which tier.
STRONG_PII = {
    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE",
    "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_PASSPORT", "US_ITIN",
    "MEDICAL_LICENSE", "CRYPTO", "IP_ADDRESS",
}
MILD_PII = {"PERSON", "LOCATION", "NRP"}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_entities(text):
    """spaCy NER -> a de-duplicated list of entity strings."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.text not in entities:
            entities.append(ent.text)
    return entities


def detect_pii(text):
    """Presidio -> the sorted list of PII entity types found in the text."""
    results = analyzer.analyze(text=text, language="en")
    return sorted({r.entity_type for r in results})


def classify_sensitivity(pii_types):
    """Map the detected PII types to one of the three tiers."""
    if any(t in STRONG_PII for t in pii_types):
        return "confidential"
    if any(t in MILD_PII for t in pii_types):
        return "internal"
    return "public"


def enrich_text(text):
    """All the metadata we attach to a chunk: entities + PII + sensitivity."""
    pii_types = detect_pii(text)
    return {
        "entities": extract_entities(text),
        "pii_types": pii_types,
        "has_pii": len(pii_types) > 0,
        "sensitivity": classify_sensitivity(pii_types),
    }


def redact(text):
    """Mask every PII span Presidio finds (e.g. <PERSON>, <EMAIL_ADDRESS>)."""
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def clearance_level(role):
    """Role -> numeric clearance (higher number = sees more)."""
    tier = ROLE_CLEARANCE.get(role, "public")
    return SENSITIVITY_ORDER[tier]
