"""Text and severity rules from the original capstone."""

import math
import re

HAZARD_WORDS = {
    "fire", "burn", "smoke", "shock", "explosion", "explode",
    "hazard", "injury", "injuries", "death", "deaths",
    "overheat", "overheating", "choking", "fall", "blade",
    "laceration", "cut", "flame", "sparks", "melt", "melted",
}

# Severity word lists for the Risk Scorer's hazard-signal scoring.
# These are split into three tiers so the risk scorer can tell
# the difference between "it broke" and "it exploded and my child
# was hospitalized." Each tier gets a different weight.

# Critical = most dangerous language (weight: 0.35 per hit)
SEVERITY_CRITICAL = {
    "death", "deaths", "died", "killed", "fatal", "fatality",
    "hospitalized", "hospital", "emergency", "er", "icu",
    "exploded", "explosion", "electrocuted", "electrocution",
    "amputation", "amputated", "suffocated", "suffocation",
    "permanent", "brain", "coma", "unconscious", "seizure",
    "third degree", "second degree",
}

# High severity = serious harm language (weight: 0.15 per hit)
SEVERITY_HIGH = {
    "fire", "burn", "burned", "burns", "smoke", "smoking",
    "flame", "flames", "ignite", "ignited", "caught fire",
    "shock", "shocked", "electrical shock", "sparks",
    "overheat", "overheating", "overheated", "melted", "melt",
    "choking", "choked", "laceration", "fracture", "fractured",
    "stitches", "bleeding", "blood", "concussion", "collapsed",
    "shattered", "broke apart", "fell apart", "came off",
    "child", "children", "infant", "baby", "toddler", "kid",
}

# Moderate severity = general concern language (weight: 0.05 per hit)
SEVERITY_MODERATE = {
    "hazard", "injury", "injuries", "injured", "hurt",
    "cut", "cuts", "bruise", "bruised", "rash", "swelling",
    "pain", "painful", "fell", "fall", "dropped", "broken",
    "defect", "defective", "malfunction", "malfunctioned",
    "recall", "recalled", "warning", "danger", "dangerous",
    "unsafe", "risk", "faulty",
}

# Custom stopwords added on top of sklearn's built-in list.
# These are common words in CPSC reports that don't help the model.
def clean_text(text):
    """
    Steps:
      1. Lowercase everything
      2. Remove URLs
      3. Remove the word "redacted"
      4. Replace dimension patterns (e.g. "4 x 2 x 1") with "dimension"
      5. Replace measurement patterns (e.g. "10 inch") with "measurement"
      6. Remove long numbers (2+ digits)
      7. Keep only letters and spaces
      8. Collapse extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\bredacted\b", " ", text)
    text = re.sub(
        r"\b\d+(\.\d+)?\s*x\s*\d+(\.\d+)?(\s*x\s*\d+(\.\d+)?)?\b",
        " dimension ", text,
    )
    text = re.sub(
        r"\b\d+(\.\d+)?\s*(inch|inches|lb|lbs|pound|pounds|oz|ounces"
        r"|cm|mm|mah|amp|amps|volt|volts|v)\b",
        " measurement ", text,
    )
    text = re.sub(r"\b\d{2,}\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_count(text, vocab):
    """
    Counts how many words from vocab appear in the cleaned text.
    Used to create the hazard-word count features.
    """
    words = clean_text(text).split()
    count = 0
    for w in words:
        if w in vocab:
            count += 1
    return count


def compute_severity_score(text):
    """Return the heuristic severity score and the three tier hit counts."""
    all_tokens = hazard_tokens(text)

    # Count how many words from each severity tier appear
    crit_hits = len(all_tokens & SEVERITY_CRITICAL)
    high_hits = len(all_tokens & SEVERITY_HIGH)
    mod_hits = len(all_tokens & SEVERITY_MODERATE)

    # Weighted sum: critical words count the most
    raw = crit_hits * 0.35 + high_hits * 0.15 + mod_hits * 0.05

    # Cap the score at 1.0
    score = min(1.0, raw)

    return score, crit_hits, high_hits, mod_hits


def hazard_tokens(text):
    words = clean_text(text).split()
    return set(words) | {f"{left} {right}" for left, right in zip(words, words[1:])}


def composite_risk_score(association, severity_score):
    """Blend two unit-interval signals using the original adaptive weights."""
    if not all(math.isfinite(value) and 0 <= value <= 1
               for value in (association, severity_score)):
        raise ValueError("Both score inputs must be finite and between 0 and 1.")
    severity_weight = 0.30 + 0.30 * severity_score
    return (1.0 - severity_weight) * association + severity_weight * severity_score
