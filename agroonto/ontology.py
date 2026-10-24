"""
Multi-faceted ontology similarity computation.
Seven facets: crop, organ, pathogen_type, order, family, genus, symptom.
Exclusive scoring at taxonomic level with normalisation.
"""
import numpy as np
import torch

# ── Facet weights (exclusive at taxonomic level) ───────────────────────
FACET_WEIGHTS = {
    "genus":         0.90,
    "family":        0.72,
    "order":         0.48,
    "pathogen_type": 0.55,
    "crop":          0.40,
    "symptom":       0.25,
}
NORM_FACTOR = 1.60   # normalisation denominator


# ═══════════════════════════════════════════════════════════════════════
#  PlantVillage ontology  (38 classes: Crop___Disease)
# ═══════════════════════════════════════════════════════════════════════
def _parse_plantvillage_class(name):
    """Extract crop, disease, pathogen type, and symptom from folder name."""
    parts = name.split("___")
    crop = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "healthy"

    # Pathogen type mapping
    bacterial = ["Bacterial spot", "bacterial spot", "Haunglongbing", "Citrus greening"]
    fungal = ["scab", "rot", "rust", "Powdery mildew", "mildew", "blight",
              "Cercospora", "leaf spot", "Gray leaf spot", "Esca", "Black Measles",
              "Isariopsis", "Late blight", "Early blight", "Leaf Mold",
              "Septoria", "Target Spot", "Spider mites", "mold"]
    viral = ["Leaf scorch", "mosaic", "Yellow Leaf Curl", "yellowing"]

    ptype = "healthy"
    if disease.lower() != "healthy":
        d_lower = disease.lower()
        if any(k.lower() in d_lower for k in bacterial):
            ptype = "bacterial"
        elif any(k.lower() in d_lower for k in viral):
            ptype = "viral"
        else:
            ptype = "fungal"

    # Symptom extraction
    symptom_keywords = ["spot", "rot", "rust", "blight", "mildew", "scab",
                        "mold", "scorch", "curl", "mosaic", "measles", "greening"]
    symptom = "none"
    for kw in symptom_keywords:
        if kw in disease.lower():
            symptom = kw
            break

    return {"crop": crop, "pathogen_type": ptype, "symptom": symptom}


# ═══════════════════════════════════════════════════════════════════════
#  DeepWeeds ontology  (9 classes: 8 weed species + Negative)
# ═══════════════════════════════════════════════════════════════════════
DEEPWEEDS_TAXONOMY = {
    0: {"species": "Chinee apple",   "genus": "Ziziphus",    "family": "Rhamnaceae",     "order": "Rosales"},
    1: {"species": "Lantana",        "genus": "Lantana",     "family": "Verbenaceae",    "order": "Lamiales"},
    2: {"species": "Parkinsonia",    "genus": "Parkinsonia", "family": "Fabaceae",       "order": "Fabales"},
    3: {"species": "Parthenium",     "genus": "Parthenium",  "family": "Asteraceae",     "order": "Asterales"},
    4: {"species": "Prickly acacia", "genus": "Vachellia",   "family": "Fabaceae",       "order": "Fabales"},
    5: {"species": "Rubber vine",    "genus": "Cryptostegia","family": "Apocynaceae",    "order": "Gentianales"},
    6: {"species": "Siam weed",      "genus": "Chromolaena", "family": "Asteraceae",     "order": "Asterales"},
    7: {"species": "Snake weed",     "genus": "Stachytarpheta","family": "Verbenaceae",  "order": "Lamiales"},
    8: {"species": "Negative",       "genus": "none",        "family": "none",           "order": "none"},
}


# ═══════════════════════════════════════════════════════════════════════
#  IP102 ontology  (102 classes grouped by insect order/family)
# ═══════════════════════════════════════════════════════════════════════
#  Simplified taxonomy: group the 102 pest classes by order and family.
#  IP102 classes 0-101 are ordered by their taxonomic grouping in the
#  original dataset. We assign order/family based on published IP102 docs.
IP102_TAXONOMY = {}

# Hemiptera — plant-sucking bugs (classes 0–17)
_hemiptera_families = {
    range(0, 5):   "Delphacidae",      # planthoppers
    range(5, 9):   "Cicadellidae",     # leafhoppers
    range(9, 14):  "Aphididae",        # aphids
    range(14, 18): "Pentatomidae",     # stink bugs
}
for rng, fam in _hemiptera_families.items():
    for c in rng:
        IP102_TAXONOMY[c] = {"order": "Hemiptera", "family": fam, "genus": f"hem_{c}"}

# Lepidoptera — moths and butterflies (classes 18–44)
_lepidoptera_families = {
    range(18, 25): "Pyralidae",        # snout moths
    range(25, 32): "Noctuidae",        # owlet moths
    range(32, 38): "Crambidae",        # grass moths
    range(38, 45): "Geometridae",      # geometer moths
}
for rng, fam in _lepidoptera_families.items():
    for c in rng:
        IP102_TAXONOMY[c] = {"order": "Lepidoptera", "family": fam, "genus": f"lep_{c}"}

# Coleoptera — beetles (classes 45–69)
_coleoptera_families = {
    range(45, 55): "Chrysomelidae",    # leaf beetles
    range(55, 63): "Curculionidae",    # weevils
    range(63, 70): "Cerambycidae",     # longhorn beetles
}
for rng, fam in _coleoptera_families.items():
    for c in rng:
        IP102_TAXONOMY[c] = {"order": "Coleoptera", "family": fam, "genus": f"col_{c}"}

# Diptera — flies (classes 70–80)
for c in range(70, 81):
    IP102_TAXONOMY[c] = {"order": "Diptera", "family": "Agromyzidae", "genus": f"dip_{c}"}

# Thysanoptera — thrips (classes 81–88)
for c in range(81, 89):
    IP102_TAXONOMY[c] = {"order": "Thysanoptera", "family": "Thripidae", "genus": f"thy_{c}"}

# Acari — mites (classes 89–95)
for c in range(89, 96):
    IP102_TAXONOMY[c] = {"order": "Acari", "family": "Tetranychidae", "genus": f"aca_{c}"}

# Orthoptera — grasshoppers/locusts (classes 96–101)
for c in range(96, 102):
    IP102_TAXONOMY[c] = {"order": "Orthoptera", "family": "Acrididae", "genus": f"ort_{c}"}


# Corrected taxonomy (class order taken from ip102.yaml); overrides the index-range version above.
from .ip102_taxonomy import IP102_TAXONOMY  # noqa: E402,F811

# ═══════════════════════════════════════════════════════════════════════
#  Similarity matrix builder
# ═══════════════════════════════════════════════════════════════════════
def compute_similarity_matrix(dataset_name, class_names=None, num_classes=None):
    """
    Build K×K ontology similarity matrix.
    Uses multi-facet scoring with off-diagonal row-normalisation (no temperature softmax).

    Returns: torch.Tensor of shape (K, K), float32
    """
    if dataset_name in ("PlantVillage", "PlantVillageTomato"):
        assert class_names is not None
        K = len(class_names)
        attrs = [_parse_plantvillage_class(n) for n in class_names]
        S = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    S[i, j] = 1.0
                    continue
                score = 0.0
                # crop facet
                if attrs[i]["crop"] == attrs[j]["crop"]:
                    score += FACET_WEIGHTS["crop"]
                # pathogen_type facet
                if (attrs[i]["pathogen_type"] == attrs[j]["pathogen_type"]
                        and attrs[i]["pathogen_type"] != "healthy"):
                    score += FACET_WEIGHTS["pathogen_type"]
                # symptom facet
                if (attrs[i]["symptom"] == attrs[j]["symptom"]
                        and attrs[i]["symptom"] != "none"):
                    score += FACET_WEIGHTS["symptom"]
                S[i, j] = min(score / NORM_FACTOR, 1.0)

    elif dataset_name == "DeepWeeds":
        K = num_classes or 9
        S = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    S[i, j] = 1.0
                    continue
                ti = DEEPWEEDS_TAXONOMY.get(i, {})
                tj = DEEPWEEDS_TAXONOMY.get(j, {})
                if ti.get("genus") == "none" or tj.get("genus") == "none":
                    S[i, j] = 0.0
                    continue
                score = 0.0
                # Exclusive taxonomic: genus > family > order
                if ti["genus"] == tj["genus"]:
                    score += FACET_WEIGHTS["genus"]
                elif ti["family"] == tj["family"]:
                    score += FACET_WEIGHTS["family"]
                elif ti["order"] == tj["order"]:
                    score += FACET_WEIGHTS["order"]
                S[i, j] = min(score / NORM_FACTOR, 1.0)

    elif dataset_name == "IP102":
        K = num_classes or 102
        S = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    S[i, j] = 1.0
                    continue
                ti = IP102_TAXONOMY.get(i, {})
                tj = IP102_TAXONOMY.get(j, {})
                if not ti or not tj:
                    continue
                score = 0.0
                if ti["genus"] == tj["genus"]:
                    score += FACET_WEIGHTS["genus"]
                elif ti["family"] == tj["family"]:
                    score += FACET_WEIGHTS["family"]
                elif ti["order"] == tj["order"]:
                    score += FACET_WEIGHTS["order"]
                S[i, j] = min(score / NORM_FACTOR, 1.0)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Off-diagonal row normalisation (no temperature softmax)
    for i in range(S.shape[0]):
        row_sum = S[i].sum() - S[i, i]
        if row_sum > 0:
            for j in range(S.shape[1]):
                if i != j:
                    S[i, j] /= row_sum

    return torch.from_numpy(S)
