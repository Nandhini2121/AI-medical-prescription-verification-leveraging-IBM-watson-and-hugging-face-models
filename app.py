ai-prescription-verification/
├─ backend/
│  ├─ app.py               # FastAPI app + endpoints
│  ├─ logic.py             # Core analysis logic
│  ├─ models.py            # Pydantic request/response models
│  ├─ nlp.py               # HF + optional IBM Watson extractors
│  ├─ data/
│  │  ├─ drugs.csv         # Basic drug metadata (class, min/max dose ranges)
│  │  ├─ interactions.csv  # Pairwise interaction rules (severity, rationale)
│  │  └─ alternatives.csv  # Suggested alternatives by class/condition
│  └─ utils.py             # Helpers: loading CSVs, age calc, units
├─ streamlit_app/
│  └─ app.py               # Streamlit UI (calls FastAPI)
├─ tests/
│  └─ test_api.py          # Smoke tests for endpoints
├─ requirements.txt
└─ README.mdfrom _future_ import annotations 
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PatientInfo(BaseModel):
    age: Optional[float] = Field(None, description="Age in years (float). If null, provide dob)")
    dob: Optional[str] = Field(None, description="Date of birth YYYY-MM-DD (optional)")
    weight_kg: Optional[float] = None
    gender: Optional[str] = Field(None, regex=r"^(male|female|other)$", description="Optional")
    pregnancy: Optional[bool] = None
    renal_impairment: Optional[bool] = None
    hepatic_impairment: Optional[bool] = None

class DrugEntry(BaseModel):
    name: str
    dose: Optional[str] = Field(None, description="e.g., '500 mg', '10 mg BID', '5 mL q8h'")
    route: Optional[str] = None
    frequency: Optional[str] = None

class PrescriptionRequest(BaseModel):
    patient: PatientInfo
    drugs: List[DrugEntry]

class Interaction(BaseModel):
    a: str
    b: str
    severity: str
    message: str

class DoseAdvice(BaseModel):
    drug: str
    recommended: Optional[str]
    status: str  # ok|adjust|unknown
    rationale: Optional[str] = None

class AlternativeSuggestion(BaseModel):
    original_drug: str
    suggestions: List[str]
    rationale: Optional[str] = None

class ExtractRequest(BaseModel):
    text: str

class ExtractResponse(BaseModel):
    drugs: List[DrugEntry]
    raw_entities: List[Dict[str, Any]]

class AnalysisResponse(BaseModel):
    interactions: List[Interaction]
    dosing: List[DoseAdvice]
    alternatives: List[AlternativeSuggestion]## backend/utils.py
```python
import csv
import os
from datetime import date, datetime
from typing import Dict, Tuple

DATA_DIR = os.path.join(os.path.dirname(_file_), 'data')

def load_csv(path: str):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

_drug_cache = None
_inter_cache = None
_alt_cache = None

def get_drug_db():
    global _drug_cache
    if _drug_cache is None:
        _drug_cache = load_csv(os.path.join(DATA_DIR, 'drugs.csv'))
    return _drug_cache

def get_interactions_db():
    global _inter_cache
    if _inter_cache is None:
        _inter_cache = load_csv(os.path.join(DATA_DIR, 'interactions.csv'))
    return _inter_cache

def get_alternatives_db():
    global _alt_cache
    if _alt_cache is None:
        _alt_cache = load_csv(os.path.join(DATA_DIR, 'alternatives.csv'))
    return _alt_cache

def parse_age(patient) -> float | None:
    if patient.age is not None:
        return float(patient.age)
    if patient.dob:
        try:
            dob = datetime.strptime(patient.dob, "%Y-%m-%d").date()
            today = date.today()
            return (today - dob).days / 365.25
        except Exception:
            return None
    return Noneimport os
from typing import List, Dict, Any
from .models import DrugEntry

# Hugging Face pipeline for NER (drug names will be matched via dictionary + NER hints)
_HF_MODEL = os.getenv('HF_NER_MODEL', 'dslim/bert-base-NER')

class HFExtractor:
    def _init_(self):
        from transformers import pipeline  # lazy import
        self.pipe = pipeline('ner', model=_HF_MODEL, aggregation_strategy="simple")
        # naive drug lexicon from drugs.csv
        from .utils import get_drug_db
        self.lexicon = {r['name'].lower(): r for r in get_drug_db()}

    def extract(self, text: str) -> dict:
        ents = self.pipe(text)
        drugs: List[DrugEntry] = []
        for token in ents:
            word = token.get('word', '').lower()
            for lex in self.lexicon:
                if lex in word and all(d.name.lower()!=lex for d in drugs):
                    drugs.append(DrugEntry(name=self.lexicon[lex]['name']))
        # quick regex pass for dose/frequency patterns
        import re
        dose_match = re.findall(r"(\d+\s?(mg|g|mcg|mL))", text, flags=re.I)
        freq_match = re.findall(r"(q\d+h|BID|TID|QID|OD|HS)", text, flags=re.I)
        if drugs:
            d = drugs[0]
            if dose_match:
                d.dose = dose_match[0][0]
            if freq_match:
                d.frequency = freq_match[0]
        return {"drugs": [d.dict() for d in drugs], "raw_entities": ents}

class WatsonExtractor:
    def _init_(self):
        # Requires env vars: WATSON_API_KEY, WATSON_URL
        from ibm_watson import NaturalLanguageUnderstandingV1
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
        key = os.getenv('WATSON_API_KEY')
        url = os.getenv('WATSON_URL')
        if not key or not url:
            raise RuntimeError("Watson credentials not set")
        self.service = NaturalLanguageUnderstandingV1(version='2022-04-07',
            authenticator=IAMAuthenticator(key))
        self.service.set_service_url(url)
        from .utils import get_drug_db
        self.lexicon = {r['name'].lower(): r for r in get_drug_db()}

    def extract(self, text: str) -> dict:
        from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
        resp = self.service.analyze(text=text, features=Features(entities=EntitiesOptions(sentiment=False, limit=50))).get_result()
        ents = resp.get('entities', [])
        drugs: List[DrugEntry] = []
        for e in ents:
            word = e.get('text','').lower()
            for lex in self.lexicon:
                if lex in word and all(d.name.lower()!=lex for d in drugs):
                    drugs.append(DrugEntry(name=self.lexicon[lex]['name']))
        return {"drugs": [d.dict() for d in drugs], "raw_entities": ents}


def get_extractor():
    # Try Watson first if configured, else HF
    try:
        if os.getenv('WATSON_API_KEY') and os.getenv('WATSON_URL'):
            return WatsonExtractor()
    except Exception:
        pass
    return HFExtractor()from _future_ import annotations
from typing import List, Dict
from .models import PrescriptionRequest, Interaction, DoseAdvice, AlternativeSuggestion
from .utils import get_interactions_db, get_drug_db, get_alternatives_db, parse_age

# --- Interaction Detection ---

def detect_interactions(drug_names: List[str]) -> List[Interaction]:
    rules = get_interactions_db()
    norm = [d.lower() for d in drug_names]
    out: List[Interaction] = []
    for r in rules:
        a, b = r['drug_a'].lower(), r['drug_b'].lower()
        if a in norm and b in norm:
            out.append(Interaction(a=r['drug_a'], b=r['drug_b'], severity=r['severity'], message=r['message']))
    return out

# --- Age‑Specific Dosing ---

def dosage_recommendations(req: PrescriptionRequest) -> List[DoseAdvice]:
    age = parse_age(req.patient)
    db = get_drug_db()
    meta = {r['name'].lower(): r for r in db}
    results: List[DoseAdvice] = []
    for d in req.drugs:
        key = d.name.lower()
        if key not in meta:
            results.append(DoseAdvice(drug=d.name, recommended=None, status='unknown', rationale='Drug not in local DB'))
            continue
        row = meta[key]
        # Simple rule: choose pediatric/adult/geriatric recommended text
        try:
            if age is None:
                rec = row['adult_dose'] or row['general_dose']
                results.append(DoseAdvice(drug=d.name, recommended=rec, status='ok', rationale='Age unknown; using general/adult guidance'))
            elif age < 12:
                rec = row['peds_dose'] or row['general_dose']
                results.append(DoseAdvice(drug=d.name, recommended=rec, status='adjust', rationale='Pediatric dosing applied'))
            elif age >= 65:
                rec = row['geriatric_dose'] or row['adult_dose'] or row['general_dose']
                results.append(DoseAdvice(drug=d.name, recommended=rec, status='adjust', rationale='Geriatric dosing applied'))
            else:
                rec = row['adult_dose'] or row['general_dose']
                results.append(DoseAdvice(drug=d.name, recommended=rec, status='ok'))
        except KeyError:
            results.append(DoseAdvice(drug=d.name, recommended=None, status='unknown', rationale='Missing dose fields'))
    return results

# --- Alternatives ---

def suggest_alternatives(req: PrescriptionRequest) -> List[AlternativeSuggestion]:
    # Map drug -> class
    meta = {r['name'].lower(): r for r in get_drug_db()}
    alts_db = get_alternatives_db()
    by_class: Dict[str, List[str]] = {}
    for row in alts_db:
        by_class.setdefault(row['drug_class'].lower(), []).append(row['alternative'])
    out: List[AlternativeSuggestion] = []
    for d in req.drugs:
        key = d.name.lower()
        drug_class = meta.get(key, {}).get('class', '')
        if not drug_class:
            out.append(AlternativeSuggestion(original_drug=d.name, suggestions=[], rationale='Class unknown'))
            continue
        suggestions = [a for a in by_class.get(drug_class.lower(), []) if a.lower()!=key]
        out.append(AlternativeSuggestion(original_drug=d.name, suggestions=suggestions, rationale=f"Same class: {drug_class}"))
    return outfrom fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import PrescriptionRequest, AnalysisResponse, ExtractRequest, ExtractResponse
from .logic import detect_interactions, dosage_recommendations, suggest_alternatives
from .nlp import get_extractor

app = FastAPI(title="AI Medical Prescription Verification")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_extractor = get_extractor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    return _extract(req.text)

# internal helper for easier testing

def _extract(text: str):
    data = _extractor.extract(text)
    return data

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: PrescriptionRequest):
    drugs = [d.name for d in req.drugs]
    return AnalysisResponse(
        interactions=detect_interactions(drugs),
        dosing=dosage_recommendations(req),
        alternatives=suggest_alternatives(req)
    )

# Run with: uvicorn backend.app:app --reloadname,class,general_dose,adult_dose,peds_dose,geriatric_dose
Paracetamol,Analgesic,"500 mg q6-8h (max 3g/day)","500-1000 mg q6-8h (max 3-4g/day)","10-15 mg/kg q6-8h (max 60 mg/kg/day)","500 mg q8h (max 3g/day)"
Ibuprofen,NSAID,"200-400 mg q6-8h (max 1200 mg/day OTC)","200-400 mg q6-8h (max 1200 mg/day OTC)","5-10 mg/kg q6-8h (max 40 mg/kg/day)","200 mg q8h (use lowest effective dose)"
Amoxicillin,Antibiotic,"500 mg q8h","500 mg q8h","25-45 mg/kg/day divided q12h","Adjust per renal function"
Aspirin,Antiplatelet,"75-325 mg OD","75-325 mg OD","Not recommended <12 years","Use caution; consider 75 mg OD"drug_a,drug_b,severity,message
Ibuprofen,Aspirin,moderate,NSAIDs may reduce antiplatelet effect of aspirin; separate dosing and monitor.
Paracetamol,Ibuprofen,low,Generally safe together at recommended doses; watch total daily limits.
Amoxicillin,Ibuprofen,low,No significant interaction expected in most patients.
Aspirin,Paracetamol,low,Can be co-administered; caution with chronic high-dose use.drug_class,alternative
Analgesic,Ibuprofen
Analgesic,Naproxen
NSAID,Paracetamol
Antibiotic,Azithromycin
Antiplatelet,Clopidogrelimport json
import os
import requests
import streamlit as st

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title="AI Prescription Verification", page_icon="💊", layout="wide")
st.title("💊 AI Medical Prescription Verification")

with st.sidebar:
    st.header("Backend")
    api = st.text_input("FastAPI URL", API_URL)
    st.caption("Start backend: uvicorn backend.app:app --reload")

st.subheader("1) NLP-Based Drug Info Extraction")
text = st.text_area("Paste prescription/clinical note", height=150, value="Rx: Paracetamol 500 mg q8h and Ibuprofen 200 mg BID for 5 days")
if st.button("Extract"):
    r = requests.post(f"{api}/extract", json={"text": text})
    if r.ok:
        data = r.json()
        st.json(data)
    else:
        st.error(r.text)

st.subheader("2) Analyze Current Prescription")
cols = st.columns(3)
with cols[0]:
    age = st.number_input("Age (years)", min_value=0.0, step=0.5, value=30.0)
with cols[1]:
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.5, value=70.0)
with cols[2]:
    gender = st.selectbox("Gender", ["male","female","other"], index=0)

st.markdown("*Drugs* (add rows)")
if 'rows' not in st.session_state:
    st.session_state.rows = [{"name":"Paracetamol","dose":"500 mg","frequency":"q8h"},{"name":"Ibuprofen","dose":"200 mg","frequency":"BID"}]

def draw_table():
    for i,row in enumerate(st.session_state.rows):
        c1,c2,c3,c4 = st.columns([3,2,2,1])
        row['name'] = c1.text_input(f"Drug {i+1}", row['name'], key=f"name{i}")
        row['dose'] = c2.text_input(f"Dose {i+1}", row['dose'], key=f"dose{i}")
        row['frequency'] = c3.text_input(f"Freq {i+1}", row['frequency'], key=f"freq{i}")
        if c4.button("✖", key=f"del{i}"):
            st.session_state.rows.pop(i)
            st.experimental_rerun()

col_add = st.columns([1,9])
if col_add[0].button("+ Add Drug"):
    st.session_state.rows.append({"name":"","dose":"","frequency":""})

draw_table()

if st.button("Run Analysis"):
    payload = {
        "patient": {"age": age, "weight_kg": weight, "gender": gender},
        "drugs": st.session_state.rows
    }
    r = requests.post(f"{api}/analyze", json=payload)
    if r.ok:
        res = r.json()
        st.success("Analysis complete")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("### Interactions")
            st.table(res['interactions'])
        with c2:
            st.markdown("### Dosing Advice")
            st.table(res['dosing'])
        with c3:
            st.markdown("### Alternatives")
            st.table(res['alternatives'])
    else:
        st.error(r.text)

st.caption("This demo uses small CSV rules shipped with the app. Replace with authoritative sources for clinical use.")from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'

def test_analyze():
    payload = {
        "patient": {"age": 70},
        "drugs": [{"name":"Paracetamol"},{"name":"Ibuprofen"}]
    }
    r = client.post('/analyze', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'interactions' in data and isinstance(data['interactions'], list)