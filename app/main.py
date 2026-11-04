
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
torch.set_num_threads(1)
import re
from sentence_transformers import SentenceTransformer, util
from experiments.intent_examples import intent_examples
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Voice Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and property dataset
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
data = pd.read_csv("data/properties.csv")

# Prepare intent examples
all_examples = []
all_intents = []

for intent, examples in intent_examples.items():
    for example in examples:
        all_examples.append(example)
        all_intents.append(intent)

example_embeddings = model.encode(
    all_examples,
    convert_to_tensor=True,
    batch_size=4
)


class Question(BaseModel):
    text: str


def detect_language(text):
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def make_answer(kind, language, **kwargs):
    hindi = language == "hi"

    if kind == "unsupported":
        return (
            "माफ़ कीजिए, मैं केवल प्रॉपर्टी से जुड़े सवालों के जवाब दे सकता हूँ।"
            if hindi else
            "Sorry, I can only answer questions about properties."
        )

    if kind == "area_prompt":
        return (
            "आप किस इलाके में प्रॉपर्टी ढूँढ रहे हैं?"
            if hindi else
            "Which area are you looking for?"
        )

    if kind == "property_id":
        return (
            "कृपया प्रॉपर्टी ID बताएँ, जैसे 104।"
            if hindi else
            "Please include a property ID, such as 104."
        )

    if kind == "not_found":
        area = kwargs["area"]
        return (
            f"{area} में कोई उपलब्ध प्रॉपर्टी नहीं मिली।"
            if hindi else
            f"No available properties found in {area}."
        )

    if kind == "rent":
        property_id = kwargs["property_id"]
        rent = kwargs["rent"]
        return (
            f"{property_id} का किराया ₹{rent} प्रति माह है।"
            if hindi else
            f"Rent: ₹{rent} per month."
        )

    if kind == "availability":
        property_id = kwargs["property_id"]
        available = kwargs["available"]

        if hindi:
            status = (
                "उपलब्ध है"
                if available.lower() == "yes"
                else "उपलब्ध नहीं है"
            )
            return f"{property_id} {status}।"

        return f"Available: {available}."

    if kind == "results":
        results = kwargs["results"]
        area = kwargs["area"]

        if hindi:
            descriptions = [
                f"{row['Property ID']}, किराया ₹{row['Rent']} प्रति माह"
                for _, row in results.iterrows()
            ]
            return (
                f"{area} में उपलब्ध प्रॉपर्टीज़: "
                + "। ".join(descriptions) + "।"
            )

        return results[
            ["Property ID", "Area", "Rent", "Available"]
        ].to_dict(orient="records")


@app.get("/")
def home():
    return {"message": "Voice Chatbot API is running"}


@app.post("/ask")
def ask_question(question: Question):
    text = question.text.strip()
    language = detect_language(text)

    # Find the user's intent
    question_embedding = model.encode(
        text,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        question_embedding,
        example_embeddings
    )[0]

    best_match = scores.argmax().item()

    if scores[best_match].item() < 0.4:
        return {
            "answer": make_answer("unsupported", language),
            "language": language,
        }

    detected_intent = all_intents[best_match]

    # Extract numeric property ID
    property_ids = set(
        data["Property ID"].astype(str).str.strip()
    )

    property_id = next(
        (
            match.group()
            for match in re.finditer(r"\b\d{3}\b", text)
            if match.group() in property_ids
        ),
        None
    )

    # Search available properties by area
    if detected_intent == "SEARCH_PROPERTIES":
        area = next(
            (
                location
                for location in data["Area"].dropna().unique()
                if str(location).lower() in text.lower()
            ),
            None
        )

        if area is None:
            return {
                "answer": make_answer("area_prompt", language),
                "language": language,
            }

        results = data[
            (data["Area"].str.lower() == str(area).lower())
            & (data["Available"].astype(str).str.lower() == "yes")
        ]

        if results.empty:
            return {
                "answer": make_answer(
                    "not_found",
                    language,
                    area=area
                ),
                "language": language,
            }

        return {
            "answer": make_answer(
                "results",
                language,
                area=area,
                results=results
            ),
            "language": language,
        }

    # Require a property ID for individual property questions
    if property_id is None:
        return {
            "answer": make_answer("property_id", language),
            "language": language,
        }

    matching_rows = data[
        data["Property ID"].astype(str).str.strip() == property_id
    ]

    if matching_rows.empty:
        return {
            "answer": make_answer("property_id", language),
            "language": language,
        }

    property_row = matching_rows.iloc[0]

    # Answer rent questions
    if detected_intent == "GET_RENT":
        answer = make_answer(
            "rent",
            language,
            property_id=property_id,
            rent=property_row["Rent"]
        )

    # Answer availability questions
    elif detected_intent == "CHECK_AVAILABILITY":
        answer = make_answer(
            "availability",
            language,
            property_id=property_id,
            available=str(property_row["Available"])
        )

    else:
        answer = make_answer("unsupported", language)

    return {
        "answer": answer,
        "language": language
    }
