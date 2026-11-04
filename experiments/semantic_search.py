from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from experiments.intent_examples import intent_examples

BASE_DIR = Path(__file__).resolve().parent.parent
data = pd.read_csv(
    BASE_DIR / "data" / "properties.csv",
    dtype={"Property ID": str}
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

all_examples = []
all_intents = []

for intent, examples in intent_examples.items():
    for example in examples:
        all_examples.append(example)
        all_intents.append(intent)

example_embeddings = model.encode(all_examples, convert_to_tensor=True)

question = input("Ask a question: ")
question_embedding = model.encode(question, convert_to_tensor=True)

scores = util.cos_sim(question_embedding, example_embeddings)[0]
best_match = scores.argmax().item()

if scores[best_match].item() < 0.4:
    print("Sorry, I can only answer questions about properties.")
    exit()

detected_intent = all_intents[best_match]

print("Detected intent:", detected_intent)

if detected_intent == "SEARCH_PROPERTIES":
    question_lower = question.lower()
    area = None

    for location in data["Area"].unique():
        if location.lower() in question_lower:
            area = location
            break

    if area is None:
        print("Which area are you looking for?")
    else:
        results = data[
            (data["Area"].str.lower() == area.lower()) &
            (data["Available"].str.lower() == "yes")
        ]

        if results.empty:
            print("No available properties found in", area)
        else:
            print(results[["Property ID", "Area", "Rent", "Available"]].to_string(index=False))

else:
    words = question.split()
    property_id = None

    for word in words:
        word = word.strip("?!.,;:").upper()
        if word in data["Property ID"].values:
            property_id = word
            break

    if property_id is None:
        print("Which property are you asking about? Please provide an ID, such as 101.")
        property_id = input("Property ID: ").strip().upper()

        while property_id not in data["Property ID"].values:
            print("Invalid property ID. Please try again.")
            property_id = input("Property ID: ").strip().upper()

    property_row = data[data["Property ID"] == property_id]

    if detected_intent == "GET_RENT":
        print("Rent:", property_row["Rent"].values[0])

    elif detected_intent == "CHECK_AVAILABILITY":
        print("Available:", property_row["Available"].values[0])

    else:
        print("Sorry, I don't know how to answer this question yet.")

print("Similarity score:", round(scores[best_match].item(), 3))
print("Matched example:", all_examples[best_match])