import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data = pd.read_csv(BASE_DIR / "data" / "properties.csv")

data["Property ID"] = data["Property ID"].astype(str)

question = input("Ask your question: ")
words = question.split()

property_id = None

for word in words:
    word = word.strip("?!.,")
    if word in data["Property ID"].values:
        property_id = word
        break

print("Property ID:", property_id)

if property_id:
    property_row = data[data["Property ID"] == property_id]

    if "rent" in question.lower() or "kiraya" in question.lower():
        rent = property_row["Rent"].values[0]
        print("Rent:", rent)

    elif "available" in question.lower():
        availability = property_row["Available"].values[0]
        print("Available:", availability)

    else:
        print(property_row)

else:
    print("Property ID not found.")