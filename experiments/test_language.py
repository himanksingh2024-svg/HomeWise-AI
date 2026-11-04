
from lingua import Language, LanguageDetectorBuilder
import re


detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.HINDI
).build()


def detect_language(question):
    # Devanagari script is a strong indicator of Hindi
    if re.search(r"[\u0900-\u097F]", question):
        return "HINDI"

    # Lingua works well for standard English
    language = detector.detect_language_of(question)

    # Romanized Hinglish can be misclassified as English.
    # Use the language confidence as a starting point.
    confidence = detector.compute_language_confidence_values(question)
    english_confidence = next(
        (item.value for item in confidence
         if item.language == Language.ENGLISH),
        0.0
    )

    if language == Language.ENGLISH and english_confidence >= 0.90:
        return "ENGLISH"

    return "HINGLISH"


questions = [
    "What is the rent of P104?",
    "Bhai P104 ka rent kitna hai?",
    "Tell me P104 ka kiraya, bhai.",
    "Noida mein kaun se ghar available hain?",
    "P104 का किराया कितना है?"
]

for question in questions:
    print(f"Question: {question}")
    print(f"Detected: {detect_language(question)}")
    print("-" * 40)
