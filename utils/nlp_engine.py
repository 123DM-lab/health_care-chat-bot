import nltk

# Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize

def extract_symptoms(user_input, symptom_list):
    tokens = word_tokenize(user_input.lower())
    extracted = []

    for symptom in symptom_list:
        symptom_words = symptom.split("_")
        if all(word in tokens for word in symptom_words):
            extracted.append(symptom)

    return extracted
