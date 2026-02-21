import numpy as np
import pickle

model = pickle.load(open("models/best_model.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))
symptom_list = pickle.load(open("models/symptom_list.pkl", "rb"))

def predict_disease(symptoms):
    input_data = np.zeros(len(symptom_list))

    for symptom in symptoms:
        if symptom in symptom_list:
            index = symptom_list.index(symptom)
            input_data[index] = 1

    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])

    prognosis = le.inverse_transform(prediction)[0]
    confidence = round(np.max(probability) * 100, 2)

    return prognosis, confidence
