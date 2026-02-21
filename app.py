import streamlit as st
from utils.nlp_engine import extract_symptoms
from utils.predictor import predict_disease, symptom_list
from utils.knowledge_base import get_disease_info
from deep_translator import GoogleTranslator


st.set_page_config(page_title="AI Healthcare Chatbot", layout="wide")

st.title("🏥 AI-Powered Healthcare Assistant")

menu = st.sidebar.selectbox("Navigation", ["Disease Prediction", "About Project"])

if menu == "Disease Prediction":

    user_input = st.text_area("Describe your symptoms:")

    language = st.selectbox("Select Output Language", ["en", "hi", "mr", "fr"])

    if st.button("Analyze"):

        extracted = extract_symptoms(user_input, symptom_list)

        if len(extracted) == 0:
            st.warning("No symptoms detected. Please try again.")
        else:
            prognosis, confidence = predict_disease(extracted)
            description, precautions = get_disease_info(prognosis)

            translated_desc = GoogleTranslator(source='auto', target=language).translate(description)
            st.success(f"Predicted Disease: {prognosis}")
            st.info(f"Confidence Level: {confidence}%")
            st.write("### Description")
            st.write(translated_desc)

            st.write("### Precautions")
            for p in precautions:
                if p != "":
                    translated_p = GoogleTranslator(source='auto', target=language).translate(p)
                    st.write("- ", translated_p)

elif menu == "About Project":
    st.write("""
    This is an Advanced AI-Powered Healthcare Chatbot developed
    using Machine Learning, NLP, and Streamlit.
    It predicts diseases based on symptoms and provides medical guidance.
    """)
