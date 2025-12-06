import streamlit as st
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("best_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))




spam_keywords = [
    "approved", "loans", "Visit", "viagra", "lottery", "win", "cash", "gift","verify", "risk", "offer", "exclusive", "%", "click", "info", "free coupon", "free card", "free gift", "Contact us", "reward"
]



def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", lambda m: m.group(0).replace("www.","").replace(".com",""), text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sms(sms, model, tfidf):
    cleaned_sms = clean_text(sms)
    
    cleaned_sms_words = cleaned_sms.lower().split()
    matched_keywords = [kw for kw in spam_keywords if any(kw in word for word in cleaned_sms_words)]
    if matched_keywords:
        return "Spam", matched_keywords
    
    vector = tfidf.transform([cleaned_sms])
    pred = model.predict(vector)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vector)[0][1]  
    return ("Spam" if pred[0] == 1 else "Ham"), matched_keywords


st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="wide")
st.title("SMS Spam Detection")
st.markdown(
    "This app detects **Spam vs Ham** SMS messages using a hybrid approach: **ML model + keyword rules**. "
    "Messages flagged with spam keywords are highlighted."
)

user_input = st.text_area("Enter your SMS message here:", height=150)

bulk_input = st.checkbox("Test multiple SMS at once (one per line)")

if bulk_input:
    user_input = st.text_area("Enter multiple SMS messages (one per line):", height=300)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter at least one SMS message!")
    else:
        sms_list = user_input.strip().split("\n") if bulk_input else [user_input]
        results = []

        for sms in sms_list:
            label, matched_keywords = predict_sms(sms, model, tfidf)
            results.append({
                "SMS": sms,
                "Prediction": label
            })

        df_results = pd.DataFrame(results)

        def highlight_prediction(val):
            color = "#ff4b4b" if val == "Spam" else "#28a745"
            return f"background-color: {color}; color: white; font-weight: bold;"

        st.markdown("### Prediction Results")
        st.dataframe(df_results.style.applymap(highlight_prediction, subset=["Prediction"]), height=400)

        if bulk_input and len(results) > 1:
            spam_count = sum(1 for r in results if r["Prediction"] == "Spam")
            ham_count = sum(1 for r in results if r["Prediction"] == "Ham")
            st.markdown("### Summary Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Total Spam Messages", spam_count)
            col2.metric("Total Ham Messages", ham_count)

        st.success("Predictions completed!")
