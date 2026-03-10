import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Spam Detection App", page_icon="📧")

# Title
st.title("📧 Email / SMS Spam Detection")
st.write("Enter a message and the model will predict whether it is Spam or Not Spam.")

st.markdown("---")

# Text input
message = st.text_area("Enter your message")

# Predict button
if st.button("Predict"):

    if message.strip() != "":

        # Transform text
        message_vector = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

    else:
        st.warning("Please enter a message first.")
