import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App
st.title("Restaurant Review Sentiment Analysis")
st.write("Enter your restaurant review below, and we'll predict if it's positive or negative!")

# User input
user_input = st.text_input("Enter your review:")

if st.button("Analyze"):
    if user_input.strip():
        # Vectorize the input review
        user_input_vectorized = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(user_input_vectorized)[0]
        result = "Good Review 😊" if prediction == 1 else "Bad Review 😞"
        
        # Display result
        st.subheader("Prediction:")
        st.write(f"Sentiment: {result}")
    else:
        st.write("Please enter a valid review.")

# Footer

st.markdown("---")
st.subheader("Tech Stack Used:")
st.write("""
- **Programming Language:** Python
- **Libraries:** Streamlit, Scikit-learn, Pandas, NumPy
- **Machine Learning Model:** Multinomial Naive Bayes
- **Vectorization Technique:** CountVectorizer
- **Deployment:** Streamlit
""")
