import joblib

def load_model_and_predict():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    print("Model Classes:", model.classes_)
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter a sentence: ")
        
        if user_input.lower() == 'exit':
            break
            
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        probabilities = model.predict_proba(input_vec)[0]
        
        print(f"Raw Prediction: {prediction[0]}")
        print(f"Probabilities: {dict(zip(model.classes_, probabilities))}")
        
        if str(prediction[0]) in ['1', '4']:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        print(f"Predicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    load_model_and_predict()