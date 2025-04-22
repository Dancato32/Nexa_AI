import joblib

# Load the saved model
model = joblib.load('sentiment_classifier.pkl')

# Try out your model
sentences = [
"Define an application software",
"What is a computer virus",
" What is a computer software",
"what is a noun",
"Who is a citizen",
]

for sentence in sentences:
    prediction = model.predict([sentence])[0]
    print(f"'{sentence}' => {prediction}")


# from transformers import AutoTokenizer

# tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

# sentence="I love to make a differnce in the world through my skills"

# tokens=tokenizer.tokenize(sentence)

# print('Token:',tokens)

# input_ids = tokenizer.convert_tokens_to_ids(tokens)
# print("Input IDs:", input_ids)


# encoded = tokenizer(sentence)
# print("Encoded Output:", encoded)
