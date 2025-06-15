from flask import Flask, request, jsonify
import pandas as pd
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FastText model
model = fasttext.load_model("fasttext_model.bin")

# Load CSV
df = pd.read_csv("Final_Drugs.csv")
df = df[df['tokenized_all_active_ingredients'].notnull()]

# Define weights
weights = {
    "name": 1.0,
    "company": 0.2,
    "dosage_form": 0.7,
    "description": 0.7,
    "entire_comp": 5.0
}

# Embedding extraction
def get_fasttext_embedding(tokens, model):
    embeddings = [model.get_word_vector(token) for token in tokens if token in model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.get_dimension())

# Compute weighted embeddings
def get_weighted_embedding(row, model, weights):
    name_embedding = get_fasttext_embedding(row['tokenized_name'], model) * weights['name']
    company_embedding = get_fasttext_embedding(row['tokenized_company'], model) * weights['company']
    dosage_form_embedding = get_fasttext_embedding(row['tokenized_dosage_form'], model) * weights['dosage_form']
    description_embedding = get_fasttext_embedding(row['tokenized_description'], model) * weights['description']
    entire_comp_embedding = get_fasttext_embedding(row['tokenized_all_active_ingredients'], model) * weights['entire_comp']
    return name_embedding + company_embedding + dosage_form_embedding + description_embedding + entire_comp_embedding

# Precompute all embeddings
drug_embeddings = {
    index: get_weighted_embedding(row, model, weights)
    for index, row in df.iterrows()
}

# API route
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    drug_name = data.get("drug_name", "").strip().lower()
    
    matched = df[df["name"].str.lower() == drug_name]
    if matched.empty:
        return jsonify({"error": "Drug not found"}), 404

    input_embedding = get_weighted_embedding(matched.iloc[0], model, weights)
    all_embeddings = np.array(list(drug_embeddings.values()))
    similarities = cosine_similarity([input_embedding], all_embeddings)
    
    top_indices = similarities[0].argsort()[-6:][::-1]
    similar_names = df.iloc[top_indices]['name'].tolist()
    similar_names = [name for name in similar_names if name.lower() != drug_name]
    
    return jsonify({"recommendations": similar_names[:5]})

@app.route("/", methods=["GET"])
def index():
    return "Drug Recommender API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
