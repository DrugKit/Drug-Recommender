from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Egyptian Drug Recommender API")

# Load FastText model
model = fasttext.load_model("fasttext_model.bin")

# Load dataset
df = pd.read_csv("Final_Drugs.csv")
df = df[df['All Active Ingredients'].notnull()]

# Tokenize all relevant columns
for col in ['name', 'company', 'dosage_form', 'description', 'All Active Ingredients']:
    df[f'tokenized_{col.lower().replace(" ", "_")}'] = df[col].fillna("").str.lower().str.split()

# Define weights
weights = {
    "name": 1.0,
    "company": 0.2,
    "dosage_form": 0.7,
    "description": 0.7,
    "entire_comp": 5.0
}

# Embedding function
def get_fasttext_embedding(tokens, model):
    embeddings = [model.get_word_vector(token) for token in tokens if token in model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.get_dimension())

# Compute weighted embedding
def get_weighted_embedding(row, model, weights):
    return (
        get_fasttext_embedding(row['tokenized_name'], model) * weights['name'] +
        get_fasttext_embedding(row['tokenized_company'], model) * weights['company'] +
        get_fasttext_embedding(row['tokenized_dosage_form'], model) * weights['dosage_form'] +
        get_fasttext_embedding(row['tokenized_description'], model) * weights['description'] +
        get_fasttext_embedding(row['tokenized_all_active_ingredients'], model) * weights['entire_comp']
    )

# Precompute dataset embeddings
drug_embeddings = {index: get_weighted_embedding(row, model, weights) for index, row in df.iterrows()}

# Request body schema
class DrugRequest(BaseModel):
    name: str
    company: str
    dosage_form: str
    description: str
    all_active_ingredients: str

@app.post("/recommend/")
def recommend_drugs(request: DrugRequest, top_n: int = 5):
    # Tokenize input fields
    row = {
        'tokenized_name': request.name.lower().split(),
        'tokenized_company': request.company.lower().split(),
        'tokenized_dosage_form': request.dosage_form.lower().split(),
        'tokenized_description': request.description.lower().split(),
        'tokenized_all_active_ingredients': request.all_active_ingredients.lower().split()
    }

    # Compute embedding and similarity
    input_embedding = get_weighted_embedding(row, model, weights)
    all_embeddings = np.array(list(drug_embeddings.values()))
    similarities = cosine_similarity([input_embedding], all_embeddings)

    # Get top matches
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    recommended_names = df.iloc[top_indices]['name'].tolist()

    return {"recommendations": recommended_names}
