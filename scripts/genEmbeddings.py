import pandas as pd
import numpy as np
from openai import OpenAI
import dotenv
import os

def get_embedding(review, client, model="text-embedding-ada-002"):
    return client.embeddings.create(input = [review], model=model).data[0].embedding

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def generate_embeddings(dataset):
    # load environment variables
    dotenv.load_dotenv()

    # initialize OpenAI client for API calls
    client = OpenAI(
        api_key = os.getenv('OPENAI_API_KEY'),
    )
    # initialize df with dataset
    df = pd.read_csv(dataset)

    # PREPROCESSING

    # Generate embeddings
    df['embedding'] = df["content"].apply(lambda x : get_embedding(x, client=client))
    # normalize embeddings
    df['normalized_embedding'] = df['embedding'].apply(normalize_embedding)

    # Store in csv
    #df.to_csv('embedded_reviews.csv', index=False)

    print("Step 1 (Embedding Generation): Successful!")

    return df