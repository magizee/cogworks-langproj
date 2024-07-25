import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import pickle
from numpy.linalg import norm

# Function to download an image
def download_image(img_url: str) -> Image:
    response = requests.get(img_url)
    return Image.open(BytesIO(response.content))

# Step 1: Convert Image Descriptors to Embeddings
def convert_to_embeddings(descriptors, embedding_matrix):
    return np.dot(descriptors, embedding_matrix)

# Step 2: Create the Image Database
def create_image_database(image_ids, embeddings):
    return {image_id: embedding for image_id, embedding in zip(image_ids, embeddings)}

# Save the image database to a file
def save_image_database(database, filename):
    with open(filename, 'wb') as f:
        pickle.dump(database, f)

# Load the image database from a file
def load_image_database(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Custom cosine similarity function
def cosine_similarity_manual(a, b):
    dot_product = np.dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    return dot_product / (norm_a * norm_b)

def cosine_similarity_matrix(matrix, vector):
    dot_products = np.dot(matrix, vector)
    norm_matrix = norm(matrix, axis=1)
    norm_vector = norm(vector)
    return dot_products / (norm_matrix * norm_vector)

# Step 3: Query the Database
def query_database(database, caption_embedding, top_k=5):
    image_ids = list(database.keys())
    embeddings = np.array(list(database.values()))
    
    similarities = cosine_similarity_matrix(embeddings, caption_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [image_ids[i] for i in top_indices]

# Step 4: Display Images
def display_images(image_urls):
    fig, axes = plt.subplots(1, len(image_urls), figsize=(15, 5))
    for ax, url in zip(axes, image_urls):
        img = download_image(url)
        ax.imshow(img)
        ax.axis('off')
    plt.show()