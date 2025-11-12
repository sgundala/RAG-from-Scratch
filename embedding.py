from sentence_transformers import util, SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# Load the all-mpnet-base-v2 model
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
