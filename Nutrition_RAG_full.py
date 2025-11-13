"""
Nutrition RAG Pipeline
----------------------

End-to-end RAG over a Human Nutrition PDF:

1. PDF ingestion with PyMuPDF
2. Sentence-based chunking with spaCy (10 sentences per chunk)
3. Embeddings with all-mpnet-base-v2
4. Retrieval with dot-product similarity
5. Prompt formatting with examples
6. Local LLM (google/gemma-2b-it) for generation

To use Gemma, set your Hugging Face token in an env var:

    export HUGGINGFACE_HUB_TOKEN=hf_xxx...

Then run this script or copy cells into Colab.
"""

import os
import random

import fitz  # PyMuPDF
import numpy as np
import torch
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import is_flash_attn_2_available


# -------------------------------------------------------------------
# 1. Device setup
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")


# -------------------------------------------------------------------
# 2. PDF ingestion
# -------------------------------------------------------------------

def text_formatter(text: str) -> str:
    """Basic cleanup: remove newlines, tabs, and extra spaces."""
    if not text:
        return ""
    cleaned = text.replace("\n", " ").replace("\t", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def read_pdf_as_dicts(pdf_path: str, skip_first_pages: int = 0):
    """
    Read a PDF into a list of page dicts.

    Each dict:
        {
            "Page_number": int,
            "text": str
        }
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        for i in tqdm(range(skip_first_pages, total), desc="Reading pages"):
            page = doc[i]
            text = page.get_text("text")
            cleaned = text_formatter(text)
            pages.append(
                {
                    "Page_number": i + 1,
                    "text": cleaned,
                }
            )
    return pages


# Change this path to where your PDF lives in the repo / Colab
PDF_PATH = "Human-Nutrition.pdf"  # e.g. put the file in the repo root
pages_and_texts = read_pdf_as_dicts(PDF_PATH, skip_first_pages=41)
print(f"[INFO] Loaded {len(pages_and_texts)} pages (after skipping front matter).")


# -------------------------------------------------------------------
# 3. Sentence-based chunking (10 sentences per chunk)
# -------------------------------------------------------------------

nlp = English()
nlp.add_pipe("sentencizer")


def split_into_sentences(pages, nlp):
    """Return list of (page_number, [sent1, sent2, ...])."""
    results = []
    for p in tqdm(pages, desc="Splitting into sentences"):
        doc = nlp(p["text"])
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        results.append((p["Page_number"], sents))
    return results


def chunk_by_sentences(page_sents, chunk_size: int = 10):
    """
    Group sentences into fixed-size chunks.

    Returns list of dicts:
        {
            "page_number": int,
            "chunk_index": int,
            "sentence_chunk": str
        }
    """
    chunks = []
    for page_num, sents in page_sents:
        for i in range(0, len(sents), chunk_size):
            group = sents[i : i + chunk_size]
            text = " ".join(group)
            chunks.append(
                {
                    "page_number": page_num,
                    "chunk_index": (i // chunk_size) + 1,
                    "sentence_chunk": text,
                }
            )
    return chunks


page_sents = split_into_sentences(pages_and_texts, nlp)
pages_and_chunks = chunk_by_sentences(page_sents, chunk_size=10)
print(f"[INFO] Created {len(pages_and_chunks)} chunks.")


# -------------------------------------------------------------------
# 4. Embeddings with all-mpnet-base-v2
# -------------------------------------------------------------------

print("[INFO] Loading embedding model: all-mpnet-base-v2")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

chunk_texts = [item["sentence_chunk"] for item in pages_and_chunks]
print("[INFO] Encoding chunks...")
embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_tensor=True,
    normalize_embeddings=True,
)
print("[INFO] Embeddings shape:", embeddings.shape)


def retrieve_relevant_resources(query: str, embeddings, n_resources_to_return: int = 5):
    """
    Embed query and return top-k scores + indices from the embeddings tensor.
    """
    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
    return scores, indices


# -------------------------------------------------------------------
# 5. Load local LLM (Gemma 2B-It) with quantization
# -------------------------------------------------------------------

print("[INFO] Setting up Gemma 2B-It...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

if device == "cuda" and is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

print(f"[INFO] Using attention implementation: {attn_implementation}")

model_id = "google/gemma-2b-it"
print(f"[INFO] Using model_id: {model_id}")

# If you set HUGGINGFACE_HUB_TOKEN in your environment, this will use it.
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=False,
    attn_implementation=attn_implementation,
)


# -------------------------------------------------------------------
# 6. Prompt formatter (with examples, Gemma chat template)
# -------------------------------------------------------------------

def prompt_formatter(query: str, context_items):
    """
    Build a prompt that includes:
    - instructions
    - 3 example Q&A pairs
    - retrieved context
    - the user query
    Then wrap it with tokenizer.apply_chat_template for Gemma.
    """

    context = "- " + "\n- ".join(
        [item["sentence_chunk"] for item in context_items]
    )

    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and stored in the body's fatty tissues and liver.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with insulin resistance, obesity, physical inactivity, and a diet high in calories and refined carbohydrates. Genetics and age can also contribute to the risk.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water helps maintain blood volume, regulate body temperature, and lubricate joints. Dehydration can lead to fatigue, reduced endurance, and impaired coordination.

Now use the following context items to answer the user query:
{context}
Relevant passages: <extract relevant passages from the context here>

User query: {query}
Answer:"""

    base_prompt = base_prompt.format(context=context, query=query)

    dialogue_template = [
        {"role": "user", "content": base_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# -------------------------------------------------------------------
# 7. ask() function: Retrieval + Augmentation + Generation
# -------------------------------------------------------------------

def ask(
    query: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    return_answer_only: bool = True,
):
    """
    Takes a query, finds relevant resources/context, and generates
    an answer using the local LLM.
    """

    # 1. Retrieve top-k relevant chunks
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)

    # 2. Build context list
    context_items = [pages_and_chunks[i] for i in indices]

    # 3. Build model-ready prompt
    prompt = prompt_formatter(query=query, context_items=context_items)

    # 4. Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # 5. Generate tokens
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
    )

    # 6. Decode tokens
    output_text = tokenizer.decode(outputs[0])

    # 7. Clean up helper text and special tokens
    output_text = (
        output_text.replace(prompt, "")
        .replace("<bos>", "")
        .replace("<eos>", "")
        .replace("Sure, here is the answer to the user query:", "")
    ).strip()

    if return_answer_only:
        return output_text

    return output_text, context_items


# -------------------------------------------------------------------
# 8. Example query list
# -------------------------------------------------------------------

gpt4_questions = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "Describe the process of digestion and absorption of nutrients in the human body.",
    "What role does fibre play in digestion? Name five fibre containing foods.",
    "Explain the concept of energy balance and its importance in weight management.",
]

manual_questions = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
]

query_list = gpt4_questions + manual_questions


# -------------------------------------------------------------------
# 9. Simple CLI / demo entrypoint
# -------------------------------------------------------------------

def main():
    # Pick a random query from the list
    query = random.choice(query_list)
    print(f"\nQuery: {query}\n")

    # Get answer and context items
    answer, context_items = ask(
        query=query,
        temperature=0.7,
        max_new_tokens=512,
        return_answer_only=False,
    )

    print("Answer:\n")
    print(answer)
    print("\nContext item page numbers:", [c["page_number"] for c in context_items])


if __name__ == "__main__":
    main()
