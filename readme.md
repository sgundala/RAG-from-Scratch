# Nutrition RAG Pipeline (Gemma 2B + MPNet)

This repo contains a **from-scratch RAG pipeline** built in Google Colab, using:

- A 1200+ page human nutrition PDF as the knowledge source
- Sentence-based chunking (10 sentences per chunk)
- `all-mpnet-base-v2` embeddings from Sentence-Transformers
- A local instruction-tuned LLM (`google/gemma-2b-it`) loaded with 4-bit quantization
- A retrieval → augmentation → generation loop closely following the “Production Level RAG Workshop (Part 2)” by Vizuara

The goal is to show the full RAG workflow **without LangChain / LlamaIndex**.

---

## Pipeline Overview

1. **Ingestion**
   - Load the Human Nutrition PDF with `PyMuPDF`
   - Clean page text (remove newlines, extra spaces)
   - Optionally skip the first 41 pages (front matter)

2. **Chunking**
   - Use `spacy`’s `English()` + `sentencizer` to split each page into sentences
   - Group sentences into chunks of 10 sentences
   - Store for each chunk: page number, chunk index, and sentence text (`sentence_chunk`)

3. **Embedding**
   - Load `all-mpnet-base-v2` from `sentence-transformers`
   - Encode all chunks into 768-dim embeddings
   - Keep embeddings as a torch tensor on GPU for fast retrieval

4. **Retrieval**
   - For a user query:
     - Embed the query with the **same** MPNet model
     - Compute dot-product (cosine) similarity between query and all chunk embeddings
     - Take top-k chunks as context

5. **Augmentation (Prompt Formatting)**
   - Combine the top-k chunks into a bullet list
   - Insert them into a long instruction prompt with a few example Q&A pairs
   - Wrap the final text using `tokenizer.apply_chat_template(...)` for Gemma

6. **Generation (Local LLM)**
   - Load `google/gemma-2b-it` with:
     - 4-bit quantization via `bitsandbytes`
     - `flash_attention_2` when available, otherwise `sdpa`
   - Tokenize the prompt, send to GPU, and call `generate(...)`
   - Decode the output and strip the original prompt to get the final RAG answer

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/rag-nutrition-rag.git
cd rag-nutrition-rag
