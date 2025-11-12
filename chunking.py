from spacy.lang.en import English
nlp = English() 
nlp.add_pipe("sentencizer")
doc = nlp("This is a sentence. This is another sentence.")
assert len(list(doc.sents)) == 2
from tqdm.auto import tqdm
import pandas as pd

from tqdm.auto import tqdm
import pandas as pd

def split_into_sentences(pages, nlp):
    """Return [(page_num, [sent1, sent2,...])]"""
    results = []
    for p in tqdm(pages, desc="Splitting into sentences"):
        doc = nlp(p["text"])
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        results.append((p["Page_number"], sents))
    return results

def chunk_by_sentences(page_sents, chunk_size=10):
    """Group sentences into fixed-size chunks."""
    chunks = []
    for page_num, sents in page_sents:
        for i in range(0, len(sents), chunk_size):
            group = sents[i:i+chunk_size]
            text = " ".join(group)
            chunks.append({
                "page_number": page_num,
                "chunk_index": (i // chunk_size) + 1,
                "chunk_sentence_count": len(group),
                "chunk_word_count": sum(len(s.split()) for s in group),
                "chunk_char_count": len(text),
                "chunk_text": text
            })
    return chunks

# ---- run it ----
page_sents = split_into_sentences(pages_and_texts, nlp)
chunks_10sent = chunk_by_sentences(page_sents, chunk_size=10)

df_chunks = pd.DataFrame(chunks_10sent)
print(df_chunks.shape)
df_chunks.head()
