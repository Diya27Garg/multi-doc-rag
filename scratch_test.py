from app.ingestion import load_pdf
from app.vectorstore import VectorStore

fname = "NovaTech_Annual_Report_FY2024.pdf"
with open(f"data/sample_docs/{fname}", "rb") as f:
    chunks = load_pdf(f.read(), fname)
print("chunks:", len(chunks))
print(chunks[0])

store = VectorStore()
store.add(chunks)
for r in store.search("What was the total revenue?", k=3):
    print(round(r["score"], 2), r["source"], "p.", r["page"], "|", r["text"][:120])