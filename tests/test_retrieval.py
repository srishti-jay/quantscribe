import sys
sys.path.insert(0, ".")
from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever

print("Loading embedding model...")
embedder = EmbeddingPipeline()

print("Loading FAISS indices...")
hdfc_index = BankIndex("HDFC_BANK_annual_report_FY25")
hdfc_index.load("indices/active")
sbi_index = BankIndex("SBI_annual_report_FY25")
sbi_index.load("indices/active")
print(f"HDFC vectors: {hdfc_index.size}, SBI vectors: {sbi_index.size}")

retriever = PeerGroupRetriever({"HDFC": hdfc_index, "SBI": sbi_index})
query_vec = embedder.embed_query("credit risk gross NPA net NPA provision coverage asset quality")
results = retriever.retrieve(query_vec, ["HDFC_BANK", "SBI"], top_k_per_bank=3)

print("\n=== Credit Risk Query Results ===")
for bank, hits in results.items():
    print(f"\n{bank}:")
    for h in hits:
        print(f"  Score: {h['score']:.3f}, Page: {h['metadata']['page_number']}, Section: {h['metadata'].get('section_header', 'N/A')}")