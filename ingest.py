import os
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Konfigurasi
PDF_FILE: str = "BUKU PEDOMAN TUGAS AKHIR DAN MAGANG INTERNAL FAKULTAS INFORMATIKA.pdf"
CHROMA_DIR: str = "./chroma_db"
COLLECTION_NAME: str = "pedoman_kampus"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


def load_pdf(file_path: str) -> list:
    """Membaca file PDF dan mengembalikan daftar dokumen per halaman."""
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' tidak ditemukan!")
        print("   Pastikan file PDF berada di direktori yang sama dengan script ini.")
        sys.exit(1)

    print(f"    Membaca file: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"    Berhasil membaca {len(documents)} halaman")
    return documents


def split_documents(documents: list) -> list:
    """Memecah dokumen menjadi chunk-chunk kecil untuk embedding."""
    print(f"    Memecah dokumen (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   Diperoleh {len(chunks)} chunk dari {len(documents)} halaman")
    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """Menginisialisasi model embedding HuggingFace (lokal, gratis)."""
    print(f"    Memuat model embedding: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("    Model embedding siap")
    return embeddings


def store_to_chroma(chunks: list, embeddings: HuggingFaceEmbeddings) -> None:
    """Menyimpan chunk dokumen ke ChromaDB sebagai vector store."""

    # Hapus database lama jika ada agar data selalu fresh
    if Path(CHROMA_DIR).exists():
        import shutil
        print(f"  Menghapus database lama di {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print(f" Menyimpan {len(chunks)} chunk ke ChromaDB ({CHROMA_DIR})")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"    Berhasil menyimpan ke {CHROMA_DIR}")
    print(f"    Total dokumen di database: {vectorstore._collection.count()}")


def main() -> None:
    """Pipeline utama: Load PDF → Split → Embed → Store."""
    print(" RAG Ingestion Pipeline — Pedoman Kampus")
    start_time = time.time()

    # Step 1: Muat PDF
    documents = load_pdf(PDF_FILE)

    # Step 2: Pecah menjadi chunk
    chunks = split_documents(documents)

    # Step 3: Inisialisasi model embedding
    embeddings = create_embeddings()

    # Step 4: Simpan ke ChromaDB
    store_to_chroma(chunks, embeddings)

    elapsed = time.time() - start_time
    print(f" Ingestion selesai dalam {elapsed:.1f} detik")
    print(f" Database tersimpan di: {os.path.abspath(CHROMA_DIR)}")



if __name__ == "__main__":
    main()
