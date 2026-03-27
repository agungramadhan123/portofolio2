import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Muat environment variables dari file .env
load_dotenv()

# Konfigurasi
CHROMA_DIR: str = "./chroma_db"
COLLECTION_NAME: str = "pedoman_kampus"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
GEMINI_MODEL: str = "gemini-2.5-flash"
TOP_K_RESULTS: int = 5

# Validasi environment
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY tidak ditemukan!\n"
        "Buat file .env dengan isi: GOOGLE_API_KEY=your_key_here\n"
        "Dapatkan API key di: https://aistudio.google.com/app/apikey"
    )

if not Path(CHROMA_DIR).exists():
    raise RuntimeError(
        "Database ChromaDB tidak ditemukan!\n"
        "Jalankan 'python ingest.py' terlebih dahulu untuk memproses dokumen."
    )

# Inisialisasi komponen RAG
# 1. Model Embedding (lokal, gratis — sama dengan ingest.py)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 2. Koneksi ke ChromaDB yang sudah ada
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

# 3. Retriever: mencari dokumen relevan dari vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_RESULTS},
)

# 4. LLM: Google Gemini (gemini-1.5-flash — stabil, hemat kuota free tier)
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

# 5. Prompt template untuk RAG
SYSTEM_PROMPT: str = (
    "Kamu adalah asisten AI kampus yang membantu menjawab pertanyaan "
    "berdasarkan dokumen pedoman kampus. Jawab pertanyaan HANYA berdasarkan "
    "konteks yang diberikan. Jika jawabannya tidak ada di konteks, katakan "
    "'Maaf, informasi tersebut tidak ditemukan dalam dokumen pedoman kampus.'\n\n"
    "Berikan jawaban yang jelas, terstruktur, dan dalam Bahasa Indonesia.\n\n"
    "Konteks:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

# 6. Bangun RAG chain menggunakan LangChain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Inisialisasi FastAPI
app = FastAPI(
    title="RAG API — Pedoman Kampus",
    description=(
        "API untuk menjawab pertanyaan seputar pedoman kampus "
        "menggunakan Retrieval-Augmented Generation (RAG). "
        "Didukung oleh LangChain, ChromaDB, HuggingFace Embeddings, "
        "dan Google Gemini (gemini-1.5-flash)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware — agar bisa diakses dari Front-End
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Izinkan semua origin (sesuaikan di production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models — Request & Response
class ChatRequest(BaseModel):
    """Schema request body untuk endpoint /chat."""
    pertanyaan: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Pertanyaan yang ingin dijawab berdasarkan dokumen kampus",
        json_schema_extra={"examples": ["Apa syarat pendaftaran TA?"]},
    )


class ChatResponse(BaseModel):
    """Schema response body untuk endpoint /chat."""
    jawaban: str = Field(
        ...,
        description="Jawaban yang dihasilkan oleh AI berdasarkan dokumen",
    )
    sumber_dokumen: list[str] = Field(
        ...,
        description="Daftar sumber halaman dokumen yang relevan",
    )

# Endpoints
@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    """Health check — verifikasi bahwa server berjalan."""
    return {
        "status": "online",
        "message": "RAG API — Pedoman Kampus | Kunjungi /docs untuk dokumentasi",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["RAG"],
    summary="Tanya jawab berbasis dokumen kampus",
    description=(
        "Kirim pertanyaan dan dapatkan jawaban yang diambil dari "
        "dokumen pedoman kampus menggunakan teknik RAG."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Endpoint utama RAG:
    1. Menerima pertanyaan dari user
    2. Mencari dokumen relevan di ChromaDB
    3. Mengirim konteks + pertanyaan ke Google Gemini (gemini-1.5-flash)
    4. Mengembalikan jawaban beserta sumber dokumen
    """
    try:
        # Jalankan RAG chain
        result: dict = rag_chain.invoke({"input": request.pertanyaan})

        # Ekstrak jawaban
        jawaban: str = result.get("answer", "Maaf, tidak dapat menghasilkan jawaban.")

        # Ekstrak sumber dokumen (halaman PDF)
        source_documents = result.get("context", [])
        sumber_dokumen: list[str] = []

        for doc in source_documents:
            source = doc.metadata.get("source", "Tidak diketahui")
            page = doc.metadata.get("page", "?")
            sumber_info = f"{source} — Halaman {int(page) + 1}" if page != "?" else source

            if sumber_info not in sumber_dokumen:
                sumber_dokumen.append(sumber_info)

        return ChatResponse(
            jawaban=jawaban,
            sumber_dokumen=sumber_dokumen,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}",
        )


@app.get("/stats", tags=["General"])
async def stats() -> dict:
    """Menampilkan statistik database vektor."""
    try:
        collection_count: int = vectorstore._collection.count()
        return {
            "total_chunks": collection_count,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": GEMINI_MODEL,
            "chroma_directory": CHROMA_DIR,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil statistik: {str(e)}",
        )
