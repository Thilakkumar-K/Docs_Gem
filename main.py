#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with RAG for Document Question Answering
Uses FAISS for vector search and Gemini for generation
Production-ready with intelligent chunking and semantic retrieval
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import logging
import os
from dotenv import load_dotenv
import time
import hashlib
import json
import uuid
from pathlib import Path

# Document processing imports
import PyPDF2
import docx
import email
from io import BytesIO

# RAG and embedding imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import re

# LLM integration (Google Generative AI)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Powered Document QA API",
    description="Advanced Document Question Answering with Retrieval-Augmented Generation",
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
# Security
security = HTTPBearer()
VALID_TOKEN = os.getenv("VALID_TOKEN")

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight but effective
CHUNK_SIZE = 2000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
TOP_K_RETRIEVAL = 10  # Number of chunks to retrieve
MAX_CONTEXT_LENGTH = 10000  # Max context for Gemini

# Validate they exist
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not VALID_TOKEN:
    raise ValueError("VALID_TOKEN environment variable is required")

# Initialize directories
UPLOAD_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vector_db")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini API: {e}")


# Request/Response Models
class DocumentQARequest(BaseModel):
    documents: Optional[str] = None  # URL to document
    questions: List[str]
    document_id: Optional[str] = None  # For pre-processed documents

    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one question is required")
        if len(v) > 10:
            raise ValueError("Maximum 10 questions allowed per request")
        return v


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int
    message: str


class DocumentQAResponse(BaseModel):
    answers: List[Dict[str, Any]]
    document_id: str
    retrieval_info: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


class DocumentProcessor:
    """Enhanced document processor with intelligent chunking"""

    @staticmethod
    async def download_document(url: str) -> bytes:
        """Download document from URL with enhanced error handling"""
        try:
            timeout = httpx.Timeout(30.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                logger.info(f"Downloading document from: {url}")
                response = await client.get(url)
                response.raise_for_status()
                logger.info(f"Downloaded {len(response.content)} bytes")
                return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download document: {str(e)}"
            )

    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF with better error handling"""
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process PDF: {str(e)}"
            )

    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX with enhanced processing"""
        try:
            docx_file = BytesIO(content)
            doc = docx.Document(docx_file)
            text = ""

            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    text += paragraph.text + "\n"

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text += " | ".join(row_text) + "\n"

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process DOCX: {str(e)}"
            )

    @staticmethod
    def extract_text_from_email(content: bytes) -> str:
        """Extract text from email content"""
        try:
            email_str = content.decode('utf-8', errors='ignore')
            msg = email.message_from_string(email_str)

            text = ""

            # Extract headers
            for header in ['From', 'To', 'Subject', 'Date']:
                if msg.get(header):
                    text += f"{header}: {msg.get(header)}\n"
            text += "\n"

            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text += payload.decode('utf-8', errors='ignore') + "\n"
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text += payload.decode('utf-8', errors='ignore')

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting email text: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process email: {str(e)}"
            )

    @classmethod
    def intelligent_chunking(cls, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[
        Dict[str, Any]]:
        """
        Intelligent text chunking that preserves semantic meaning
        """
        # Clean the text
        text = cls._clean_text(text)

        # Split into sentences
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""
        current_chunk_sentences = []

        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
                current_chunk_sentences.append(sentence)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_id": len(chunks),
                        "sentence_count": len(current_chunk_sentences),
                        "char_count": len(current_chunk)
                    })

                # Start new chunk with overlap
                if overlap > 0 and current_chunk_sentences:
                    # Calculate how many sentences to include for overlap
                    overlap_sentences = []
                    overlap_chars = 0

                    for sent in reversed(current_chunk_sentences):
                        if overlap_chars + len(sent) <= overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_chars += len(sent)
                        else:
                            break

                    current_chunk = " ".join(overlap_sentences + [sentence])
                    current_chunk_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_chunk_sentences = [sentence]

        # Add the last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": len(chunks),
                "sentence_count": len(current_chunk_sentences),
                "char_count": len(current_chunk)
            })

        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Handle None or empty text
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page breaks and form feeds
        text = re.sub(r'[\f\r]+', '\n', text)

        # Normalize quotes - using Unicode escapes to avoid syntax issues
        text = re.sub(r'["""]', '"', text)  # Double quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Single quotes (left and right)

        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    @classmethod
    async def process_document(cls, url: str) -> tuple[str, str]:
        """Process document and return text with document ID"""
        # Generate document ID
        doc_id = hashlib.md5(url.encode()).hexdigest()

        # Download document
        content = await cls.download_document(url)

        # Determine file type and extract text
        url_lower = url.lower()

        if '.pdf' in url_lower or content.startswith(b'%PDF'):
            text = cls.extract_text_from_pdf(content)
        elif '.docx' in url_lower or content.startswith(b'PK'):
            text = cls.extract_text_from_docx(content)
        elif '.eml' in url_lower or b'From:' in content[:1000]:
            text = cls.extract_text_from_email(content)
        else:
            try:
                text = content.decode('utf-8', errors='ignore')
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Unsupported document format. Supported: PDF, DOCX, EML, TXT"
                )

        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document appears to be empty or contains no extractable text"
            )

        return text, doc_id


class VectorStore:
    """FAISS-based vector store for document embeddings"""

    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.indexes = {}  # document_id -> faiss index
        self.chunks = {}  # document_id -> list of chunks

        logger.info(f"✅ Initialized vector store with {EMBEDDING_MODEL_NAME} (dim: {self.dimension})")

    async def create_embeddings(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Create embeddings for document chunks and build FAISS index"""
        try:
            logger.info(f"Creating embeddings for {len(chunks)} chunks")

            # Extract text from chunks
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings = embeddings.astype('float32')

            # Create FAISS index
            index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add embeddings to index
            index.add(embeddings)

            # Store index and chunks
            self.indexes[document_id] = index
            self.chunks[document_id] = chunks

            # Save to disk
            await self._save_index(document_id, index, chunks)

            logger.info(f"✅ Created FAISS index with {len(chunks)} vectors")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create embeddings: {str(e)}"
            )

    async def search_similar_chunks(self, document_id: str, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[
        Dict[str, Any]]:
        """Search for similar chunks using semantic similarity"""
        try:
            if document_id not in self.indexes:
                # Try to load from disk
                await self._load_index(document_id)

            if document_id not in self.indexes:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {document_id} not found in vector store"
                )

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search similar vectors
            index = self.indexes[document_id]
            scores, indices = index.search(query_embedding, min(top_k, index.ntotal))

            # Retrieve corresponding chunks
            retrieved_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid index
                    chunk = self.chunks[document_id][idx].copy()
                    chunk["similarity_score"] = float(score)
                    chunk["rank"] = i + 1
                    retrieved_chunks.append(chunk)

            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks for query")
            return retrieved_chunks

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to search chunks: {str(e)}"
            )

    async def _save_index(self, document_id: str, index, chunks: List[Dict[str, Any]]):
        """Save FAISS index and chunks to disk"""
        try:
            doc_dir = VECTOR_DB_DIR / document_id
            doc_dir.mkdir(exist_ok=True)

            # Save FAISS index
            faiss.write_index(index, str(doc_dir / "index.faiss"))

            # Save chunks metadata
            with open(doc_dir / "chunks.json", 'w') as f:
                json.dump(chunks, f, indent=2)

            logger.info(f"Saved index and chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    async def _load_index(self, document_id: str):
        """Load FAISS index and chunks from disk"""
        try:
            doc_dir = VECTOR_DB_DIR / document_id
            index_path = doc_dir / "index.faiss"
            chunks_path = doc_dir / "chunks.json"

            if index_path.exists() and chunks_path.exists():
                # Load FAISS index
                index = faiss.read_index(str(index_path))

                # Load chunks
                with open(chunks_path, 'r') as f:
                    chunks = json.load(f)

                self.indexes[document_id] = index
                self.chunks[document_id] = chunks

                logger.info(f"Loaded index and chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Error loading index: {e}")


class RAGLLMService:
    """Enhanced LLM service with RAG capabilities"""

    def __init__(self):
        self.model = None
        self.model_name = None
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Gemini model with optimized settings"""
        try:
            model_variants = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-1.0-pro-latest',
                'gemini-1.0-pro'
            ]

            for model_name in model_variants:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    test_response = test_model.generate_content(
                        "Test message. Respond with 'OK'.",
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=10,
                        ),
                        safety_settings=self.safety_settings
                    )

                    if test_response and test_response.text:
                        self.model = test_model
                        self.model_name = model_name
                        logger.info(f"✅ Successfully initialized {model_name}")
                        return

                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {e}")
                    continue

            raise Exception("No working Gemini model found")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model: {e}")
            self.model = None
            self.model_name = None

    async def generate_rag_answer(self, question: str, relevant_chunks: List[Dict[str, Any]], document_id: str) -> Dict[
        str, Any]:
        """Generate answer using RAG approach"""
        if not self.model:
            return {
                "answer": "LLM service not available",
                "confidence": 0.0,
                "sources": []
            }

        try:
            # Construct context from relevant chunks
            context = self._build_context(relevant_chunks)

            # Create RAG prompt
            prompt = self._create_rag_prompt(question, context)

            # Generate response
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    top_p=0.8,
                    top_k=40
                )
            )

            if response and response.text:
                answer_text = response.text.strip()

                # Extract confidence and create sources
                sources = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "similarity_score": chunk["similarity_score"],
                        "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    }
                    for chunk in relevant_chunks[:3]  # Top 3 sources
                ]

                return {
                    "answer": answer_text,
                    "confidence": self._estimate_confidence(relevant_chunks),
                    "sources": sources,
                    "context_used": len(context),
                    "chunks_retrieved": len(relevant_chunks)
                }
            else:
                return {
                    "answer": "Unable to generate response",
                    "confidence": 0.0,
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        total_length = 0

        for i, chunk in enumerate(chunks):
            chunk_text = f"[Context {i + 1}]\n{chunk['text']}\n"

            if total_length + len(chunk_text) > MAX_CONTEXT_LENGTH:
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return "\n".join(context_parts)

    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create optimized RAG prompt"""
        return f"""You are an expert document analyst. Answer the question based on the provided context from the document.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say "The provided context doesn't contain sufficient information to answer this question"
3. Be specific and cite relevant details from the context
4. Keep your answer concise but comprehensive
5. If you find conflicting information, mention it

ANSWER:"""

    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Estimate confidence based on similarity scores"""
        if not chunks:
            return 0.0

        # Use weighted average of top chunks
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        total_score = 0.0
        total_weight = 0.0

        for i, chunk in enumerate(chunks[:5]):
            weight = weights[i] if i < len(weights) else 0.1
            total_score += chunk["similarity_score"] * weight
            total_weight += weight

        confidence = total_score / total_weight if total_weight > 0 else 0.0
        return min(confidence, 1.0)


# Initialize services
vector_store = VectorStore()
rag_llm_service = RAGLLMService()


# API Routes
@app.get("/api/v1/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "gemini_available": rag_llm_service.model is not None,
            "gemini_model": rag_llm_service.model_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "vector_store_ready": True
        },
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "max_context_length": MAX_CONTEXT_LENGTH
        }
    }


@app.post("/api/v1/documents/upload")
async def upload_document(
        file: UploadFile = File(...),
        token: str = Depends(verify_token)
):
    """Upload and process document for RAG"""
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())

        # Save uploaded file
        file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document
        processor = DocumentProcessor()

        # Extract text based on file type
        filename_lower = file.filename.lower()
        if filename_lower.endswith('.pdf') or content.startswith(b'%PDF'):
            text = processor.extract_text_from_pdf(content)
        elif filename_lower.endswith('.docx') or content.startswith(b'PK'):
            text = processor.extract_text_from_docx(content)
        elif filename_lower.endswith('.eml'):
            text = processor.extract_text_from_email(content)
        else:
            text = content.decode('utf-8', errors='ignore')

        # Create intelligent chunks
        chunks = processor.intelligent_chunking(text)

        # Create embeddings and vector index
        chunks_created = await vector_store.create_embeddings(document_id, chunks)

        # Clean up uploaded file
        file_path.unlink()

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            chunks_created=chunks_created,
            message=f"Document processed successfully with {chunks_created} chunks"
        )

    except Exception as e:
        logger.error(f"Error processing uploaded document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@app.post(
    "/api/v1/hackrx/run",
    response_model=DocumentQAResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def process_document_qa_rag(
        request: DocumentQARequest,
        token: str = Depends(verify_token)
):
    """
    Enhanced RAG-powered document QA endpoint
    """
    try:
        document_id = None

        # Process document if URL provided
        if request.documents and not request.document_id:
            logger.info(f"Processing document from URL: {request.documents}")

            # Extract text and get document ID
            text, document_id = await DocumentProcessor.process_document(request.documents)

            # Check if we already have this document processed
            if document_id not in vector_store.indexes:
                # Create intelligent chunks
                chunks = DocumentProcessor.intelligent_chunking(text)

                # Create embeddings and vector index
                await vector_store.create_embeddings(document_id, chunks)
                logger.info(f"Created new vector index for document {document_id}")
            else:
                logger.info(f"Using existing vector index for document {document_id}")

        elif request.document_id:
            document_id = request.document_id
            logger.info(f"Using pre-processed document: {document_id}")

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'documents' URL or 'document_id' must be provided"
            )

        # Process questions using RAG
        answers = []
        retrieval_stats = {
            "total_questions": len(request.questions),
            "avg_chunks_retrieved": 0,
            "avg_confidence": 0.0
        }

        total_chunks = 0
        total_confidence = 0.0

        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i + 1}/{len(request.questions)}: {question[:50]}...")

            # Retrieve relevant chunks
            relevant_chunks = await vector_store.search_similar_chunks(
                document_id, question, TOP_K_RETRIEVAL
            )

            # Generate answer using RAG
            answer_data = await rag_llm_service.generate_rag_answer(
                question, relevant_chunks, document_id
            )

            answer_data["question"] = question
            answers.append(answer_data)

            # Update stats
            total_chunks += answer_data.get("chunks_retrieved", 0)
            total_confidence += answer_data.get("confidence", 0.0)

            # Rate limiting for free tier
            await asyncio.sleep(0.5)

        # Calculate averages
        retrieval_stats["avg_chunks_retrieved"] = total_chunks / len(request.questions)
        retrieval_stats["avg_confidence"] = total_confidence / len(request.questions)

        logger.info("Successfully processed all questions using RAG")

        return DocumentQAResponse(
            answers=answers,
            document_id=document_id,
            retrieval_info=retrieval_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG document QA: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/v1/documents/{document_id}/chunks")
async def get_document_chunks(
        document_id: str,
        token: str = Depends(verify_token)
):
    """Get chunks for a specific document"""
    try:
        if document_id not in vector_store.chunks:
            # Try to load from disk
            await vector_store._load_index(document_id)

        if document_id not in vector_store.chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        chunks = vector_store.chunks[document_id]

        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "char_count": chunk["char_count"],
                    "sentence_count": chunk["sentence_count"]
                }
                for chunk in chunks
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


@app.post("/api/v1/documents/{document_id}/search")
async def search_document_chunks(
        document_id: str,
        query: str,
        top_k: int = 5,
        token: str = Depends(verify_token)
):
    """Search for relevant chunks in a specific document"""
    try:
        relevant_chunks = await vector_store.search_similar_chunks(
            document_id, query, min(top_k, 20)  # Limit to max 20
        )

        return {
            "document_id": document_id,
            "query": query,
            "results_count": len(relevant_chunks),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "similarity_score": chunk["similarity_score"],
                    "rank": chunk["rank"],
                    "text": chunk["text"],
                    "char_count": chunk["char_count"]
                }
                for chunk in relevant_chunks
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search chunks: {str(e)}"
        )


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(
        document_id: str,
        token: str = Depends(verify_token)
):
    """Delete a document and its associated data"""
    try:
        # Remove from memory
        if document_id in vector_store.indexes:
            del vector_store.indexes[document_id]
        if document_id in vector_store.chunks:
            del vector_store.chunks[document_id]

        # Remove from disk
        doc_dir = VECTOR_DB_DIR / document_id
        if doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)

        return {
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.get("/api/v1/documents")
async def list_documents(token: str = Depends(verify_token)):
    """List all processed documents"""
    try:
        documents = []

        # Check memory
        for doc_id in vector_store.chunks.keys():
            chunk_count = len(vector_store.chunks[doc_id])
            documents.append({
                "document_id": doc_id,
                "chunks_count": chunk_count,
                "status": "loaded"
            })

        # Check disk for additional documents
        if VECTOR_DB_DIR.exists():
            for doc_dir in VECTOR_DB_DIR.iterdir():
                if doc_dir.is_dir() and doc_dir.name not in [d["document_id"] for d in documents]:
                    chunks_file = doc_dir / "chunks.json"
                    if chunks_file.exists():
                        try:
                            with open(chunks_file, 'r') as f:
                                chunks_data = json.load(f)
                            documents.append({
                                "document_id": doc_dir.name,
                                "chunks_count": len(chunks_data),
                                "status": "stored"
                            })
                        except Exception:
                            pass

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.get("/api/v1/test-embedding")
async def test_embedding():
    """Test endpoint for embedding functionality"""
    try:
        test_texts = [
            "This is a test sentence for embedding.",
            "Machine learning is transforming many industries.",
            "The weather is nice today."
        ]

        # Generate embeddings
        embeddings = vector_store.embedding_model.encode(test_texts)

        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)

        return {
            "status": "success",
            "model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": vector_store.dimension,
            "test_texts": test_texts,
            "similarity_matrix": similarities.tolist(),
            "message": "Embedding service is working correctly"
        }

    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        return {
            "status": "error",
            "message": f"Embedding test failed: {str(e)}"
        }


@app.get("/api/v1/test-gemini")
async def test_gemini():
    """Test Gemini API connectivity with RAG context"""
    try:
        if not rag_llm_service.model:
            return {"status": "error", "message": "Gemini model not initialized"}

        # Test with a simple RAG-style prompt
        test_context = """
        Context: Artificial Intelligence (AI) is a broad field of computer science 
        that aims to create machines capable of performing tasks that typically 
        require human intelligence.
        """

        test_question = "What is AI?"

        prompt = f"""Based on the following context, answer the question:

{test_context}

Question: {test_question}

Answer:"""

        response = rag_llm_service.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100,
            )
        )

        return {
            "status": "success",
            "message": "Gemini API is working with RAG",
            "model_name": rag_llm_service.model_name,
            "test_response": response.text if response and response.text else "No response text",
            "context_length": len(test_context)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini RAG test failed: {str(e)}"
        }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Enhanced request logging middleware"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response with timing
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")

    return response


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    logger.error(f"HTTP Exception [{request.url.path}]: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler"""
    logger.error(f"Unhandled exception [{request.url.path}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc) if os.getenv("DEBUG") else None,
            "path": request.url.path
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting RAG-Powered Document QA API")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"🗃️ Vector DB directory: {VECTOR_DB_DIR}")
    logger.info(f"🧠 Embedding model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"⚙️ Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f"🔍 Top-K retrieval: {TOP_K_RETRIEVAL}")

    # Test services
    if rag_llm_service.model:
        logger.info(f"✅ Gemini model ready: {rag_llm_service.model_name}")
    else:
        logger.warning("⚠️ Gemini model not available")

    logger.info(f"✅ Embedding model ready: {EMBEDDING_MODEL_NAME} (dim: {vector_store.dimension})")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down RAG-Powered Document QA API")


# Replace the bottom section of your main.py with this:

if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable (important for Docker/cloud deployment)
    port = int(os.getenv("PORT", 8000))

    # Enhanced configuration for Docker
    config = {
        "app": "main:app",  # Use string format for better compatibility
        "host": "0.0.0.0",  # Must be 0.0.0.0 for Docker
        "port": port,
        "log_level": "info",
        "reload": False,  # Always False in production/Docker
        "workers": 1,
        "access_log": True,
        "use_colors": False,  # Better for Docker logs
        "loop": "asyncio"  # Specify event loop
    }

    print(f"🚀 Starting server on http://0.0.0.0:{port}")

    try:
        uvicorn.run(**config)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import sys

        sys.exit(1)

