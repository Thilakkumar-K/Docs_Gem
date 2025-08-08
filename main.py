#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with RAG for Document Question Answering
Uses FAISS for vector search, Gemini for generation, and Supabase for storage
Production-ready with intelligent chunking and semantic retrieval - NO LOCAL STORAGE
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
import pickle
from io import BytesIO

# Document processing imports
import PyPDF2
import docx
import email

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

# Supabase storage utilities - FIXED IMPORTS
from supabase_utils import (
    upload_file_to_supabase,
    download_file_from_supabase,
    download_document_content,
    get_public_url,
    delete_file_from_supabase,
    get_supabase_manager,
    list_supabase_files,
    test_supabase_upload_standalone  # Added for testing
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import tempfile
import os


def save_faiss_index_to_bytes(index) -> bytes:
    """
    Save FAISS index to bytes using temporary file

    Args:
        index: FAISS index object

    Returns:
        bytes: Serialized FAISS index data
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        try:
            # Write index to temporary file
            faiss.write_index(index, tmp_file.name)

            # Read the file content as bytes
            with open(tmp_file.name, "rb") as f:
                data = f.read()

            return data
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


async def upload_faiss_index_to_supabase(index, file_key: str):
    """
    Upload FAISS index to Supabase Storage

    Args:
        index: FAISS index object
        file_key (str): Supabase storage path/key
    """
    logger.info(f"🧠 Converting FAISS index to bytes for upload to {file_key}")
    index_bytes = save_faiss_index_to_bytes(index)
    logger.info(f"✅ FAISS index converted to {len(index_bytes)} bytes")

    await upload_file_to_supabase(file_key, index_bytes)
    logger.info(f"✅ FAISS index uploaded to Supabase: {file_key}")


def load_faiss_index_from_bytes(index_bytes: bytes):
    """
    Load FAISS index from bytes using temporary file

    Args:
        index_bytes (bytes): Serialized FAISS index data

    Returns:
        FAISS index object
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        try:
            # Write bytes to temporary file
            tmp_file.write(index_bytes)
            tmp_file.flush()

            # Load index from temporary file
            index = faiss.read_index(tmp_file.name)
            return index
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass

# Configure logging with more detailed format for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables FIRST
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Powered Document QA API with Full Supabase Storage",
    description="Advanced Document Question Answering with Retrieval-Augmented Generation and Complete Cloud Storage",
    version="2.2.0",
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

# DEBUG: Log all critical environment variables on startup
logger.info("🔧 ENVIRONMENT VARIABLES DEBUG:")
logger.info(f"   SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
logger.info(
    f"   SUPABASE_KEY: {'*' * (len(os.getenv('SUPABASE_KEY', '')) - 8) + os.getenv('SUPABASE_KEY', '')[-8:] if os.getenv('SUPABASE_KEY') else 'NOT_SET'}")
logger.info(f"   SUPABASE_BUCKET: {os.getenv('SUPABASE_BUCKET', 'documents')}")
logger.info(f"   GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT_SET'}")
logger.info(f"   VALID_TOKEN: {'SET' if VALID_TOKEN else 'NOT_SET'}")

# Validate they exist
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not VALID_TOKEN:
    raise ValueError("VALID_TOKEN environment variable is required")

# Validate Supabase credentials
if not os.getenv("SUPABASE_URL"):
    raise ValueError("SUPABASE_URL environment variable is required")
if not os.getenv("SUPABASE_KEY"):
    raise ValueError("SUPABASE_KEY environment variable is required")

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini API: {e}")


# Request/Response Models
class DocumentQARequest(BaseModel):
    documents: Optional[str] = None  # URL or Supabase file path
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
    supabase_path: str
    public_url: Optional[str] = None


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
    """Enhanced document processor with intelligent chunking and Supabase integration"""

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
    async def process_document_from_source(cls, source: str) -> tuple[str, str]:
        """Process document from URL or Supabase storage and return text with document ID"""
        # Generate document ID based on source
        doc_id = hashlib.md5(source.encode()).hexdigest()

        # Download document content
        content, source_type = await download_document_content(source)

        logger.info(f"Processing document from {source_type}: {len(content)} bytes")

        # Determine file type and extract text
        source_lower = source.lower()

        if '.pdf' in source_lower or content.startswith(b'%PDF'):
            text = cls.extract_text_from_pdf(content)
        elif '.docx' in source_lower or content.startswith(b'PK'):
            text = cls.extract_text_from_docx(content)
        elif '.eml' in source_lower or b'From:' in content[:1000]:
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

    @classmethod
    async def process_uploaded_file(cls, content: bytes, filename: str) -> str:
        """Process uploaded file content and return extracted text"""
        logger.info(f"Processing uploaded file: {filename} ({len(content)} bytes)")

        # Determine file type and extract text
        filename_lower = filename.lower()

        if filename_lower.endswith('.pdf') or content.startswith(b'%PDF'):
            text = cls.extract_text_from_pdf(content)
        elif filename_lower.endswith('.docx') or content.startswith(b'PK'):
            text = cls.extract_text_from_docx(content)
        elif filename_lower.endswith('.eml'):
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

        return text


class VectorStore:
    """FAISS-based vector store with complete Supabase storage - NO LOCAL STORAGE"""

    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.indexes = {}  # document_id -> faiss index (in-memory cache)
        self.chunks = {}  # document_id -> list of chunks (in-memory cache)

        logger.info(f"✅ Initialized vector store with {EMBEDDING_MODEL_NAME} (dim: {self.dimension})")
        logger.info("📡 Using FULL Supabase storage - NO local file storage")

    async def create_embeddings(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Create embeddings for document chunks and store everything in Supabase"""
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

            # Store in memory cache for immediate use
            self.indexes[document_id] = index
            self.chunks[document_id] = chunks

            # Save to Supabase Storage instead of local disk
            await self._save_to_supabase(document_id, index, chunks)

            logger.info(f"✅ Created FAISS index with {len(chunks)} vectors and saved to Supabase")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create embeddings: {str(e)}"
            )

    async def search_similar_chunks(self, document_id: str, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[
        Dict[str, Any]]:
        """Search for similar chunks using semantic similarity - ENHANCED LOGGING"""
        try:
            logger.info(f"🔍 Searching for chunks in document {document_id} with query: '{query[:50]}...'")

            # Check if document is in memory cache
            if document_id not in self.indexes:
                logger.info(f"📥 Document {document_id} not in memory cache, loading from Supabase...")
                # Load from Supabase instead of local disk
                await self._load_from_supabase(document_id)
            else:
                logger.info(f"🎯 Using cached document {document_id}")

            if document_id not in self.indexes:
                logger.error(f"❌ Document {document_id} not found after attempting to load from Supabase")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {document_id} not found in vector store"
                )

            # Log vector store info
            index = self.indexes[document_id]
            chunks = self.chunks[document_id]
            logger.info(f"📊 Vector store info - Index vectors: {index.ntotal}, Cached chunks: {len(chunks)}")

            # Generate query embedding
            logger.info("🧠 Generating query embedding...")
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search similar vectors
            logger.info(f"🔎 Searching for top-{min(top_k, index.ntotal)} similar vectors...")
            scores, indices = index.search(query_embedding, min(top_k, index.ntotal))

            # Retrieve corresponding chunks
            retrieved_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid index
                    chunk = self.chunks[document_id][idx].copy()
                    chunk["similarity_score"] = float(score)
                    chunk["rank"] = i + 1
                    retrieved_chunks.append(chunk)

            logger.info(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks for query")

            # Log top chunks for debugging
            if retrieved_chunks:
                logger.info("🏆 Top retrieved chunks:")
                for chunk in retrieved_chunks[:3]:
                    logger.info(
                        f"   Rank {chunk['rank']}: Score {chunk['similarity_score']:.3f} - {chunk['text'][:80]}...")
            else:
                logger.warning(f"⚠️ No chunks retrieved for query: '{query}'")

            return retrieved_chunks

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error searching similar chunks: {e}")
            logger.error(f"❌ Search error details: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to search chunks: {str(e)}"
            )

    async def _save_to_supabase(self, document_id: str, index, chunks: List[Dict[str, Any]]):
        """Save FAISS index and chunks to Supabase Storage"""
        try:
            logger.info(f"Saving vector data to Supabase for document {document_id}")

            # Upload FAISS index using the helper function
            index_path = f"vectors/{document_id}/index.faiss"
            await upload_faiss_index_to_supabase(index, index_path)

            # Serialize chunks to JSON bytes
            chunks_json = json.dumps(chunks, indent=2)
            chunks_bytes = chunks_json.encode('utf-8')

            # Upload chunks to Supabase
            chunks_path = f"vectors/{document_id}/chunks.json"
            await upload_file_to_supabase(chunks_path, chunks_bytes)

            # Also save metadata file
            metadata = {
                "document_id": document_id,
                "chunks_count": len(chunks),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "dimension": self.dimension,
                "created_at": time.time(),
                "total_characters": sum(chunk["char_count"] for chunk in chunks)
            }
            metadata_json = json.dumps(metadata, indent=2)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_path = f"vectors/{document_id}/metadata.json"
            await upload_file_to_supabase(metadata_path, metadata_bytes)

            logger.info(f"✅ Saved vector data to Supabase: {index_path}, {chunks_path}, {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving to Supabase: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save vector data to Supabase: {str(e)}"
            )

    async def _load_from_supabase(self, document_id: str):
        """Load FAISS index and chunks from Supabase Storage"""
        try:
            logger.info(f"Loading vector data from Supabase for document {document_id}")

            # Define file paths
            index_path = f"vectors/{document_id}/index.faiss"
            chunks_path = f"vectors/{document_id}/chunks.json"

            # Download index from Supabase
            try:
                index_bytes = await download_file_from_supabase(index_path)
                index = load_faiss_index_from_bytes(index_bytes)
            except Exception as e:
                logger.error(f"Failed to download index from {index_path}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Vector index not found for document {document_id}"
                )

            # Download chunks from Supabase
            try:
                chunks_bytes = await download_file_from_supabase(chunks_path)
                chunks_json = chunks_bytes.decode('utf-8')
                chunks = json.loads(chunks_json)
            except Exception as e:
                logger.error(f"Failed to download chunks from {chunks_path}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chunks data not found for document {document_id}"
                )

            # Store in memory cache
            self.indexes[document_id] = index
            self.chunks[document_id] = chunks

            logger.info(f"✅ Loaded vector data from Supabase for document {document_id}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading from Supabase: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load vector data from Supabase: {str(e)}"
            )


    async def delete_document_vectors(self, document_id: str) -> bool:
        """Delete document vectors from both memory and Supabase"""
        try:
            logger.info(f"Deleting vector data for document {document_id}")

            # Remove from memory cache
            if document_id in self.indexes:
                del self.indexes[document_id]
            if document_id in self.chunks:
                del self.chunks[document_id]

            # Delete from Supabase
            index_path = f"vectors/{document_id}/index.faiss"
            chunks_path = f"vectors/{document_id}/chunks.json"
            metadata_path = f"vectors/{document_id}/metadata.json"

            # Delete all files (don't fail if one doesn't exist)
            results = []
            for path in [index_path, chunks_path, metadata_path]:
                try:
                    result = await delete_file_from_supabase(path)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {e}")
                    results.append(False)

            success = any(results)  # Success if at least one file was deleted
            logger.info(f"✅ Deleted vector data for document {document_id} from Supabase")
            return success

        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            return False

    async def list_stored_documents(self) -> List[Dict[str, Any]]:
        """List all documents with vector data in Supabase"""
        try:
            supabase_manager = get_supabase_manager()
            files = await supabase_manager.list_files()

            # Find all metadata files
            documents = []
            for file_info in files:
                file_path = file_info.get('name', '')
                if file_path.startswith('vectors/') and file_path.endswith('/metadata.json'):
                    try:
                        # Extract document_id from path
                        document_id = file_path.split('/')[1]

                        # Download and parse metadata
                        metadata_bytes = await download_file_from_supabase(file_path)
                        metadata = json.loads(metadata_bytes.decode('utf-8'))

                        # Check if document is currently loaded in memory
                        is_loaded = document_id in self.indexes

                        documents.append({
                            "document_id": document_id,
                            "chunks_count": metadata.get("chunks_count", 0),
                            "total_characters": metadata.get("total_characters", 0),
                            "embedding_model": metadata.get("embedding_model", "unknown"),
                            "created_at": metadata.get("created_at", 0),
                            "status": "loaded" if is_loaded else "stored",
                            "supabase_path": f"vectors/{document_id}/"
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process metadata file {file_path}: {e}")
                        continue

            logger.info(f"Found {len(documents)} documents with vector data in Supabase")
            return documents

        except Exception as e:
            logger.error(f"Error listing stored documents: {e}")
            return []


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
        """Generate answer using RAG approach - FIXED VERSION"""
        if not self.model:
            return {
                "answer": "LLM service not available",
                "confidence": 0.0,
                "sources": [],
                "chunks_retrieved": 0
            }

        try:
            # Log retrieval info
            logger.info(f"🔍 Retrieved {len(relevant_chunks)} chunks for question: {question[:50]}...")

            if not relevant_chunks:
                logger.warning(f"⚠️ No context retrieved for document: {document_id}")
                return {
                    "answer": "No relevant context found in the document to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "chunks_retrieved": 0
                }

            # Log top retrieved chunks for debugging
            for i, chunk in enumerate(relevant_chunks[:3]):
                logger.info(
                    f"📄 Chunk {i + 1} (score: {chunk.get('similarity_score', 0):.3f}): {chunk['text'][:100]}...")

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
                confidence = self._estimate_confidence(relevant_chunks)

                # Extract sources
                sources = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "similarity_score": chunk["similarity_score"],
                        "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    }
                    for chunk in relevant_chunks[:3]  # Top 3 sources
                ]

                logger.info(f"✅ Generated answer with confidence {confidence:.2f}")

                return {
                    "answer": answer_text,
                    "confidence": confidence,
                    "sources": sources,
                    "chunks_retrieved": len(relevant_chunks)
                }
            else:
                logger.error("❌ Gemini returned empty response")
                return {
                    "answer": "Unable to generate response - empty response from LLM",
                    "confidence": 0.0,
                    "sources": [],
                    "chunks_retrieved": len(relevant_chunks)
                }

        except Exception as e:
            logger.error(f"❌ Error generating RAG answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "chunks_retrieved": len(relevant_chunks)
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
        """Create optimized RAG prompt - IMPROVED VERSION"""
        return f"""You are a helpful AI assistant that answers questions based on provided document context.

    DOCUMENT CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    - Answer the question using ONLY the information provided in the context above
    - Be specific and detailed in your response
    - If the context contains the answer, provide a comprehensive response
    - If the context doesn't contain enough information, clearly state: "The provided context doesn't contain sufficient information to answer this question completely."
    - Quote relevant parts from the context when appropriate
    - Be factual and avoid speculation beyond what's stated in the context

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
    """Enhanced health check with Supabase status"""
    supabase_status = "unknown"
    try:
        supabase_manager = get_supabase_manager()
        # Test Supabase connection by attempting to list files
        await supabase_manager.list_files()
        supabase_status = "connected"
    except Exception as e:
        supabase_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "gemini_available": rag_llm_service.model is not None,
            "gemini_model": rag_llm_service.model_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "vector_store_ready": True,
            "supabase_status": supabase_status
        },
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "max_context_length": MAX_CONTEXT_LENGTH,
            "supabase_bucket": os.getenv("SUPABASE_BUCKET", "documents"),
            "storage_mode": "FULL_SUPABASE_ONLY"
        }
    }


@app.post("/api/v1/documents/upload")
async def upload_document(
        file: UploadFile = File(...),
        token: str = Depends(verify_token)
):
    """Upload and process document for RAG using Supabase Storage - FIXED VERSION"""
    logger.info("=" * 100)
    logger.info("📤 DOCUMENT UPLOAD DEBUG SESSION STARTED")
    logger.info("=" * 100)

    try:
        # Log upload details
        logger.info(f"📁 Original filename: {file.filename}")
        logger.info(f"📋 Content type: {file.content_type}")
        logger.info(f"🆔 Generating document ID...")

        # Generate document ID
        document_id = str(uuid.uuid4())
        logger.info(f"✅ Generated document ID: {document_id}")

        # Create Supabase file path with timestamp for uniqueness
        timestamp = int(time.time())
        safe_filename = file.filename.replace(" ", "_").replace("/", "_")
        supabase_file_path = f"documents/{document_id}_{timestamp}_{safe_filename}"
        logger.info(f"📍 Supabase file path: {supabase_file_path}")

        # Read file content
        logger.info("📖 Reading file content...")
        content = await file.read()
        logger.info(f"✅ File content read: {len(content)} bytes")

        # CRITICAL: Upload original file to Supabase FIRST
        logger.info("🚀 UPLOADING FILE TO SUPABASE...")
        logger.info(f"   📁 File path: {supabase_file_path}")
        logger.info(f"   📊 File size: {len(content)} bytes")
        logger.info(f"   🪣 Target bucket: {os.getenv('SUPABASE_BUCKET', 'documents')}")

        try:
            uploaded_path = await upload_file_to_supabase(supabase_file_path, content)
            logger.info(f"✅ SUPABASE UPLOAD SUCCESS: {uploaded_path}")
        except Exception as upload_error:
            logger.error(f"❌ SUPABASE UPLOAD FAILED: {upload_error}")
            logger.error(f"❌ Upload error details: {type(upload_error).__name__}: {str(upload_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to Supabase: {str(upload_error)}"
            )

        # Get public URL (now async)
        logger.info("🔗 Getting public URL...")
        try:
            public_url = await get_public_url(uploaded_path)
            logger.info(f"✅ Public URL generated: {public_url}")
        except Exception as url_error:
            logger.warning(f"⚠️ Could not generate public URL: {url_error}")
            public_url = None

        # Process document content
        logger.info("🔤 Processing document text...")
        processor = DocumentProcessor()
        text = await processor.process_uploaded_file(content, file.filename)
        logger.info(f"✅ Text extracted: {len(text)} characters")

        # Create intelligent chunks
        logger.info("🧩 Creating intelligent chunks...")
        chunks = processor.intelligent_chunking(text)
        logger.info(f"✅ Created {len(chunks)} chunks")

        # Create embeddings and vector index (stored in Supabase)
        logger.info("🧠 Creating embeddings and saving to Supabase...")
        chunks_created = await vector_store.create_embeddings(document_id, chunks)
        logger.info(f"✅ Created and stored {chunks_created} embeddings in Supabase")

        logger.info("=" * 100)
        logger.info("✅ DOCUMENT UPLOAD COMPLETED SUCCESSFULLY!")
        logger.info(f"   📄 Document ID: {document_id}")
        logger.info(f"   📁 Filename: {file.filename}")
        logger.info(f"   📍 Supabase path: {uploaded_path}")
        logger.info(f"   🧩 Chunks created: {chunks_created}")
        logger.info(f"   🔗 Public URL: {public_url or 'Not generated'}")
        logger.info("=" * 100)

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            chunks_created=chunks_created,
            message=f"Document processed successfully with {chunks_created} chunks (stored in Supabase)",
            supabase_path=uploaded_path,
            public_url=public_url if public_url else None
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("=" * 100)
        logger.error("❌ DOCUMENT UPLOAD FAILED!")
        logger.error(f"❌ Error: {type(e).__name__}: {str(e)}")
        logger.error(f"❌ Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 100)

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
    Enhanced RAG-powered document QA endpoint with full Supabase Storage support
    """
    try:
        document_id = None

        # Process document if source provided
        if request.documents and not request.document_id:
            logger.info(f"Processing document from source: {request.documents}")

            # Extract text and get document ID
            text, document_id = await DocumentProcessor.process_document_from_source(request.documents)

            # Check if we already have this document processed (check Supabase)
            if document_id not in vector_store.indexes:
                # Try to load from Supabase first
                try:
                    await vector_store._load_from_supabase(document_id)
                    logger.info(f"Loaded existing vector index from Supabase for document {document_id}")
                except HTTPException:
                    # Document not found in Supabase, create new one
                    chunks = DocumentProcessor.intelligent_chunking(text)
                    await vector_store.create_embeddings(document_id, chunks)
                    logger.info(f"Created new vector index in Supabase for document {document_id}")
            else:
                logger.info(f"Using cached vector index for document {document_id}")

        elif request.document_id:
            document_id = request.document_id
            logger.info(f"Using pre-processed document: {document_id}")

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'documents' URL/Supabase path or 'document_id' must be provided"
            )

        # Process questions using RAG
        # Process questions using RAG - FIXED VERSION
        answers = []
        retrieval_stats = {
            "total_questions": len(request.questions),
            "avg_chunks_retrieved": 0,
            "avg_confidence": 0.0,
            "storage_source": "supabase"
        }

        total_chunks = 0
        total_confidence = 0.0

        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i + 1}/{len(request.questions)}: {question[:50]}...")

            # Retrieve relevant chunks from Supabase-stored vectors
            relevant_chunks = await vector_store.search_similar_chunks(
                document_id, question, TOP_K_RETRIEVAL
            )

            # Log chunk retrieval for debugging
            logger.info(f"📊 Retrieved {len(relevant_chunks)} chunks for question {i + 1}")

            # Generate answer using RAG
            answer_data = await rag_llm_service.generate_rag_answer(
                question, relevant_chunks, document_id
            )

            answer_data["question"] = question
            answers.append(answer_data)

            # Update stats - FIX: Use correct key from answer_data
            chunks_retrieved = answer_data.get("chunks_retrieved", 0)
            confidence = answer_data.get("confidence", 0.0)

            total_chunks += chunks_retrieved
            total_confidence += confidence

            logger.info(f"📈 Question {i + 1} stats: {chunks_retrieved} chunks, {confidence:.2f} confidence")

            # Rate limiting for free tier
            await asyncio.sleep(0.5)

        # Calculate averages
        retrieval_stats["avg_chunks_retrieved"] = total_chunks / len(request.questions) if len(
            request.questions) > 0 else 0
        retrieval_stats["avg_confidence"] = total_confidence / len(request.questions) if len(
            request.questions) > 0 else 0

        logger.info(
            f"🎯 Final stats: avg_chunks={retrieval_stats['avg_chunks_retrieved']:.1f}, avg_confidence={retrieval_stats['avg_confidence']:.2f}")

        logger.info("Successfully processed all questions using RAG with Supabase storage")
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
    """Get chunks for a specific document from Supabase"""
    try:
        # Check memory cache first
        if document_id not in vector_store.chunks:
            # Load from Supabase
            await vector_store._load_from_supabase(document_id)

        if document_id not in vector_store.chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in Supabase storage"
            )

        chunks = vector_store.chunks[document_id]

        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "storage_source": "supabase",
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
    """Search for relevant chunks in a specific document stored in Supabase"""
    try:
        relevant_chunks = await vector_store.search_similar_chunks(
            document_id, query, min(top_k, 20)  # Limit to max 20
        )

        return {
            "document_id": document_id,
            "query": query,
            "results_count": len(relevant_chunks),
            "storage_source": "supabase",
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
    """Delete a document and its associated vector data from Supabase"""
    try:
        # Delete from Supabase (this will also clear memory cache)
        success = await vector_store.delete_document_vectors(document_id)

        return {
            "message": f"Document {document_id} deleted from Supabase storage",
            "document_id": document_id,
            "success": success,
            "note": "Original document file may still exist in Supabase. Use /storage endpoint to delete it."
        }

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.delete("/api/v1/documents/{document_id}/storage")
async def delete_document_from_storage(
        document_id: str,
        supabase_path: str,
        token: str = Depends(verify_token)
):
    """Delete document from both vector store and original file from Supabase storage"""
    try:
        # Remove vector data
        vector_success = await vector_store.delete_document_vectors(document_id)

        # Remove original file from Supabase storage
        file_success = await delete_file_from_supabase(supabase_path)

        return {
            "message": f"Document {document_id} deleted from Supabase storage",
            "document_id": document_id,
            "vector_data_deleted": vector_success,
            "original_file_deleted": file_success,
            "supabase_path": supabase_path
        }

    except Exception as e:
        logger.error(f"Error deleting document from storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document from storage: {str(e)}"
        )


@app.get("/api/v1/documents")
async def list_documents(token: str = Depends(verify_token)):
    """List all processed documents stored in Supabase"""
    try:
        # Get documents from Supabase
        documents = await vector_store.list_stored_documents()

        return {
            "total_documents": len(documents),
            "storage_mode": "supabase_only",
            "documents": documents
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.get("/api/v1/storage/files")
async def list_storage_files(token: str = Depends(verify_token)):
    """List all files in Supabase storage"""
    try:
        # Use the new list function
        files = await list_supabase_files()

        # Separate by file type
        document_files = []
        vector_files = []
        other_files = []

        for file_info in files:
            file_path = file_info.get('name', '')
            if file_path.startswith('documents/'):
                document_files.append(file_info)
            elif file_path.startswith('vectors/'):
                vector_files.append(file_info)
            else:
                other_files.append(file_info)

        return {
            "total_files": len(files),
            "storage_mode": "supabase_only",
            "breakdown": {
                "original_documents": len(document_files),
                "vector_data_files": len(vector_files),
                "other_files": len(other_files)
            },
            "files": {
                "documents": document_files,
                "vectors": vector_files,
                "other": other_files
            }
        }

    except Exception as e:
        logger.error(f"Error listing storage files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list storage files: {str(e)}"
        )


# NEW: Standalone Supabase test endpoint
@app.get("/api/v1/test-supabase-standalone")
async def test_supabase_standalone_upload():
    """Run the standalone Supabase upload test"""
    try:
        logger.info("🧪 Running standalone Supabase upload test via API endpoint...")
        result = await test_supabase_upload_standalone()

        return {
            "status": "success" if result else "failed",
            "message": "Standalone Supabase upload test completed",
            "result": result,
            "note": "Check server logs for detailed information"
        }

    except Exception as e:
        logger.error(f"Standalone test failed: {e}")
        return {
            "status": "failed",
            "message": f"Standalone test failed: {str(e)}",
            "result": False
        }


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
            "storage_mode": "supabase_only",
            "message": "Embedding service is working correctly with Supabase storage"
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
            "context_length": len(test_context),
            "storage_mode": "supabase_only"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini RAG test failed: {str(e)}"
        }


@app.get("/api/v1/test-supabase")
async def test_supabase():
    """Test Supabase Storage connectivity"""
    try:
        supabase_manager = get_supabase_manager()

        # Test upload
        test_content = b"This is a test file for Supabase storage."
        test_filename = f"test_{int(time.time())}.txt"

        # Upload test file
        uploaded_path = await supabase_manager.upload_file_to_supabase(test_filename, test_content)

        # Download test file
        downloaded_content = await supabase_manager.download_file_from_supabase(uploaded_path)

        # Get public URL (now async)
        public_url = await supabase_manager.get_public_url(uploaded_path)

        # Clean up test file
        cleanup_result = await supabase_manager.delete_file_from_supabase(uploaded_path)

        return {
            "status": "success",
            "message": "Supabase Storage is working correctly",
            "test_results": {
                "upload_successful": len(uploaded_path) > 0,
                "download_successful": downloaded_content == test_content,
                "public_url_generated": len(public_url) > 0,
                "cleanup_successful": cleanup_result
            },
            "bucket": supabase_manager.bucket_name,
            "storage_mode": "supabase_only"
        }

    except Exception as e:
        logger.error(f"Supabase Storage test failed: {e}")
        return {
            "status": "error",
            "message": f"Supabase Storage test failed: {str(e)}"
        }


@app.get("/api/v1/test-vector-storage")
async def test_vector_storage():
    """Test vector storage in Supabase"""
    try:
        # Create test chunks
        test_chunks = [
            {
                "text": "This is a test chunk for vector storage testing.",
                "chunk_id": 0,
                "sentence_count": 1,
                "char_count": 49
            },
            {
                "text": "Machine learning is transforming industries rapidly.",
                "chunk_id": 1,
                "sentence_count": 1,
                "char_count": 52
            }
        ]

        # Generate test document ID
        test_doc_id = f"test_{int(time.time())}"

        # Create embeddings and save to Supabase
        chunks_created = await vector_store.create_embeddings(test_doc_id, test_chunks)

        # Test retrieval
        search_results = await vector_store.search_similar_chunks(
            test_doc_id, "machine learning", 2
        )

        # Clean up
        await vector_store.delete_document_vectors(test_doc_id)

        return {
            "status": "success",
            "message": "Vector storage in Supabase is working correctly",
            "test_results": {
                "chunks_created": chunks_created,
                "search_results_count": len(search_results),
                "cleanup_successful": True
            },
            "storage_mode": "supabase_only"
        }

    except Exception as e:
        logger.error(f"Vector storage test failed: {e}")
        return {
            "status": "error",
            "message": f"Vector storage test failed: {str(e)}"
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
            "path": request.url.path,
            "storage_mode": "supabase_only"
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
            "path": request.url.path,
            "storage_mode": "supabase_only"
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with comprehensive debugging"""
    logger.info("🚀 Starting RAG-Powered Document QA API with FULL Supabase Storage")
    logger.info("📡 NO LOCAL STORAGE - All data stored in Supabase")
    logger.info(f"🧠 Embedding model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"⚙️ Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f"🔍 Top-K retrieval: {TOP_K_RETRIEVAL}")
    logger.info(f"☁️ Supabase bucket: {os.getenv('SUPABASE_BUCKET', 'documents')}")

    # Test services
    if rag_llm_service.model:
        logger.info(f"✅ Gemini model ready: {rag_llm_service.model_name}")
    else:
        logger.warning("⚠️ Gemini model not available")

    logger.info(f"✅ Embedding model ready: {EMBEDDING_MODEL_NAME} (dim: {vector_store.dimension})")

    # Test Supabase connection properly with detailed logging
    logger.info("🔧 Testing Supabase Storage connection...")
    try:
        supabase_manager = get_supabase_manager()
        files = await supabase_manager.list_files()
        logger.info(f"✅ Supabase Storage connected successfully - Found {len(files)} files")
        logger.info("📦 Vector data will be stored in 'vectors/' folder")
        logger.info("📄 Original documents will be stored in 'documents/' folder")

        # Log first few files for debugging
        if files:
            logger.info("📋 Sample files in bucket:")
            for i, file_info in enumerate(files[:5]):  # Show first 5 files
                logger.info(f"   {i + 1}. {file_info.get('name', 'unknown')}")
        else:
            logger.info("📭 Bucket is empty (no files found)")

    except Exception as e:
        logger.error(f"❌ Supabase Storage connection failed: {e}")
        logger.error("🔧 Debug Info:")
        logger.error(f"   SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
        logger.error(f"   SUPABASE_KEY: {'SET' if os.getenv('SUPABASE_KEY') else 'NOT_SET'}")
        logger.error(f"   SUPABASE_BUCKET: {os.getenv('SUPABASE_BUCKET', 'documents')}")
        logger.warning("Make sure SUPABASE_URL, SUPABASE_KEY, and SUPABASE_BUCKET are set correctly")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down RAG-Powered Document QA API")
    logger.info("📡 All data remains safely stored in Supabase")


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
    print("📡 Using FULL Supabase storage - NO local files")

    try:
        uvicorn.run(**config)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import sys
        sys.exit(1)
        _create_rag_prompt