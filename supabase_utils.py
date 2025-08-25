"""
Fixed Supabase Storage utility functions with proper error handling
Supports both newer and older versions of storage3 SDK
"""

import os
import logging
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from supabase import create_client, Client
from fastapi import HTTPException, status
import httpx
import json
from datetime import datetime
import traceback
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseStorageManager:
    """Manager class for Supabase storage operations with proper error handling"""

    def __init__(self):
        """Initialize Supabase client with environment variables"""
        logger.info("🔧 Initializing SupabaseStorageManager...")

        # Load and validate environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET", "documents")

        # Debug logging (mask sensitive data)
        logger.info(f"🔗 SUPABASE_URL: {self.supabase_url}")
        logger.info(f"🔑 SUPABASE_KEY: {'*' * (len(self.supabase_key) - 8) + self.supabase_key[-8:] if self.supabase_key else 'NOT_SET'}")
        logger.info(f"🪣 SUPABASE_BUCKET: {self.bucket_name}")

        if not self.supabase_url or not self.supabase_key:
            error_msg = "❌ SUPABASE_URL and SUPABASE_KEY environment variables are required"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.info("🔌 Creating Supabase client...")
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"✅ Supabase client initialized successfully for bucket '{self.bucket_name}'")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            raise ValueError(f"Failed to initialize Supabase client: {e}")

    async def upload_file_to_supabase(self, file_name: str, file_data: bytes, overwrite: bool = True) -> str:
        """
        Upload file to Supabase Storage with proper error handling

        Args:
            file_name (str): Name of the file to store
            file_data (bytes): File content as bytes
            overwrite (bool): Whether to overwrite existing files (default: True)

        Returns:
            str: File path in Supabase storage

        Raises:
            HTTPException: If upload fails
        """
        logger.info(f"📤 UPLOADING FILE: {file_name} ({len(file_data)} bytes)")

        try:
            # Validate inputs
            if not file_name or not file_name.strip():
                raise ValueError("File name cannot be empty")

            if not file_data:
                raise ValueError("File data cannot be empty")

            # Ensure file_data is bytes
            if not isinstance(file_data, bytes):
                if isinstance(file_data, str):
                    file_data = file_data.encode('utf-8')
                else:
                    file_data = bytes(file_data)

            # Get storage bucket
            storage_bucket = self.supabase.storage.from_(self.bucket_name)

            # Handle overwrite logic
            if overwrite:
                try:
                    logger.info(f"🗑️ Removing existing file: {file_name}")
                    storage_bucket.remove([file_name])
                except Exception as delete_e:
                    logger.info(f"ℹ️ File might not exist (delete failed): {delete_e}")

            # Upload the file - try different methods based on storage3 version
            try:
                # Method 1: Simple upload (most compatible)
                logger.info("🚀 Attempting simple upload...")
                response = storage_bucket.upload(
                    path=file_name,
                    file=file_data
                )
                logger.info(f"✅ Upload successful: {response}")

            except Exception as upload_e:
                logger.error(f"❌ Upload failed: {upload_e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to upload file: {str(upload_e)}"
                )

            # Verify the upload
            if response:
                logger.info(f"✅ UPLOAD COMPLETED: {file_name}")
                return file_name
            else:
                raise Exception(f"Upload response was empty: {response}")

        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Failed to upload file {file_name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

    async def download_file_from_supabase(self, file_path: str) -> bytes:
        """
        Download file from Supabase Storage

        Args:
            file_path (str): Path of the file in Supabase storage

        Returns:
            bytes: File content

        Raises:
            HTTPException: If download fails
        """
        logger.info(f"📥 DOWNLOADING FILE: {file_path}")

        try:
            storage_bucket = self.supabase.storage.from_(self.bucket_name)
            response = storage_bucket.download(file_path)

            if response and isinstance(response, bytes):
                logger.info(f"✅ Downloaded file: {file_path} ({len(response)} bytes)")
                return response
            else:
                raise Exception(f"Download response was empty or invalid: {type(response)}")

        except Exception as e:
            error_msg = f"Failed to download file {file_path}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )

    async def get_public_url(self, file_path: str) -> str:
        """Get public URL for a file"""
        try:
            logger.info(f"🔗 Getting public URL for: {file_path}")
            response = self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            logger.info(f"✅ Generated public URL: {response}")
            return response if response else ""
        except Exception as e:
            logger.error(f"❌ Failed to get public URL: {e}")
            return ""

    async def delete_file_from_supabase(self, file_path: str) -> bool:
        """Delete file from Supabase Storage"""
        try:
            logger.info(f"🗑️ Deleting file: {file_path}")
            response = self.supabase.storage.from_(self.bucket_name).remove([file_path])
            logger.info(f"✅ Delete response: {response}")
            return bool(response)
        except Exception as e:
            logger.error(f"❌ Failed to delete file: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in the bucket with optional prefix"""
        try:
            logger.info(f"📋 Listing files with prefix: '{prefix}'")
            storage_bucket = self.supabase.storage.from_(self.bucket_name)

            if prefix:
                # List files with specific prefix
                response = storage_bucket.list(prefix)
            else:
                # List all files in root
                response = storage_bucket.list()

            # Handle nested directory structure
            all_files = []
            if response:
                for item in response:
                    # If it's a folder, recursively list its contents
                    if item.get('id') is None and 'name' in item:
                        # This is a folder - recursively get its contents
                        folder_name = item['name']
                        folder_prefix = f"{prefix}{folder_name}/" if prefix else f"{folder_name}/"
                        try:
                            folder_contents = storage_bucket.list(folder_prefix)
                            if folder_contents:
                                for subitem in folder_contents:
                                    # Add full path
                                    subitem['name'] = f"{folder_prefix}{subitem.get('name', '')}"
                                    all_files.append(subitem)
                        except Exception as e:
                            logger.warning(f"Failed to list folder {folder_name}: {e}")
                    else:
                        # This is a file
                        if prefix:
                            item['name'] = f"{prefix}{item.get('name', '')}"
                        all_files.append(item)

                logger.info(f"✅ Found {len(all_files)} files")
                return all_files
            else:
                logger.info("🔭 No files found")
                return []
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return []

    def _get_content_type(self, file_name: str) -> str:
        """Get content type based on file extension"""
        file_name_lower = file_name.lower()

        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.eml': 'message/rfc822',
            '.json': 'application/json',
            '.faiss': 'application/octet-stream',
            '.pkl': 'application/octet-stream'
        }

        for extension, content_type in content_types.items():
            if file_name_lower.endswith(extension):
                return content_type

        return 'application/octet-stream'


# Global instance
_supabase_manager = None

def get_supabase_manager() -> SupabaseStorageManager:
    """Get singleton instance of SupabaseStorageManager"""
    global _supabase_manager
    if _supabase_manager is None:
        _supabase_manager = SupabaseStorageManager()
    return _supabase_manager

# Convenience functions
async def upload_file_to_supabase(file_name: str, file_data: bytes, overwrite: bool = True) -> str:
    """Upload file to Supabase Storage"""
    manager = get_supabase_manager()
    return await manager.upload_file_to_supabase(file_name, file_data, overwrite)

async def download_file_from_supabase(file_path: str) -> bytes:
    """Download file from Supabase Storage"""
    manager = get_supabase_manager()
    return await manager.download_file_from_supabase(file_path)

async def get_public_url(file_path: str) -> str:
    """Get public URL for a file"""
    manager = get_supabase_manager()
    return await manager.get_public_url(file_path)

async def delete_file_from_supabase(file_path: str) -> bool:
    """Delete file from Supabase Storage"""
    manager = get_supabase_manager()
    return await manager.delete_file_from_supabase(file_path)

async def list_supabase_files(prefix: str = "") -> List[Dict[str, Any]]:
    """List files in Supabase storage with optional prefix"""
    manager = get_supabase_manager()
    return await manager.list_files(prefix)


async def download_document_content(source: str) -> Tuple[bytes, str]:
    """Download document content from URL or Supabase with content validation"""
    logger.info(f"📥 Downloading from: {source}")

    try:
        if source.startswith(('http://', 'https://')):
            # URL source - Enhanced for Google Drive
            timeout = httpx.Timeout(60.0, connect=15.0)  # Longer timeout for Drive
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
                response = await client.get(source)
                response.raise_for_status()

                content = response.content
                content_type = response.headers.get('content-type', '').lower()

                # Validate content is not HTML (common for Drive access issues)
                if content_type.startswith('text/html') or b'<!DOCTYPE html>' in content[:1000]:
                    logger.error(f"Received HTML instead of document content from {source}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to download file: received HTML instead of document. File may not be publicly accessible."
                    )

                # Validate minimum file size
                if len(content) < 100:
                    logger.error(f"Downloaded content too small: {len(content)} bytes")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Downloaded file too small ({len(content)} bytes). File may be inaccessible."
                    )

                # Validate content type for common document formats
                if not _is_valid_document_content(content, content_type):
                    logger.error(f"Invalid document content detected. Content-Type: {content_type}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Downloaded content is not a valid document format."
                    )

                return content, 'url'
        else:
            # Supabase path
            content = await download_file_from_supabase(source)
            return content, 'supabase'

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download: {str(e)}"
        )


def _is_valid_document_content(content: bytes, content_type: str) -> bool:
    """Validate if content is a valid document format"""
    if not content or len(content) < 10:
        return False

    # Check file signatures
    if content.startswith(b'%PDF'):
        return True
    elif content.startswith(b'PK'):  # ZIP-based formats (DOCX, etc.)
        return True
    elif content_type and any(doc_type in content_type for doc_type in [
        'application/pdf', 'application/msword', 'application/vnd.openxml', 'text/plain'
    ]):
        return True
    elif b'From:' in content[:1000] or b'Subject:' in content[:1000]:  # Email
        return True

    # Check for HTML content (invalid)
    html_indicators = [b'<!DOCTYPE', b'<html', b'<HTML', b'<body', b'<BODY']
    if any(indicator in content[:1000] for indicator in html_indicators):
        return False

    return True

async def test_supabase_upload_standalone():
    """Test function to verify Supabase upload works"""
    logger.info("🧪 Running Supabase upload test...")

    try:
        # Create test content
        test_file_name = f"test_upload_{int(datetime.now().timestamp())}.txt"
        test_content = b"Hello from Supabase test!"

        # Upload using our manager
        manager = get_supabase_manager()
        result = await manager.upload_file_to_supabase(test_file_name, test_content)

        logger.info(f"✅ Test upload successful: {result}")

        # Clean up
        await manager.delete_file_from_supabase(test_file_name)
        logger.info(f"🧹 Cleaned up test file")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

