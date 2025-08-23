"""
Fixed Supabase Storage utility functions with proper upsert handling
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseStorageManager:
    """Manager class for Supabase storage operations with proper upsert handling"""

    def __init__(self):
        """Initialize Supabase client with environment variables"""
        logger.info("🔧 Initializing SupabaseStorageManager...")

        # Load and validate environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET", "documents")

        # Debug logging (mask sensitive data)
        logger.info(f"🔑 SUPABASE_URL: {self.supabase_url}")
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

            # Verify bucket access
            self._verify_bucket_access()

        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            raise ValueError(f"Failed to initialize Supabase client: {e}")

    def _verify_bucket_access(self):
        """Verify that we can access the bucket with timeout"""
        try:
            logger.info(f"🔍 Verifying access to bucket '{self.bucket_name}'...")

            # Add a simple timeout mechanism
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Bucket verification timeout")

            # Set up timeout (only on non-Windows systems)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)  # 5 second timeout

            try:
                result = self.supabase.storage.from_(self.bucket_name).list()
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout

                if result is not None:
                    logger.info(
                        f"✅ Successfully verified access to bucket '{self.bucket_name}' - Found {len(result) if isinstance(result, list) else 'unknown'} items")
                else:
                    logger.warning(f"⚠️ Bucket access returned None for '{self.bucket_name}'")
            except TimeoutError:
                logger.warning(f"⚠️ Bucket verification timed out for '{self.bucket_name}'")
            except Exception as e:
                logger.warning(f"⚠️ Could not verify bucket access: {e}")
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Ensure timeout is cancelled

        except Exception as e:
            logger.warning(f"⚠️ Bucket verification setup failed: {e}")

    async def upload_file_to_supabase(self, file_name: str, file_data: bytes, overwrite: bool = True) -> str:
        """
        Upload file to Supabase Storage with proper overwrite handling

        Args:
            file_name (str): Name of the file to store
            file_data (bytes): File content as bytes
            overwrite (bool): Whether to overwrite existing files (default: True)

        Returns:
            str: File path in Supabase storage

        Raises:
            HTTPException: If upload fails
        """
        logger.info("=" * 80)
        logger.info(f"📤 STARTING SUPABASE UPLOAD: {file_name}")
        logger.info("=" * 80)

        try:
            logger.info(f"📁 File Name: {file_name}")
            logger.info(f"📊 File Size: {len(file_data)} bytes ({len(file_data) / 1024:.2f} KB)")
            logger.info(f"🪣 Target Bucket: {self.bucket_name}")
            logger.info(f"🔄 Overwrite Mode: {overwrite}")

            # Validate inputs
            if not file_name or not file_name.strip():
                raise ValueError("File name cannot be empty")

            if not file_data:
                raise ValueError("File data cannot be empty")

            # Ensure file_data is bytes
            if not isinstance(file_data, bytes):
                logger.info(f"🔄 Converting file data from {type(file_data)} to bytes")
                if isinstance(file_data, str):
                    file_data = file_data.encode('utf-8')
                else:
                    file_data = bytes(file_data)
                logger.info(f"✅ Converted to {len(file_data)} bytes")

            # Prepare file options (content type)
            content_type = self._get_content_type(file_name)
            file_options = {
                "content-type": content_type
            }
            logger.info(f"⚙️ File Options: {file_options}")

            # Run the synchronous Supabase operation in a thread pool
            loop = asyncio.get_event_loop()

            def do_upload():
                try:
                    logger.info("🔧 Inside thread executor - starting upload...")
                    storage_bucket = self.supabase.storage.from_(self.bucket_name)

                    # Handle overwrite logic
                    if overwrite:
                        # Try to delete existing file first (ignore errors if file doesn't exist)
                        try:
                            logger.info(f"🗑️ Attempting to remove existing file: {file_name}")
                            delete_response = storage_bucket.remove([file_name])
                            logger.info(f"🗑️ Delete response: {delete_response}")
                        except Exception as delete_e:
                            logger.info(f"ℹ️ File might not exist (delete failed): {delete_e}")
                            # This is expected if the file doesn't exist, so we continue

                    # Now upload the file using the correct method signature
                    # Try different method signatures based on storage3 version
                    try:
                        # Method 1: Try with file_options parameter (newer versions)
                        logger.info("🚀 Attempting upload with file_options parameter...")
                        response = storage_bucket.upload(
                            path=file_name,
                            file=file_data,
                            file_options=file_options
                        )
                        logger.info(f"✅ Upload successful with file_options: {response}")

                    except TypeError as te:
                        # Method 2: Try without file_options (older versions)
                        logger.info(f"⚠️ file_options not supported, trying without: {te}")
                        logger.info("🚀 Attempting upload without file_options parameter...")

                        response = storage_bucket.upload(
                            path=file_name,
                            file=file_data
                        )
                        logger.info(f"✅ Upload successful without file_options: {response}")

                    except Exception as upload_e:
                        # Method 3: Try with headers instead of file_options (alternative)
                        logger.info(f"⚠️ Standard upload failed, trying with headers: {upload_e}")

                        # Some versions might use 'headers' instead of 'file_options'
                        try:
                            response = storage_bucket.upload(
                                path=file_name,
                                file=file_data,
                                headers={"Content-Type": content_type}
                            )
                            logger.info(f"✅ Upload successful with headers: {response}")
                        except Exception as headers_e:
                            logger.error(f"❌ All upload methods failed. Last error with headers: {headers_e}")
                            raise upload_e  # Raise the original error

                    return response

                except Exception as thread_e:
                    logger.error(f"❌ Error inside thread executor: {thread_e}")
                    logger.error(f"❌ Thread executor stack trace: {traceback.format_exc()}")
                    raise thread_e

            response = await loop.run_in_executor(None, do_upload)

            # Analyze response
            logger.info(f"📨 Upload Response: {response}")

            # Check if upload was successful
            if response:
                logger.info("✅ Upload appears successful based on response")

                # Verify the file was actually uploaded
                try:
                    logger.info("🔍 Verifying file was uploaded...")
                    verification_result = await loop.run_in_executor(
                        None,
                        lambda: self.supabase.storage.from_(self.bucket_name).list()
                    )

                    if verification_result:
                        uploaded_files = [f.get('name', 'unknown') for f in verification_result]
                        if file_name in uploaded_files:
                            logger.info(f"✅ CONFIRMED: File '{file_name}' is now in the bucket!")
                        else:
                            logger.warning(f"⚠️ File '{file_name}' not found in bucket listing")
                            # Check if the file name is in a nested structure
                            all_paths = []
                            for f in verification_result:
                                if isinstance(f, dict):
                                    all_paths.append(f.get('name', ''))
                            logger.info(f"📋 All files in bucket: {all_paths}")

                except Exception as verify_e:
                    logger.warning(f"⚠️ Could not verify upload: {verify_e}")

                logger.info("=" * 80)
                logger.info(f"✅ UPLOAD COMPLETED SUCCESSFULLY: {file_name}")
                logger.info("=" * 80)
                return file_name

            else:
                raise Exception(f"Upload response was empty or failed: {response}")

        except Exception as e:
            error_msg = f"Failed to upload file {file_name} to Supabase bucket '{self.bucket_name}': {str(e)}"
            logger.error("=" * 80)
            logger.error(f"❌ UPLOAD FAILED: {error_msg}")
            logger.error(f"❌ Exception Type: {type(e)}")
            logger.error(f"❌ Exception Details: {str(e)}")
            logger.error(f"❌ Full Stack Trace:")
            logger.error(traceback.format_exc())
            logger.error("=" * 80)

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
        logger.info(f"📥 DOWNLOADING FILE: {file_path} from bucket '{self.bucket_name}'")

        try:
            # Run the synchronous Supabase operation in a thread pool
            loop = asyncio.get_event_loop()

            def do_download():
                try:
                    logger.info(f"🔧 Inside download thread executor for: {file_path}")
                    storage_bucket = self.supabase.storage.from_(self.bucket_name)
                    response = storage_bucket.download(file_path)
                    logger.info(f"📨 Download response type: {type(response)}")
                    logger.info(f"📨 Download response length: {len(response) if response else 0}")
                    return response
                except Exception as thread_e:
                    logger.error(f"❌ Download thread executor error: {thread_e}")
                    logger.error(f"❌ Download stack trace: {traceback.format_exc()}")
                    raise thread_e

            response = await loop.run_in_executor(None, do_download)

            if response and isinstance(response, bytes):
                logger.info(f"✅ Successfully downloaded file: {file_path} ({len(response)} bytes)")
                return response
            else:
                raise Exception(f"Download response was empty or invalid format: {type(response)}")

        except Exception as e:
            error_msg = f"Failed to download file {file_path} from Supabase bucket '{self.bucket_name}': {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Download stack trace: {traceback.format_exc()}")

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )

    async def get_public_url(self, file_path: str) -> str:
        """
        Get public URL for a file in Supabase Storage

        Args:
            file_path (str): Path of the file in Supabase storage

        Returns:
            str: Public URL of the file
        """
        try:
            logger.info(f"🔗 Getting public URL for: {file_path}")

            # Run the synchronous Supabase operation in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            )

            logger.info(f"✅ Generated public URL for {file_path}: {response}")
            return response if response else ""

        except Exception as e:
            logger.error(f"❌ Failed to get public URL for {file_path}: {e}")
            return ""

    async def delete_file_from_supabase(self, file_path: str) -> bool:
        """
        Delete file from Supabase Storage

        Args:
            file_path (str): Path of the file in Supabase storage

        Returns:
            bool: True if deletion was successful
        """
        try:
            logger.info(f"🗑️ Deleting file from Supabase bucket '{self.bucket_name}': {file_path}")

            # Run the synchronous Supabase operation in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.supabase.storage.from_(self.bucket_name).remove([file_path])
            )

            logger.info(f"📨 Delete response: {response}")

            if response:
                logger.info(f"✅ Successfully deleted file from Supabase: {file_path}")
                return True
            else:
                logger.warning(f"⚠️ Delete response was empty for: {file_path}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to delete file {file_path} from Supabase: {e}")
            logger.error(f"❌ Delete stack trace: {traceback.format_exc()}")
            return False

    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List all files in the Supabase storage bucket

        Args:
            prefix (str): Optional prefix to filter files

        Returns:
            list: List of file objects with metadata
        """
        try:
            logger.info(f"📋 Listing files from Supabase bucket '{self.bucket_name}' with prefix '{prefix}'")

            # Run the synchronous Supabase operation in a thread pool
            loop = asyncio.get_event_loop()

            def do_list():
                storage_bucket = self.supabase.storage.from_(self.bucket_name)
                if prefix:
                    return storage_bucket.list(prefix)
                else:
                    return storage_bucket.list()

            response = await loop.run_in_executor(None, do_list)

            if response:
                logger.info(f"✅ Listed {len(response)} files from Supabase storage")
                return response
            else:
                logger.info("📭 No files found in Supabase storage")
                return []

        except Exception as e:
            logger.error(f"❌ Failed to list files from Supabase: {e}")
            logger.error(f"❌ List files stack trace: {traceback.format_exc()}")
            return []

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in Supabase storage

        Args:
            file_path (str): Path of the file to check

        Returns:
            bool: True if file exists
        """
        try:
            logger.info(f"🔍 Checking if file exists: {file_path}")
            files = await self.list_files()
            exists = any(f.get('name') == file_path for f in files)
            logger.info(f"📍 File exists check result: {exists}")
            return exists
        except Exception as e:
            logger.error(f"❌ Error checking file existence: {e}")
            return False

    def _get_content_type(self, file_name: str) -> str:
        """
        Get content type based on file extension

        Args:
            file_name (str): Name of the file

        Returns:
            str: Content type
        """
        file_name_lower = file_name.lower()

        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.eml': 'message/rfc822',
            '.json': 'application/json',
            '.faiss': 'application/octet-stream',
            '.pkl': 'application/octet-stream',
            '.pickle': 'application/octet-stream'
        }

        for extension, content_type in content_types.items():
            if file_name_lower.endswith(extension):
                logger.info(f"📎 Detected content type '{content_type}' for file '{file_name}'")
                return content_type

        logger.info(f"📎 Using default content type 'application/octet-stream' for file '{file_name}'")
        return 'application/octet-stream'


# Global instance
_supabase_manager = None


def get_supabase_manager() -> SupabaseStorageManager:
    """Get singleton instance of SupabaseStorageManager"""
    global _supabase_manager
    if _supabase_manager is None:
        logger.info("🔧 Creating new SupabaseStorageManager instance")
        _supabase_manager = SupabaseStorageManager()
    else:
        logger.info("♻️ Reusing existing SupabaseStorageManager instance")
    return _supabase_manager


# Convenience functions for backward compatibility
async def upload_file_to_supabase(file_name: str, file_data: bytes, overwrite: bool = True) -> str:
    """
    Upload file to Supabase Storage with overwrite capability

    Args:
        file_name (str): Name of the file to store
        file_data (bytes): File content as bytes
        overwrite (bool): Whether to overwrite existing files

    Returns:
        str: File path in Supabase storage
    """
    manager = get_supabase_manager()
    return await manager.upload_file_to_supabase(file_name, file_data, overwrite)


async def download_file_from_supabase(file_path: str) -> bytes:
    """Download file from Supabase Storage"""
    manager = get_supabase_manager()
    return await manager.download_file_from_supabase(file_path)


async def get_public_url(file_path: str) -> str:
    """Get public URL for a file in Supabase Storage"""
    manager = get_supabase_manager()
    return await manager.get_public_url(file_path)


async def delete_file_from_supabase(file_path: str) -> bool:
    """Delete file from Supabase Storage"""
    manager = get_supabase_manager()
    return await manager.delete_file_from_supabase(file_path)


async def list_supabase_files(prefix: str = "") -> List[Dict[str, Any]]:
    """List files in Supabase storage"""
    manager = get_supabase_manager()
    return await manager.list_files(prefix)

async def download_document_content(source: str) -> Tuple[bytes, str]:
    """
    Download document content from either URL or Supabase storage

    Args:
        source (str): Either a URL or Supabase file path

    Returns:
        Tuple[bytes, str]: (content, source_type) where source_type is 'url' or 'supabase'
    """
    logger.info(f"📥 Downloading document content from: {source}")

    try:
        # Check if it's a URL
        if source.startswith(('http://', 'https://')):
            logger.info(f"🌐 Detected URL source: {source}")

            timeout = httpx.Timeout(30.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(source)
                response.raise_for_status()
                logger.info(f"✅ Downloaded {len(response.content)} bytes from URL")
                return response.content, 'url'
        else:
            # Assume it's a Supabase file path
            logger.info(f"📦 Detected Supabase path: {source}")
            content = await download_file_from_supabase(source)
            logger.info(f"✅ Downloaded {len(content)} bytes from Supabase")
            return content, 'supabase'

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading document from {source}: {e}")
        logger.error(f"❌ Download stack trace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document from {source}: {str(e)}"
        )

async def file_exists_in_supabase(file_path: str) -> bool:
    """Check if file exists in Supabase storage"""
    manager = get_supabase_manager()
    return await manager.file_exists(file_path)


# Test function
async def test_supabase_upload_standalone():
    """Standalone test function to verify Supabase upload works"""
    logger.info("🧪 Running standalone Supabase upload test...")

    try:
        # Create test content
        test_file_name = f"test_upload_{int(datetime.now().timestamp())}.txt"
        test_content = b"Hello from standalone Supabase test!"

        # Upload using our manager
        manager = get_supabase_manager()
        result = await manager.upload_file_to_supabase(test_file_name, test_content, overwrite=True)

        logger.info(f"✅ Standalone test upload successful: {result}")

        # Verify by listing files
        files = await manager.list_files()
        uploaded_files = [f.get('name') for f in files]

        if test_file_name in uploaded_files:
            logger.info(f"✅ Standalone test VERIFIED: File {test_file_name} found in bucket")
        else:
            logger.error(f"❌ Standalone test FAILED: File {test_file_name} not found in bucket")

        # Clean up test file
        await manager.delete_file_from_supabase(test_file_name)
        logger.info(f"🧹 Cleaned up test file: {test_file_name}")

        return True

    except Exception as e:
        logger.error(f"❌ Standalone test failed: {e}")
        logger.error(f"❌ Standalone test stack trace: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Run standalone test if this file is executed directly
    asyncio.run(test_supabase_upload_standalone())