#!/usr/bin/env python3
"""
Database initialization script
Run this script to create database tables and perform initial setup
"""

import asyncio
import logging
from database_config import init_database, check_database_health, get_db_info
from models import Document, Question

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main initialization function"""
    logger.info("🚀 Starting database initialization...")

    try:
        # Check database connectivity
        logger.info("📡 Checking database connectivity...")
        if not await check_database_health():
            logger.error("❌ Cannot connect to database. Please check your configuration.")
            return False

        logger.info("✅ Database connection successful!")

        # Get database info
        db_info = await get_db_info()
        logger.info(f"📊 Database info: {db_info}")

        # Initialize tables
        logger.info("🏗️ Creating database tables...")
        await init_database()

        logger.info("✅ Database initialization completed successfully!")
        logger.info("📝 Tables created:")
        logger.info("   - documents: Store document metadata and content")
        logger.info("   - questions: Store questions and answers")

        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        exit(1)

    print("\n" + "=" * 50)
    print("🎉 Database setup complete!")
    print("=" * 50)
    print("You can now start the FastAPI application with:")
    print("  python main.py")
    print("or")
    print("  uvicorn main:app --reload")
    print("=" * 50)