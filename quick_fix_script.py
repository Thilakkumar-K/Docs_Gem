#!/usr/bin/env python3
"""
Quick fix script to create the PostgreSQL database
Run this before starting the application
"""

import asyncio
import asyncpg
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Loaded environment variables from .env file")
    else:
        print("ℹ️ No .env file found, using system environment variables")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_document_qa")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Password")

async def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not to our specific database)
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database='postgres'  # Connect to default postgres database
        )
        
        logger.info(f"✅ Connected to PostgreSQL server at {DB_HOST}:{DB_PORT}")
        
        # Check if database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", DB_NAME
        )
        
        if db_exists:
            logger.info(f"ℹ️ Database '{DB_NAME}' already exists")
        else:
            # Create the database
            logger.info(f"🏗️ Creating database '{DB_NAME}'...")
            
            # Note: CREATE DATABASE cannot be run in a transaction
            await conn.execute(f'CREATE DATABASE "{DB_NAME}"')
            logger.info(f"✅ Database '{DB_NAME}' created successfully!")
        
        await conn.close()
        
        # Test connection to the new database
        logger.info(f"🧪 Testing connection to database '{DB_NAME}'...")
        test_conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        
        result = await test_conn.fetchval("SELECT 'Connection successful!' as message")
        logger.info(f"✅ {result}")
        await test_conn.close()
        
        return True
        
    except asyncpg.InvalidAuthorizationSpecificationError:
        logger.error(f"❌ Authentication failed for user '{DB_USER}'")
        logger.error("   Please check your username and password")
        return False
        
    except asyncpg.ConnectionError as e:
        logger.error(f"❌ Cannot connect to PostgreSQL server: {e}")
        logger.error("   Please ensure PostgreSQL is running and accepting connections")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

async def main():
    """Main function"""
    print("=" * 60)
    print("🔧 PostgreSQL Database Setup")
    print("=" * 60)
    print(f"📍 Host: {DB_HOST}:{DB_PORT}")
    print(f"👤 User: {DB_USER}")
    print(f"🗄️ Database: {DB_NAME}")
    print("=" * 60)
    
    success = await create_database()
    
    if success:
        print("\n🎉 Database setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Run: python init_db.py")
        print("   2. Start the application: python main.py")
        print(f"\n🔗 Connection string:")
        print(f"   postgresql+asyncpg://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    else:
        print("\n❌ Database setup failed!")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Make sure PostgreSQL is installed and running:")
        print("      sudo systemctl status postgresql")
        print("   2. Check if the user exists and has correct permissions:")
        print(f"      sudo -u postgres psql -c \"SELECT usename FROM pg_user WHERE usename='{DB_USER}';\"")
        print("   3. Create user if needed:")
        print(f"      sudo -u postgres createuser --interactive --pwprompt {DB_USER}")
        print("   4. Verify connection manually:")
        print(f"      psql -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -d postgres")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)