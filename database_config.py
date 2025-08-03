#!/usr/bin/env python3
"""
Database configuration and connection management
Handles PostgreSQL connection with async support
"""

import os
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import asyncio
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv

    # Try to load .env file from current directory
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("✅ Loaded environment variables from .env file")
    else:
        logger.info("ℹ️ No .env file found, using system environment variables")

except ImportError:
    logger.warning("⚠️ python-dotenv not installed, using system environment variables only")


def get_database_url() -> str:
    """
    Get database URL from environment variables
    Supports both complete DATABASE_URL and individual components
    """
    # Option 1: Complete DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Option 2: Individual components
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "rag_document_qa")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "password")

    database_url = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return database_url


def get_engine_config() -> Dict[str, Any]:
    """Get engine configuration from environment variables"""
    return {
        "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "300")),
        "pool_pre_ping": True,
        "connect_args": {
            "server_settings": {
                "application_name": "rag_document_qa",
            }
        }
    }


# Database URL and Engine Configuration
DATABASE_URL = get_database_url()
ENGINE_CONFIG = get_engine_config()

# Create async engine
try:
    engine = create_async_engine(DATABASE_URL, **ENGINE_CONFIG)
    logger.info(f"✅ Database engine created successfully")
    logger.info(
        f"🔗 Database URL: {DATABASE_URL.replace(DATABASE_URL.split('@')[0].split(':')[-1], '***')}")  # Hide password
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    engine = None

# Create session factory
if engine:
    SessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False
    )
    logger.info("✅ Session factory created successfully")
else:
    SessionLocal = None
    logger.error("❌ Cannot create session factory without engine")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session
    Use this as a FastAPI dependency
    """
    if not SessionLocal:
        raise RuntimeError("Database not initialized. Check your database configuration.")

    async with SessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health() -> bool:
    """
    Check if database connection is healthy
    Returns True if connection is successful, False otherwise
    """
    if not engine:
        logger.error("❌ Database engine not available")
        return False

    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("✅ Database health check passed")
        return True
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return False


async def get_db_info() -> Dict[str, Any]:
    """
    Get database information and statistics
    """
    if not engine:
        return {"status": "engine_not_available"}

    try:
        async with engine.begin() as conn:
            # Get PostgreSQL version
            version_result = await conn.execute(text("SELECT version()"))
            version = version_result.scalar()

            # Get current database name
            db_result = await conn.execute(text("SELECT current_database()"))
            database_name = db_result.scalar()

            # Get current user
            user_result = await conn.execute(text("SELECT current_user"))
            current_user = user_result.scalar()

            # Get connection count
            try:
                conn_result = await conn.execute(text("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                """))
                connection_count = conn_result.scalar()
            except Exception:
                connection_count = "unavailable"

            # Get database size
            try:
                size_result = await conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                database_size = size_result.scalar()
            except Exception:
                database_size = "unavailable"

            return {
                "status": "connected",
                "database_name": database_name,
                "current_user": current_user,
                "version": version,
                "connection_count": connection_count,
                "database_size": database_size,
                "engine_pool_size": engine.pool.size(),
                "engine_pool_checked_out": engine.pool.checkedout(),
            }

    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def init_database():
    """
    Initialize database tables
    Creates all tables defined in models
    """
    if not engine:
        raise RuntimeError("Database engine not available")

    try:
        # Import models to ensure they're registered
        from models import Base, Document, Question

        logger.info("🏗️ Creating database tables...")

        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ Database tables created successfully")

        # Log table information
        async with engine.begin() as conn:
            # Check if tables exist
            tables_result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('documents', 'questions')
                ORDER BY table_name
            """))
            tables = [row[0] for row in tables_result.fetchall()]

            logger.info(f"📋 Created tables: {', '.join(tables)}")

            # Get table row counts
            for table in tables:
                count_result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = count_result.scalar()
                logger.info(f"📊 Table '{table}': {count} rows")

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


async def close_database():
    """
    Close database connections
    Call this on application shutdown
    """
    if engine:
        try:
            await engine.dispose()
            logger.info("✅ Database connections closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {e}")
    else:
        logger.info("ℹ️ No database engine to close")


async def reset_database():
    """
    Reset database by dropping and recreating all tables
    WARNING: This will delete all data!
    """
    if not engine:
        raise RuntimeError("Database engine not available")

    try:
        from models import Base

        logger.warning("⚠️ RESETTING DATABASE - ALL DATA WILL BE LOST!")

        async with engine.begin() as conn:
            # Drop all tables
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("🗑️ Dropped all tables")

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("🏗️ Recreated all tables")

        logger.info("✅ Database reset completed")

    except Exception as e:
        logger.error(f"❌ Failed to reset database: {e}")
        raise


# Test function for standalone testing
async def test_database_connection():
    """Test database connection and operations"""
    logger.info("🧪 Testing database connection...")

    # Test basic connection
    if not await check_database_health():
        logger.error("❌ Basic connection test failed")
        return False

    # Test session creation
    try:
        async with SessionLocal() as session:
            result = await session.execute(text("SELECT 'Session test successful' as message"))
            message = result.scalar()
            logger.info(f"✅ Session test: {message}")
    except Exception as e:
        logger.error(f"❌ Session test failed: {e}")
        return False

    # Get database info
    db_info = await get_db_info()
    logger.info(f"📊 Database info: {db_info}")

    logger.info("✅ All database tests passed!")
    return True


if __name__ == "__main__":
    """
    Run this file directly to test database configuration
    """


    async def main():
        print("=" * 60)
        print("🔧 Database Configuration Test")
        print("=" * 60)

        print(f"📍 Database URL: {DATABASE_URL.replace(DATABASE_URL.split('@')[0].split(':')[-1], '***')}")
        print(f"⚙️ Engine Config: {ENGINE_CONFIG}")
        print()

        success = await test_database_connection()

        if success:
            print("🎉 Database configuration is working correctly!")
        else:
            print("❌ Database configuration has issues!")

        print("=" * 60)

        return success


    # Run the test
    success = asyncio.run(main())
    exit(0 if success else 1)