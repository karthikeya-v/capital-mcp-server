"""
Database configuration and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from .models import Base

# Database URL from environment variable or default to SQLite for development
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'sqlite:///./ml_trading_bot.db'  # Default SQLite for development
)

# For production, use PostgreSQL:
# postgresql://username:password@localhost:5432/trading_bot

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Thread-safe session
Session = scoped_session(SessionLocal)


def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


def drop_database():
    """Drop all database tables (USE WITH CAUTION!)"""
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped!")


@contextmanager
def get_db():
    """
    Context manager for database sessions.

    Usage:
        with get_db() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_session():
    """Get a database session"""
    return SessionLocal()
