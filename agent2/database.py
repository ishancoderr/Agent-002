import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_host = os.getenv("DB_HOST", "localhost")
_port = os.getenv("DB_PORT", "5433")
_user = os.getenv("DB_USER", "postgres")
_pass = os.getenv("DB_PASS", "")
_name = os.getenv("DB_NAME", "geostats_agent2")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{_user}:{_pass}@{_host}:{_port}/{_name}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
