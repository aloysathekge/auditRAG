from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from auditrag.core.config import get_settings
from auditrag.db.models import Base

settings = get_settings()
engine = create_engine(settings.postgres_url, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False, autoflush=False)


def init_db() -> None:
    """Create query_log table if it does not exist."""
    Base.metadata.create_all(bind=engine)


def check_postgres() -> tuple[bool, str | None]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except SQLAlchemyError as e:
        return False, str(e)


def get_db_session() -> Session:
    return SessionLocal()
