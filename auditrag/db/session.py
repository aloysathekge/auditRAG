from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from auditrag.core.config import get_settings

settings = get_settings()
engine = create_engine(settings.postgres_url, future=True, pool_pre_ping=True)


def check_postgres() -> tuple[bool, str | None]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except SQLAlchemyError as e:
        return False, str(e)
