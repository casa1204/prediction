"""SQLAlchemy 엔진, 세션 팩토리, 테이블 초기화 설정."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """모든 ORM 모델의 테이블을 생성한다."""
    from db.models import Base

    Base.metadata.create_all(bind=engine)


def get_session():
    """새 DB 세션을 반환한다. 사용 후 반드시 close() 호출."""
    return SessionLocal()
