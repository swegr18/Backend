from sqlmodel import create_engine, Session

DATABASE_URL = "postgresql+psycopg://postgres:postgres@db:5432/audiodb"
engine = create_engine(DATABASE_URL, echo=True)


def get_session():
    """Yield a database session for FastAPI dependency injection"""
    with Session(engine) as session:
        yield session
