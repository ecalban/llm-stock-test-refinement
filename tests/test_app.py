from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app import models
from app.db import Base, get_session
from app.main import app


def make_client() -> TestClient:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override
    return TestClient(app)


def test_health():
    assert make_client().get("/health").json() == {"status": "ok"}


def test_health_db():
    response = make_client().get("/health/db")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_schema_creates_all_tables():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    assert set(inspect(engine).get_table_names()) == {
        "stocks", "prices", "signals", "test_runs", "mutants",
    }


def test_mutation_score_excludes_equivalent_mutants():
    run = models.TestRun(mutants_total=10, mutants_killed=6, mutants_equivalent=2)
    assert run.mutation_score == 0.75


def test_mutation_score_is_none_without_mutants():
    run = models.TestRun(mutants_total=0, mutants_killed=0, mutants_equivalent=0)
    assert run.mutation_score is None
