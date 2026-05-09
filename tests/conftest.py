import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from pvz_ai.config import Settings
from pvz_ai.database import Base


class FakeLLM:
    provider = "test"
    model = "test-model"

    def __init__(self, answer: str = "test answer") -> None:
        self.answer = answer
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return self.answer


@pytest_asyncio.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def test_settings(tmp_path):
    return Settings(
        APP_ENV="test",
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'test.sqlite'}",
        LLM_PROVIDER="echo",
        AUTO_CREATE_TABLES=False,
    )
