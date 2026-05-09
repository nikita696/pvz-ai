from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from pvz_ai.config import get_settings


class Base(DeclarativeBase):
    pass


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


def ensure_sqlite_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite+aiosqlite:///"):
        return

    raw_path = database_url.removeprefix("sqlite+aiosqlite:///")
    if raw_path in {":memory:", ""}:
        return

    Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def create_engine(database_url: str) -> AsyncEngine:
    normalized_url = normalize_database_url(database_url)
    ensure_sqlite_parent(normalized_url)
    return create_async_engine(normalized_url, pool_pre_ping=True)


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        create_engine(database_url),
        expire_on_commit=False,
        autoflush=False,
    )


settings = get_settings()
SessionLocal = create_session_factory(settings.database_url)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def create_tables(engine: AsyncEngine) -> None:
    from pvz_ai import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
