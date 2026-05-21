from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    if database_url.startswith("postgresql+asyncpg://"):
        database_url = normalize_asyncpg_query(database_url)

    return database_url


def normalize_asyncpg_query(database_url: str) -> str:
    parsed = urlsplit(database_url)
    query_items = []
    ssl_value: str | None = None

    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key == "sslmode":
            if value in {"require", "verify-ca", "verify-full"}:
                ssl_value = value
            continue
        if key == "channel_binding":
            continue
        if key == "ssl":
            ssl_value = value
            continue
        query_items.append((key, value))

    if ssl_value:
        query_items.append(("ssl", ssl_value))

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )


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
