"""Database schema (draft).

Prediction layer: stocks, prices, signals.
Test layer: test_runs (one row per generation round) and mutants.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


# --- Prediction layer -------------------------------------------------------


class Stock(Base):
    __tablename__ = "stocks"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. THYAO.IS
    name: Mapped[str | None] = mapped_column(String(128))

    prices: Mapped[list["Price"]] = relationship(back_populates="stock")


class Price(Base):
    """Daily OHLCV bar."""

    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("symbol", "date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    high: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    low: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    close: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    adj_close: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    volume: Mapped[int] = mapped_column(BigInteger)

    stock: Mapped[Stock] = relationship(back_populates="prices")


class Signal(Base):
    """Model output for one stock on one day. Trades are executed on the next day."""

    __tablename__ = "signals"
    __table_args__ = (UniqueConstraint("symbol", "date", "model_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(8))  # buy | sell | hold
    probability: Mapped[float | None] = mapped_column(Float)


# --- Test layer -------------------------------------------------------------


class TestRun(Base):
    """One round of LLM test generation + mutation testing for one module."""

    __tablename__ = "test_runs"
    __test__ = False  # keep pytest from collecting this class

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment: Mapped[str] = mapped_column(String(64), index=True)
    target_module: Mapped[str] = mapped_column(String(128))  # e.g. prediction.indicators
    llm_model: Mapped[str] = mapped_column(String(64))
    feedback_format: Mapped[str] = mapped_column(String(16))  # none | score | line | full
    round: Mapped[int] = mapped_column(Integer)

    tests_generated: Mapped[int] = mapped_column(Integer, default=0)
    tests_valid: Mapped[int] = mapped_column(Integer, default=0)
    mutants_total: Mapped[int] = mapped_column(Integer, default=0)
    mutants_killed: Mapped[int] = mapped_column(Integer, default=0)
    mutants_equivalent: Mapped[int] = mapped_column(Integer, default=0)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    mutants: Mapped[list["Mutant"]] = relationship(back_populates="run", cascade="all, delete-orphan")

    @property
    def mutation_score(self) -> float | None:
        """Killed / (total - equivalent). None until mutants have been counted."""
        effective = self.mutants_total - self.mutants_equivalent
        return self.mutants_killed / effective if effective > 0 else None


class Mutant(Base):
    __tablename__ = "mutants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("test_runs.id"), index=True)
    mutant_key: Mapped[str] = mapped_column(String(128))  # mutmut's mutant name
    file_path: Mapped[str] = mapped_column(String(256))
    line: Mapped[int | None] = mapped_column(Integer)
    operator: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # killed | survived | timeout | equivalent
    diff: Mapped[str | None] = mapped_column(Text)

    run: Mapped[TestRun] = relationship(back_populates="mutants")
