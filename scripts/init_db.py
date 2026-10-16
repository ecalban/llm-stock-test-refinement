"""Create all tables. Temporary until migrations are added (week 2)."""

from app import models  # noqa: F401  (registers tables on Base.metadata)
from app.db import Base, engine


def main() -> None:
    Base.metadata.create_all(engine)
    print("Tables created:", ", ".join(sorted(Base.metadata.tables)))


if __name__ == "__main__":
    main()
