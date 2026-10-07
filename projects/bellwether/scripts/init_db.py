"""Create the Bellwether SQLite database from the schema. Idempotent."""

from bellwether import db
from bellwether.config import DB_PATH
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.init_db")


def main() -> None:
    conn = db.init_db()
    tables = db.table_names(conn)
    conn.close()
    log.info("initialized %s with tables: %s", DB_PATH.name, ", ".join(tables))


if __name__ == "__main__":
    main()
