import os
import sqlite3
from datetime import datetime
from pathlib import Path


def init_db(db_path: Path) -> None:
    """
    Creates DB file and table if not exists.
    Enables WAL mode.
    SRS ref: REL-4
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                file_name        TEXT NOT NULL,
                sha256           TEXT NOT NULL,
                file_format      TEXT NOT NULL,
                file_size        INTEGER NOT NULL,
                predicted_family TEXT NOT NULL,
                confidence       REAL NOT NULL,
                device_used      TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    # Enforce restrictive permissions
    try:
        os.chmod(db_path, 0o600)
    except OSError:
        pass


def log_detection_event(
    db_path: Path,
    file_name: str,
    sha256: str,
    file_format: str,
    file_size: int,
    predicted_family: str,
    confidence: float,
    device: str,
) -> None:
    """
    Inserts one detection event row.
    On IntegrityError: retry once. On second failure: log to stderr, do NOT raise.
    SRS ref: FR-B3, FR5
    """
    db_path = Path(db_path)
    timestamp = datetime.utcnow().isoformat()

    def _insert(conn):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            INSERT INTO detection_events
                (timestamp, file_name, sha256, file_format, file_size,
                 predicted_family, confidence, device_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, file_name, sha256, file_format, file_size,
             predicted_family, confidence, device),
        )
        conn.commit()

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            _insert(conn)
        except sqlite3.IntegrityError:
            # Retry once
            try:
                _insert(conn)
            except sqlite3.IntegrityError as e:
                import sys
                print(f"[maltwin db] Failed to log detection event: {e}", file=sys.stderr)
        finally:
            conn.close()
    except Exception as e:
        import sys
        print(f"[maltwin db] Database error: {e}", file=sys.stderr)


def get_recent_events(db_path: Path, limit: int = 5) -> list:
    """
    Returns last `limit` rows ordered by id DESC.
    SRS ref: FR1.4
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute(
            "SELECT * FROM detection_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return rows
