"""Neo4j driver factory.

Reads credentials from `.env` / environment, fails with a clear, actionable
message when the database is unreachable or auth is wrong, and retries with
exponential backoff so callers can ride out container start-up.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_URI = "bolt://localhost:7687"


def get_driver(max_retries: int = 6, base_delay: float = 1.0) -> Driver:
    """Return a connectivity-verified Neo4j driver.

    Raises:
        RuntimeError  — missing password or authentication failure.
        ConnectionError — database unreachable after `max_retries` attempts.
    """
    uri = os.environ.get("NEO4J_URI", DEFAULT_URI)
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Copy .env.example to .env and set it "
            "(matching docker-compose's NEO4J_AUTH)."
        )

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            # Silence "label/relationship does not exist" notifications: the
            # ontology defines labels (Claim, Discrepancy, ...) that are
            # populated in later phases, so querying them before then is expected.
            driver = GraphDatabase.driver(
                uri, auth=(user, password), notifications_min_severity="OFF")
            driver.verify_connectivity()
            return driver
        except AuthError as e:
            raise RuntimeError(
                f"Neo4j authentication failed for user '{user}' at {uri}. "
                "Check NEO4J_USER/NEO4J_PASSWORD in .env match the container's "
                "NEO4J_AUTH (if you changed the password, remove ./neo4j/data "
                "to reset)."
            ) from e
        except (ServiceUnavailable, OSError) as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"[neo4j] {uri} unreachable (attempt {attempt}/{max_retries}); "
                      f"retrying in {delay:.0f}s ...")
                time.sleep(delay)

    raise ConnectionError(
        f"Cannot reach Neo4j at {uri} after {max_retries} attempts. "
        "Is the container up?  ->  docker compose up -d neo4j\n"
        f"Last error: {last_err!r}"
    )


if __name__ == "__main__":
    drv = get_driver()
    with drv.session() as s:
        who = s.run("RETURN 'connected' AS status").single()["status"]
    print(f"[neo4j] {who} to {os.environ.get('NEO4J_URI', DEFAULT_URI)}")
    drv.close()
