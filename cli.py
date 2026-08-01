#!/usr/bin/env python3
"""Unified command-line interface for the crypto-news and trading project.

The CLI intentionally imports the crawler and AI trading modules lazily. This
keeps database inspection and ``--help`` usable even when optional network or
AI dependencies are not installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

APP_NAME = "crypto-cli"
VERSION = "1.0.0"
DEFAULT_DB = "crypto_news.db"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DEPENDENCY = 3
EXIT_RUNTIME = 4
EXIT_INTERRUPTED = 130


class CliError(RuntimeError):
    """Expected user-facing CLI failure with an explicit exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_RUNTIME) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = non_negative_float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def percentage(value: str) -> float:
    parsed = non_negative_float(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def output_path(value: str) -> Path:
    return Path(value).expanduser()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _jsonable(value: Any) -> Any:
    """Convert common Pydantic/dataclass/runtime objects to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        public = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public:
            return _jsonable(public)
    return str(value)


def emit_json(payload: Any, *, compact: bool, destination: Path | None) -> None:
    text = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
    )
    if destination is None:
        print(text)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote {destination}", file=sys.stderr)


def parse_keywords(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for item in raw.split(","):
            keyword = item.strip()
            folded = keyword.casefold()
            if keyword and folded not in seen:
                seen.add(folded)
                result.append(keyword)
    return result or None


def ensure_parent_for_sqlite(db_path: str) -> None:
    if db_path == ":memory:" or db_path.startswith("file:"):
        return
    parent = Path(db_path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def command_crawl(args: argparse.Namespace) -> int:
    try:
        from crawler import CryptoNewsCrawler
    except ImportError as exc:
        raise CliError(
            "Crawler dependencies are unavailable. Install requests and "
            "beautifulsoup4, and keep crawler.py beside crypto_cli.py. "
            f"Original error: {exc}",
            EXIT_DEPENDENCY,
        ) from exc

    ensure_parent_for_sqlite(args.db)
    crawler: Any | None = None
    try:
        crawler = CryptoNewsCrawler(
            db_path=args.db,
            delay=args.delay,
            timeout=args.timeout,
            max_articles=args.max_articles,
            max_discovered=args.max_discovered,
            refresh=args.refresh,
        )
        saved, skipped, rejected = crawler.run()
    except sqlite3.Error as exc:
        raise CliError(f"SQLite failure: {exc}") from exc
    finally:
        if crawler is not None:
            crawler.close()

    summary = {
        "saved": saved,
        "skipped_existing": skipped,
        "rejected_or_failed": rejected,
        "database": args.db,
    }
    if args.json:
        emit_json(summary, compact=args.compact, destination=args.output)
    else:
        print(
            "Crawl complete: "
            f"saved={saved} skipped_existing={skipped} "
            f"rejected_or_failed={rejected} db={args.db}"
        )
    return EXIT_OK


def _load_trading_module() -> tuple[Any, Any]:
    try:
        import main as trading
        from data import TradingDeps
    except ImportError as exc:
        raise CliError(
            "Trading dependencies are unavailable. Ensure main.py, data.py, "
            "and crypto_tools.py are in the project, then install the packages "
            f"they import. Original error: {exc}",
            EXIT_DEPENDENCY,
        ) from exc
    except Exception as exc:
        raise CliError(
            "The trading module could not be initialized. Check model/API "
            f"configuration and environment variables. Original error: {exc}",
            EXIT_DEPENDENCY,
        ) from exc
    return trading, TradingDeps


def _default_request(symbol: str, timeframe: str) -> str:
    return (
        f"Analyze {symbol} on the {timeframe} timeframe. Propose a trade only "
        "when the EMA and recent-news evidence are clear and risk is bounded."
    )


async def _run_analysis(args: argparse.Namespace) -> tuple[Any, Any]:
    trading, TradingDeps = _load_trading_module()

    if args.fast_period >= args.slow_period:
        raise CliError(
            "--fast-period must be lower than --slow-period.",
            EXIT_USAGE,
        )

    deps = TradingDeps(
        user_id=args.user_id,
        paper_trading=args.paper_trading,
        max_risk_per_trade_pct=args.max_risk_per_trade_pct,
        minimum_reward_risk_ratio=args.minimum_reward_risk_ratio,
        news_db_path=args.db,
        require_news_for_trade=args.require_news,
        minimum_news_confidence=args.minimum_news_confidence,
    )

    return await trading.create_trade_plan(
        args.request or _default_request(args.symbol, args.timeframe),
        deps,
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        fast_period=args.fast_period,
        slow_period=args.slow_period,
        news_keywords=parse_keywords(args.news_keyword),
        news_time_filter=args.news_time_filter,
        news_limit=args.news_limit,
    )


def command_analyze(args: argparse.Namespace) -> int:
    if args.require_news and args.db != ":memory:" and not Path(args.db).expanduser().exists():
        raise CliError(
            f"News database not found: {args.db}. Run '{APP_NAME} crawl --db {args.db}' first, "
            "or pass --no-require-news.",
            EXIT_USAGE,
        )

    response, usage = asyncio.run(_run_analysis(args))
    payload: dict[str, Any] = {"result": response}
    if args.include_usage:
        payload["usage"] = usage

    emit_json(payload, compact=args.compact, destination=args.output)

    executable = bool(getattr(response, "executable", False))
    if args.fail_if_not_executable and not executable:
        return 10
    return EXIT_OK


def connect_news_db(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:" and not Path(db_path).expanduser().exists():
        raise CliError(f"Database not found: {db_path}", EXIT_USAGE)
    try:
        if db_path == ":memory:":
            conn = sqlite3.connect(db_path)
        else:
            resolved = Path(db_path).expanduser().resolve()
            conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
        ).fetchone()
        if table is None:
            conn.close()
            raise CliError(
                f"Database does not contain an articles table: {db_path}",
                EXIT_USAGE,
            )
        return conn
    except sqlite3.Error as exc:
        raise CliError(f"Could not open database {db_path}: {exc}") from exc


def command_db_stats(args: argparse.Namespace) -> int:
    with connect_news_db(args.db) as conn:
        overview = conn.execute(
            """
            SELECT
                COUNT(*) AS article_count,
                MIN(NULLIF(published_at, '')) AS earliest_published_at,
                MAX(NULLIF(published_at, '')) AS latest_published_at,
                MAX(NULLIF(scraped_at, '')) AS latest_scraped_at,
                COUNT(DISTINCT NULLIF(category, '')) AS category_count,
                COUNT(DISTINCT NULLIF(author, '')) AS author_count
            FROM articles
            """
        ).fetchone()
        categories = conn.execute(
            """
            SELECT COALESCE(NULLIF(category, ''), '(uncategorized)') AS category,
                   COUNT(*) AS article_count
            FROM articles
            GROUP BY COALESCE(NULLIF(category, ''), '(uncategorized)')
            ORDER BY article_count DESC, category ASC
            LIMIT ?
            """,
            (args.top_categories,),
        ).fetchall()

    payload = {
        "database": args.db,
        **dict(overview),
        "top_categories": [dict(row) for row in categories],
    }
    emit_json(payload, compact=args.compact, destination=args.output)
    return EXIT_OK


def _article_query(
    *,
    search: bool,
    limit: int,
    offset: int,
    query: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    fields = (
        "id, title, published_at, author, category, url, scraped_at, "
        "substr(content, 1, 240) AS excerpt"
    )
    if not search:
        return (
            f"SELECT {fields} FROM articles "
            "ORDER BY COALESCE(NULLIF(published_at, ''), scraped_at) DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        )

    pattern = f"%{query}%"
    return (
        f"SELECT {fields} FROM articles "
        "WHERE title LIKE ? COLLATE NOCASE "
        "OR description LIKE ? COLLATE NOCASE "
        "OR content LIKE ? COLLATE NOCASE "
        "OR author LIKE ? COLLATE NOCASE "
        "OR tags_json LIKE ? COLLATE NOCASE "
        "ORDER BY COALESCE(NULLIF(published_at, ''), scraped_at) DESC "
        "LIMIT ? OFFSET ?",
        (pattern, pattern, pattern, pattern, pattern, limit, offset),
    )


def print_article_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No articles found.")
        return

    print(f"{'ID':>6}  {'PUBLISHED':19}  {'CATEGORY':18}  TITLE")
    print(f"{'-' * 6}  {'-' * 19}  {'-' * 18}  {'-' * 50}")
    for row in rows:
        published = (row.get("published_at") or "")[:19]
        category = (row.get("category") or "")[:18]
        title = " ".join((row.get("title") or "").split())
        print(f"{row['id']:>6}  {published:19}  {category:18}  {title}")


def command_db_list(args: argparse.Namespace) -> int:
    sql, params = _article_query(search=False, limit=args.limit, offset=args.offset)
    with connect_news_db(args.db) as conn:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    if args.format == "json":
        emit_json(rows, compact=args.compact, destination=args.output)
    else:
        print_article_table(rows)
    return EXIT_OK


def command_db_search(args: argparse.Namespace) -> int:
    sql, params = _article_query(
        search=True,
        query=args.query,
        limit=args.limit,
        offset=args.offset,
    )
    with connect_news_db(args.db) as conn:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    if args.format == "json":
        emit_json(rows, compact=args.compact, destination=args.output)
    else:
        print_article_table(rows)
    return EXIT_OK


def add_json_output_options(parser: argparse.ArgumentParser, *, format_option: bool = False) -> None:
    if format_option:
        parser.add_argument(
            "--format",
            choices=("table", "json"),
            default="table",
            help="Output format (default: table).",
        )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    parser.add_argument(
        "--output",
        type=output_path,
        help="Write JSON to this file instead of standard output.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=(
            "Crawl crypto.news, inspect the SQLite article store, and run the "
            "multi-agent crypto trading analysis pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("CRYPTO_CLI_LOG_LEVEL", "INFO").upper(),
        help="Logging verbosity.",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    crawl = commands.add_parser(
        "crawl",
        help="Crawl crypto.news articles into SQLite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    crawl.add_argument("--db", default=DEFAULT_DB, help="SQLite database path.")
    crawl.add_argument(
        "--max-articles",
        type=positive_int,
        default=50,
        help="Maximum number of new or updated articles to save.",
    )
    crawl.add_argument(
        "--max-discovered",
        type=positive_int,
        default=500,
        help="Maximum number of discovered candidate URLs to retain.",
    )
    crawl.add_argument(
        "--delay",
        type=non_negative_float,
        default=1.0,
        help="Minimum delay between HTTP requests in seconds.",
    )
    crawl.add_argument(
        "--timeout",
        type=positive_float,
        default=25.0,
        help="Per-request timeout in seconds.",
    )
    crawl.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch and upsert articles already in the database.",
    )
    crawl.add_argument("--json", action="store_true", help="Print a JSON summary.")
    add_json_output_options(crawl)
    crawl.set_defaults(handler=command_crawl)

    analyze = commands.add_parser(
        "analyze",
        help="Create a risk-reviewed trade plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyze.add_argument("symbol", help="Trading symbol, for example BTCUSDT.")
    analyze.add_argument("--timeframe", default="1h", help="Chart timeframe.")
    analyze.add_argument("--request", help="Custom natural-language request.")
    analyze.add_argument("--fast-period", type=positive_int, default=20)
    analyze.add_argument("--slow-period", type=positive_int, default=50)
    analyze.add_argument(
        "--news-keyword",
        action="append",
        help="News keyword; repeat the option or supply comma-separated values.",
    )
    analyze.add_argument("--news-time-filter", default="24h")
    analyze.add_argument("--news-limit", type=positive_int, default=10)
    analyze.add_argument("--db", default=DEFAULT_DB, help="News SQLite database path.")
    analyze.add_argument("--user-id", default="cli-user")
    analyze.add_argument(
        "--paper-trading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a fully approved plan to be marked executable in paper mode.",
    )
    analyze.add_argument(
        "--max-risk-per-trade-pct",
        type=percentage,
        default=1.0,
    )
    analyze.add_argument(
        "--minimum-reward-risk-ratio",
        type=positive_float,
        default=2.0,
    )
    analyze.add_argument(
        "--require-news",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require relevant news before a trade can pass the risk gate.",
    )
    analyze.add_argument(
        "--minimum-news-confidence",
        type=probability,
        default=0.25,
    )
    analyze.add_argument(
        "--include-usage",
        action="store_true",
        help="Include model/tool usage details in the JSON output.",
    )
    analyze.add_argument(
        "--fail-if-not-executable",
        action="store_true",
        help="Exit with status 10 when the final plan is not executable.",
    )
    add_json_output_options(analyze)
    analyze.set_defaults(handler=command_analyze)

    db = commands.add_parser("db", help="Inspect the crawled-news SQLite database.")
    db_commands = db.add_subparsers(dest="db_command", required=True)

    stats = db_commands.add_parser(
        "stats",
        help="Show article-store statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stats.add_argument("--db", default=DEFAULT_DB)
    stats.add_argument("--top-categories", type=positive_int, default=10)
    add_json_output_options(stats)
    stats.set_defaults(handler=command_db_stats)

    list_parser = db_commands.add_parser(
        "list",
        help="List the newest stored articles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    list_parser.add_argument("--db", default=DEFAULT_DB)
    list_parser.add_argument("--limit", type=positive_int, default=20)
    list_parser.add_argument("--offset", type=int, default=0)
    add_json_output_options(list_parser, format_option=True)
    list_parser.set_defaults(handler=command_db_list)

    search = db_commands.add_parser(
        "search",
        help="Search stored article text and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    search.add_argument("query", help="Case-insensitive text query.")
    search.add_argument("--db", default=DEFAULT_DB)
    search.add_argument("--limit", type=positive_int, default=20)
    search.add_argument("--offset", type=int, default=0)
    add_json_output_options(search, format_option=True)
    search.set_defaults(handler=command_db_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "offset", 0) < 0:
        parser.error("--offset must be zero or greater")

    configure_logging(args.log_level)
    try:
        return int(args.handler(args))
    except CliError as exc:
        logging.error("%s", exc)
        return exc.exit_code
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.log_level == "DEBUG":
            logging.exception("Unhandled error")
        else:
            logging.error("%s", exc)
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.exit(main())
