# Trading Agent

A command-line crypto research tool that combines EMA market data, locally
collected news, and LLM-based analysis to produce a structured, risk-reviewed
trade plan.

The application can:

- crawl public articles from [crypto.news](https://crypto.news/) into SQLite;
- search and inspect the local article database;
- retrieve Binance spot candles and calculate EMA signals;
- coordinate technical-analysis, news-analysis, and risk-review agents;
- enforce deterministic risk rules before marking a plan executable; and
- run through either a JSON-oriented CLI or an interactive terminal UI.

> [!IMPORTANT]
> This project is for research and experimentation, not financial advice. It
> does not place exchange orders. An `executable` result means only that a plan
> passed the application's checks and is eligible for paper trading.

## Requirements

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended), or another Python package
  manager
- Internet access for crawling news, querying Binance, and calling Groq
- A Groq API key

## Setup

Clone the repository and install its dependencies:

```bash
uv sync
```

Then create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

The project uses `python-dotenv`, so values in `.env` are loaded automatically
by the trading pipeline. You can alternatively export `GROQ_API_KEY` in your
shell.

## Quick start

First, collect news into the default `crypto_news.db` database:

```bash
uv run python cli.py crawl --max-articles 50
```

Inspect what was saved:

```bash
uv run python cli.py db stats
uv run python cli.py db list --limit 10
uv run python cli.py db search "bitcoin ETF"
```

Then generate a trade plan:

```bash
uv run python cli.py analyze BTCUSDT --timeframe 1h
```

Analysis is emitted as JSON and includes the technical report, news report,
draft trade, qualitative risk review, deterministic hard-risk result, and final
`executable` flag.

## Interactive terminal UI

Launch the full-screen interface with:

```bash
uv run python tui.py
```

The TUI provides dashboard, crawler, article browser, trade-analysis, and log
views. Analysis results are presented as a visual report with a decision
summary, trade levels, signal confidence, risk checks, and supporting evidence;
the underlying JSON can still be saved for automation. The report opens in a
dedicated panel, and the last result remains available while parameters are
edited or quickly rerun. It requires a terminal of at least 76 columns by 22
rows. On Windows, install `windows-curses` separately.

## CLI usage

Use the built-in help for the complete option list:

```bash
uv run python cli.py --help
uv run python cli.py crawl --help
uv run python cli.py analyze --help
uv run python cli.py db --help
```

### Crawl news

```bash
uv run python cli.py crawl \
  --db crypto_news.db \
  --max-articles 100 \
  --delay 1.5
```

The crawler respects `crypto.news/robots.txt`, rate-limits requests, retries
temporary failures, validates article pages, and deduplicates records by their
canonical URL. Use `--refresh` to re-fetch records already in the database.

### Analyze a market

```bash
uv run python cli.py analyze ETHUSDT \
  --timeframe 4h \
  --fast-period 20 \
  --slow-period 50 \
  --news-keyword ethereum \
  --news-keyword "ethereum ETF" \
  --news-time-filter 7d \
  --news-limit 15 \
  --include-usage
```

Useful policy options include:

- `--max-risk-per-trade-pct` (default `1.0`)
- `--minimum-reward-risk-ratio` (default `2.0`)
- `--minimum-news-confidence` (default `0.25`)
- `--no-require-news` to allow analysis without news evidence
- `--fail-if-not-executable` to return exit status `10` for rejected plans
- `--output result.json` to write JSON to a file

The fast EMA period must be lower than the slow EMA period. News keywords may
be repeated or supplied as comma-separated values.

### Inspect the database

```bash
uv run python cli.py db stats --top-categories 5
uv run python cli.py db list --limit 20 --format json
uv run python cli.py db search solana --limit 10 --format json
```

Database inspection is read-only. JSON-producing commands support `--compact`
and `--output PATH`.

## How analysis works

1. The technical specialist requests Binance candle data and evaluates EMA
   evidence.
2. The news specialist queries only the configured local SQLite database and
   assesses sentiment, impact, catalysts, and uncertainty.
3. The coordinator creates a structured buy, sell, or hold draft from those
   reports.
4. An independent LLM reviewer evaluates the proposed plan.
5. Application code applies hard checks for evidence consistency, news
   confidence, maximum risk, price ordering, and reward-to-risk ratio.

The final plan is marked executable only when both the qualitative review and
the deterministic gate approve it and paper trading is enabled. Missing or
weak evidence should result in `hold` or a rejected plan.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Authenticates the Pydantic AI Groq model | Required for analysis |
| `CRYPTO_CLI_LOG_LEVEL` | CLI/TUI logging verbosity | `INFO` |

The model is currently configured in `main.py` as
`openai/gpt-oss-120b` through Groq.

## Project structure

| File | Role |
| --- | --- |
| `cli.py` | Unified command-line interface |
| `tui.py` | Interactive curses terminal UI |
| `crawler.py` | `crypto.news` crawler and SQLite persistence |
| `main.py` | Agent definitions, orchestration, and hard-risk gate |
| `data.py` | Dependencies and Pydantic response models |
| `crypto_tools.py` | Agent-facing technical and news tools |
| `initicators.py` | Binance candle retrieval and EMA calculation |
| `news.py` | Queries and formats records from the news database |

## Development

Run lightweight syntax checks with:

```bash
uv run python -m compileall -q .
```

There is currently no automated test suite. Before changing trading logic,
add tests around the deterministic hard-risk gate and database queries first;
those are the most important safety boundaries in the application.

## Known limitations

- No live order execution or portfolio integration is implemented.
- Analysis depends on third-party availability and the quality/freshness of the
  locally crawled news.
- The current EMA tool implementation uses fixed Binance parameters internally;
  requested symbols, timeframes, and periods are not yet fully propagated to
  the data call.
- LLM output is probabilistic even though its schema and final risk gate are
  constrained.

## License

This project is licensed under the **MIT License**. You may use, copy, modify,
merge, publish, distribute, sublicense, and sell copies of the software,
provided that the copyright and permission notices are retained in copies or
substantial portions of the software.

The software is provided **as is**, without warranty, and the authors or
copyright holders are not liable for claims or damages arising from its use.

Copyright © 2026 Dewmal Handapangoda. See the [LICENSE](LICENSE) file for the
complete license text.
