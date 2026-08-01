#!/usr/bin/env python3
"""Interactive terminal UI for the crypto crawler and analysis project.

This module uses only the Python standard library for the interface. On Unix,
``curses`` is included with Python. On Windows, install ``windows-curses``.
"""

from __future__ import annotations

import argparse
import asyncio
import curses
import json
import logging
import os
import queue
import sqlite3
import sys
import textwrap
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cli import _jsonable, parse_keywords

APP_TITLE = "Crypto Research Terminal"
VERSION = "1.0.0"
DEFAULT_DB = "crypto_news.db"


@dataclass
class FormField:
    key: str
    label: str
    value: Any
    kind: str = "text"  # text, int, float, bool, action
    help: str = ""


class QueueLogHandler(logging.Handler):
    """Forward log records to the TUI event queue."""

    def __init__(self, events: queue.Queue[tuple[str, Any]]) -> None:
        super().__init__()
        self.events = events
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.events.put(("log", self.format(record)))
        except Exception:
            self.handleError(record)


class CryptoTUI:
    """A full-screen curses interface around the existing project modules."""

    VIEWS = ("Dashboard", "Crawler", "Articles", "Trade Analysis", "Logs", "Help")

    def __init__(self, screen: Any, *, db_path: str, log_level: str = "INFO") -> None:
        self.screen = screen
        self.db_path = db_path
        self.log_level = log_level
        self.running = True
        self.view_index = 0
        self.status = "Ready"
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crypto-tui")
        self.busy = False
        self.task_name = ""
        self.logs: deque[str] = deque(maxlen=500)
        self.log_scroll = 0
        self.dashboard: dict[str, Any] = {}
        self.articles: list[dict[str, Any]] = []
        self.article_query = ""
        self.article_offset = 0
        self.article_limit = 50
        self.article_selected = 0
        self.article_detail: dict[str, Any] | None = None
        self.article_detail_scroll = 0
        self.analysis_result = ""
        self.analysis_scroll = 0
        self.crawler_selected = 0
        self.crawler_scroll = 0
        self.analysis_selected = 0
        self.analysis_form_scroll = 0
        self.colors: dict[str, int] = {}

        self.crawler_fields = [
            FormField("db", "Database", db_path, "text", "SQLite output path"),
            FormField("max_articles", "Max articles", 50, "int", "Maximum saved articles"),
            FormField("max_discovered", "Max discovered", 500, "int", "Candidate URL cap"),
            FormField("delay", "Delay seconds", 1.0, "float", "Minimum delay between requests"),
            FormField("timeout", "Timeout seconds", 25.0, "float", "Per-request timeout"),
            FormField("refresh", "Refresh existing", False, "bool", "Re-fetch stored URLs"),
            FormField("run", "Start crawler", None, "action", "Run crawler in background"),
        ]
        self.analysis_fields = [
            FormField("symbol", "Symbol", "BTCUSDT", "text", "Trading pair"),
            FormField("timeframe", "Timeframe", "1h", "text", "Chart timeframe"),
            FormField("fast_period", "Fast EMA", 20, "int", "Must be below slow EMA"),
            FormField("slow_period", "Slow EMA", 50, "int", "Must exceed fast EMA"),
            FormField("keywords", "News keywords", "bitcoin,BTC,bitcoin ETF", "text", "Comma-separated"),
            FormField("news_filter", "News window", "24h", "text", "Examples: 24h, 7d"),
            FormField("news_limit", "News limit", 10, "int", "Maximum records"),
            FormField("db", "News database", db_path, "text", "SQLite article database"),
            FormField("paper", "Paper trading", True, "bool", "Never place a live order"),
            FormField("require_news", "Require news", True, "bool", "News required by hard gate"),
            FormField("max_risk", "Max risk %", 1.0, "float", "Risk-per-trade cap"),
            FormField("min_rr", "Minimum R:R", 2.0, "float", "Minimum reward/risk ratio"),
            FormField("min_conf", "News confidence", 0.25, "float", "Required value from 0 to 1"),
            FormField("request", "Custom request", "", "text", "Optional coordinator instruction"),
            FormField("run", "Run analysis", None, "action", "Create a risk-reviewed plan"),
        ]

        self._configure_terminal()
        self._install_log_handler()
        self.refresh_dashboard()
        self.refresh_articles()

    def _configure_terminal(self) -> None:
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.timeout(100)
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        if curses.has_colors():
            curses.start_color()
            palette = {
                "title": (curses.COLOR_CYAN, -1),
                "accent": (curses.COLOR_GREEN, -1),
                "warning": (curses.COLOR_YELLOW, -1),
                "error": (curses.COLOR_RED, -1),
                "muted": (curses.COLOR_BLUE, -1),
                "selected": (curses.COLOR_BLACK, curses.COLOR_CYAN),
            }
            for index, (name, colors) in enumerate(palette.items(), start=1):
                try:
                    curses.init_pair(index, *colors)
                    self.colors[name] = curses.color_pair(index)
                except curses.error:
                    self.colors[name] = 0

    def _install_log_handler(self) -> None:
        self.log_handler = QueueLogHandler(self.events)
        root = logging.getLogger()
        root.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        # Console handlers corrupt a full-screen curses display. Preserve them
        # and restore them when the application exits.
        self.previous_log_handlers = list(root.handlers)
        for handler in self.previous_log_handlers:
            root.removeHandler(handler)
        root.addHandler(self.log_handler)

    def close(self) -> None:
        root = logging.getLogger()
        root.removeHandler(self.log_handler)
        for handler in self.previous_log_handlers:
            root.addHandler(handler)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def style(self, name: str, extra: int = 0) -> int:
        return self.colors.get(name, 0) | extra

    def add(self, y: int, x: int, text: Any, attr: int = 0, *, max_width: int | None = None) -> None:
        height, width = self.screen.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        rendered = str(text).replace("\t", "    ")
        allowed = width - x - 1
        if max_width is not None:
            allowed = min(allowed, max_width)
        if allowed <= 0:
            return
        try:
            self.screen.addnstr(y, x, rendered, allowed, attr)
        except curses.error:
            pass

    def fill_line(self, y: int, char: str = " ", attr: int = 0) -> None:
        _, width = self.screen.getmaxyx()
        self.add(y, 0, char * max(0, width - 1), attr)

    def run(self) -> int:
        try:
            while self.running:
                self.process_events()
                self.draw()
                key = self.screen.getch()
                if key != -1:
                    self.handle_key(key)
            return 0
        finally:
            self.close()

    def process_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                self.logs.append(str(payload))
            elif kind == "done":
                task_name, result = payload
                self.busy = False
                self.task_name = ""
                self.status = f"{task_name} completed"
                if task_name == "Crawler":
                    self.logs.append(
                        "Crawler complete: "
                        f"saved={result['saved']} skipped={result['skipped_existing']} "
                        f"rejected={result['rejected_or_failed']}"
                    )
                    self.db_path = result["database"]
                    self.refresh_dashboard()
                    self.refresh_articles()
                elif task_name == "Trade analysis":
                    self.analysis_result = json.dumps(_jsonable(result), indent=2, ensure_ascii=False)
                    self.analysis_scroll = 0
            elif kind == "error":
                task_name, message = payload
                self.busy = False
                self.task_name = ""
                self.status = f"{task_name} failed"
                self.logs.append(f"ERROR {task_name}: {message}")
                self.show_message("Operation failed", str(message), error=True)

    def start_task(self, name: str, function: Callable[[], Any]) -> None:
        if self.busy:
            self.status = f"Already running: {self.task_name}"
            return
        self.busy = True
        self.task_name = name
        self.status = f"Running {name}..."
        self.logs.append(f"Starting {name}")

        def wrapped() -> None:
            try:
                result = function()
            except Exception as exc:
                logging.exception("%s failed", name)
                self.events.put(("error", (name, str(exc))))
            else:
                self.events.put(("done", (name, result)))

        self.executor.submit(wrapped)

    def draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 22 or width < 76:
            self.add(1, 2, "Terminal is too small", self.style("error", curses.A_BOLD))
            self.add(3, 2, f"Current: {width}x{height}. Required: at least 76x22.")
            self.add(5, 2, "Resize the terminal, or press q to quit.")
            self.screen.refresh()
            return

        self.draw_header()
        self.draw_navigation()
        content_x = 21
        content_width = width - content_x - 1
        content_height = height - 4
        view = self.VIEWS[self.view_index]
        if view == "Dashboard":
            self.draw_dashboard(2, content_x, content_height, content_width)
        elif view == "Crawler":
            self.draw_crawler(2, content_x, content_height, content_width)
        elif view == "Articles":
            self.draw_articles(2, content_x, content_height, content_width)
        elif view == "Trade Analysis":
            self.draw_analysis(2, content_x, content_height, content_width)
        elif view == "Logs":
            self.draw_logs(2, content_x, content_height, content_width)
        else:
            self.draw_help(2, content_x, content_height, content_width)
        self.draw_footer()
        self.screen.refresh()

    def draw_header(self) -> None:
        _, width = self.screen.getmaxyx()
        self.fill_line(0, " ", curses.A_REVERSE)
        self.add(0, 2, APP_TITLE, curses.A_REVERSE | curses.A_BOLD)
        right = f"v{VERSION}"
        self.add(0, max(2, width - len(right) - 2), right, curses.A_REVERSE)

    def draw_navigation(self) -> None:
        height, _ = self.screen.getmaxyx()
        self.add(2, 2, "NAVIGATION", self.style("title", curses.A_BOLD))
        for index, name in enumerate(self.VIEWS):
            marker = "●" if index == self.view_index else " "
            attr = self.style("selected", curses.A_BOLD) if index == self.view_index else 0
            self.add(4 + index * 2, 1, f" {index + 1} {marker} {name:<14}", attr, max_width=19)
        for y in range(1, height - 2):
            self.add(y, 20, "│", self.style("muted"))
        if self.busy:
            self.add(height - 5, 2, "BUSY", self.style("warning", curses.A_BOLD))
            self.add(height - 4, 2, self.task_name, max_width=17)

    def draw_footer(self) -> None:
        height, width = self.screen.getmaxyx()
        self.fill_line(height - 2, "─", self.style("muted"))
        status_attr = self.style("warning" if self.busy else "accent")
        self.add(height - 1, 1, self.status, status_attr, max_width=max(10, width - 43))
        keys = "1-6 views  Tab next  ? help  q quit"
        self.add(height - 1, max(1, width - len(keys) - 2), keys, self.style("muted"))

    def draw_section_title(self, y: int, x: int, title: str, width: int) -> None:
        self.add(y, x, title, self.style("title", curses.A_BOLD), max_width=width)
        self.add(y + 1, x, "─" * max(1, width - 1), self.style("muted"), max_width=width)

    def draw_dashboard(self, y: int, x: int, height: int, width: int) -> None:
        self.draw_section_title(y, x, "Dashboard", width)
        row = y + 3
        if "error" in self.dashboard:
            self.add(row, x, "Database unavailable", self.style("error", curses.A_BOLD))
            self.add(row + 2, x, self.dashboard["error"], max_width=width)
            self.add(row + 4, x, "Press 2 to configure and run the crawler.", self.style("warning"))
            return

        metrics = [
            ("Database", self.dashboard.get("database", self.db_path)),
            ("Articles", self.dashboard.get("article_count", 0)),
            ("Categories", self.dashboard.get("category_count", 0)),
            ("Authors", self.dashboard.get("author_count", 0)),
            ("Latest published", self.dashboard.get("latest_published_at") or "—"),
            ("Latest scrape", self.dashboard.get("latest_scraped_at") or "—"),
        ]
        label_width = 18
        for label, value in metrics:
            self.add(row, x, f"{label:<{label_width}}", curses.A_BOLD)
            self.add(row, x + label_width + 1, value, self.style("accent"), max_width=width - label_width - 2)
            row += 2

        categories = self.dashboard.get("top_categories", [])
        if categories and row < y + height - 3:
            self.add(row, x, "Top categories", self.style("title", curses.A_BOLD))
            row += 2
            max_count = max(int(item["article_count"]) for item in categories) or 1
            bar_width = max(8, min(32, width - 30))
            for item in categories[: min(6, max(1, y + height - row - 2))]:
                name = str(item["category"])[:18]
                count = int(item["article_count"])
                bars = max(1, round(count / max_count * bar_width))
                self.add(row, x, f"{name:<18} {count:>5} ", max_width=26)
                self.add(row, x + 26, "█" * bars, self.style("accent"), max_width=bar_width)
                row += 1

    def visible_form_window(self, fields: list[FormField], selected: int, available_rows: int) -> tuple[int, int]:
        size = max(1, available_rows)
        start = max(0, selected - size + 1)
        end = min(len(fields), start + size)
        return start, end

    def draw_form(
        self,
        fields: list[FormField],
        selected: int,
        y: int,
        x: int,
        rows: int,
        width: int,
    ) -> None:
        start, end = self.visible_form_window(fields, selected, rows)
        for screen_row, index in enumerate(range(start, end)):
            field = fields[index]
            attr = self.style("selected", curses.A_BOLD) if index == selected else 0
            if field.kind == "bool":
                value = "ON" if field.value else "OFF"
            elif field.kind == "action":
                value = "▶ RUN" if not self.busy else "BUSY"
            else:
                value = str(field.value) if field.value != "" else "—"
            line = f" {field.label:<20} {value}"
            self.add(y + screen_row * 2, x, line, attr, max_width=width)
            self.add(y + screen_row * 2 + 1, x + 3, field.help, self.style("muted"), max_width=width - 3)

    def draw_crawler(self, y: int, x: int, height: int, width: int) -> None:
        self.draw_section_title(y, x, "Crawler", width)
        self.add(y + 3, x, "↑/↓ select   Enter edit/run   Space toggle", self.style("muted"))
        form_y = y + 5
        rows = max(1, (height - 9) // 2)
        self.draw_form(self.crawler_fields, self.crawler_selected, form_y, x, rows, width)

    def draw_articles(self, y: int, x: int, height: int, width: int) -> None:
        title = "Articles"
        if self.article_query:
            title += f" — search: {self.article_query}"
        self.draw_section_title(y, x, title, width)
        if self.article_detail is not None:
            self.draw_article_detail(y + 3, x, height - 5, width)
            return

        self.add(y + 3, x, "/ search   Enter details   n/p page   r refresh", self.style("muted"))
        table_y = y + 5
        available = max(1, height - 8)
        if not self.articles:
            self.add(table_y, x, "No matching articles.", self.style("warning"))
            return

        self.add(table_y, x, f"{'DATE':16} {'CATEGORY':16} TITLE", curses.A_BOLD, max_width=width)
        self.add(table_y + 1, x, "─" * max(1, width - 1), self.style("muted"), max_width=width)
        visible_count = max(1, available - 2)
        start = max(0, self.article_selected - visible_count + 1)
        end = min(len(self.articles), start + visible_count)
        for row_index, index in enumerate(range(start, end), start=table_y + 2):
            article = self.articles[index]
            date = (article.get("published_at") or article.get("scraped_at") or "")[:16]
            category = (article.get("category") or "—")[:16]
            title_text = " ".join((article.get("title") or "").split())
            attr = self.style("selected", curses.A_BOLD) if index == self.article_selected else 0
            self.add(row_index, x, f"{date:16} {category:16} {title_text}", attr, max_width=width)
        page = self.article_offset // self.article_limit + 1
        self.add(y + height - 2, x, f"Page {page} • {len(self.articles)} rows loaded", self.style("muted"))

    def draw_article_detail(self, y: int, x: int, height: int, width: int) -> None:
        article = self.article_detail or {}
        self.add(y, x, "Article detail — Esc to return", self.style("accent", curses.A_BOLD))
        raw_lines = [
            f"Title: {article.get('title', '')}",
            f"Published: {article.get('published_at', '')}",
            f"Author: {article.get('author', '') or '—'}",
            f"Category: {article.get('category', '') or '—'}",
            f"URL: {article.get('url', '')}",
            "",
            article.get("description", "") or "",
            "",
            article.get("content", "") or "",
        ]
        wrapped: list[str] = []
        for line in raw_lines:
            wrapped.extend(textwrap.wrap(str(line), max(20, width - 2), replace_whitespace=False) or [""])
        max_scroll = max(0, len(wrapped) - max(1, height - 2))
        self.article_detail_scroll = min(self.article_detail_scroll, max_scroll)
        for index, line in enumerate(wrapped[self.article_detail_scroll : self.article_detail_scroll + height - 2]):
            self.add(y + 2 + index, x, line, max_width=width)

    def draw_analysis(self, y: int, x: int, height: int, width: int) -> None:
        self.draw_section_title(y, x, "Trade Analysis", width)
        if self.analysis_result:
            split = max(32, width // 2)
            self.add(y + 3, x, "Parameters", self.style("title", curses.A_BOLD))
            self.add(y + 3, x + split, "Latest result", self.style("title", curses.A_BOLD))
            form_rows = max(1, (height - 7) // 2)
            self.draw_form(self.analysis_fields, self.analysis_selected, y + 5, x, form_rows, split - 2)
            lines: list[str] = []
            for raw in self.analysis_result.splitlines():
                lines.extend(textwrap.wrap(raw, max(20, width - split - 2), replace_whitespace=False) or [""])
            shown = max(1, height - 6)
            max_scroll = max(0, len(lines) - shown)
            self.analysis_scroll = min(self.analysis_scroll, max_scroll)
            for index, line in enumerate(lines[self.analysis_scroll : self.analysis_scroll + shown]):
                self.add(y + 5 + index, x + split, line, max_width=width - split)
            self.add(y + height - 2, x + split, "PgUp/PgDn scroll • o save • x clear", self.style("muted"), max_width=width - split)
        else:
            self.add(y + 3, x, "↑/↓ select   Enter edit/run   Space toggle", self.style("muted"))
            rows = max(1, (height - 7) // 2)
            self.draw_form(self.analysis_fields, self.analysis_selected, y + 5, x, rows, width)

    def draw_logs(self, y: int, x: int, height: int, width: int) -> None:
        self.draw_section_title(y, x, "Logs", width)
        self.add(y + 3, x, "↑/↓ scroll   Home/End jump   c clear", self.style("muted"))
        rows = max(1, height - 7)
        all_lines: list[str] = []
        for entry in self.logs:
            all_lines.extend(textwrap.wrap(entry, max(20, width - 1), subsequent_indent="  ") or [""])
        max_scroll = max(0, len(all_lines) - rows)
        self.log_scroll = min(self.log_scroll, max_scroll)
        start = max(0, max_scroll - self.log_scroll)
        for index, line in enumerate(all_lines[start : start + rows]):
            attr = self.style("error") if "ERROR" in line else self.style("warning") if "WARNING" in line else 0
            self.add(y + 5 + index, x, line, attr, max_width=width)

    def draw_help(self, y: int, x: int, height: int, width: int) -> None:
        self.draw_section_title(y, x, "Keyboard Help", width)
        sections = [
            ("Global", "1–6 switch view  •  Tab/Shift+Tab cycle  •  ? help  •  q quit"),
            ("Forms", "↑/↓ select  •  Enter edit or run  •  Space toggle switches"),
            ("Articles", "/ search  •  Enter details  •  n/p page  •  r refresh  •  Esc close detail"),
            ("Analysis result", "PgUp/PgDn scroll  •  o save JSON  •  x clear result"),
            ("Logs", "↑/↓ scroll  •  Home/End jump  •  c clear"),
            ("Safety", "The analysis workflow is paper-trading by default. This UI does not submit exchange orders."),
        ]
        row = y + 3
        for heading, body in sections:
            self.add(row, x, heading, self.style("accent", curses.A_BOLD))
            row += 1
            for line in textwrap.wrap(body, max(24, width - 4)):
                self.add(row, x + 2, line, max_width=width - 2)
                row += 1
            row += 1

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), ord("Q")):
            if self.busy:
                self.status = f"Cannot quit while {self.task_name} is running"
            else:
                self.running = False
            return
        if key == ord("?"):
            self.view_index = self.VIEWS.index("Help")
            return
        if ord("1") <= key <= ord("6"):
            self.view_index = key - ord("1")
            return
        if key in (9, curses.KEY_RIGHT):
            self.view_index = (self.view_index + 1) % len(self.VIEWS)
            return
        if key in (curses.KEY_BTAB, curses.KEY_LEFT):
            self.view_index = (self.view_index - 1) % len(self.VIEWS)
            return

        view = self.VIEWS[self.view_index]
        if view == "Dashboard":
            self.handle_dashboard_key(key)
        elif view == "Crawler":
            self.handle_form_key(key, self.crawler_fields, "crawler")
        elif view == "Articles":
            self.handle_articles_key(key)
        elif view == "Trade Analysis":
            self.handle_analysis_key(key)
        elif view == "Logs":
            self.handle_logs_key(key)

    def handle_dashboard_key(self, key: int) -> None:
        if key in (ord("r"), ord("R")):
            self.refresh_dashboard()
            self.status = "Dashboard refreshed"
        elif key in (ord("c"), ord("C")):
            self.view_index = self.VIEWS.index("Crawler")
        elif key in (ord("a"), ord("A")):
            self.view_index = self.VIEWS.index("Articles")
        elif key in (ord("t"), ord("T")):
            self.view_index = self.VIEWS.index("Trade Analysis")

    def handle_form_key(self, key: int, fields: list[FormField], form_name: str) -> None:
        selected_attr = "crawler_selected" if form_name == "crawler" else "analysis_selected"
        selected = getattr(self, selected_attr)
        if key == curses.KEY_UP:
            selected = max(0, selected - 1)
        elif key == curses.KEY_DOWN:
            selected = min(len(fields) - 1, selected + 1)
        elif key in (10, 13, curses.KEY_ENTER, ord(" ")):
            field = fields[selected]
            if field.kind == "bool":
                field.value = not bool(field.value)
            elif field.kind == "action":
                if form_name == "crawler":
                    self.launch_crawler()
                else:
                    self.launch_analysis()
            elif key != ord(" "):
                self.edit_field(field)
        setattr(self, selected_attr, selected)

    def handle_articles_key(self, key: int) -> None:
        if self.article_detail is not None:
            if key == 27:
                self.article_detail = None
                self.article_detail_scroll = 0
            elif key in (curses.KEY_UP, ord("k")):
                self.article_detail_scroll = max(0, self.article_detail_scroll - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                self.article_detail_scroll += 1
            elif key == curses.KEY_PPAGE:
                self.article_detail_scroll = max(0, self.article_detail_scroll - 10)
            elif key == curses.KEY_NPAGE:
                self.article_detail_scroll += 10
            return

        if key == curses.KEY_UP:
            self.article_selected = max(0, self.article_selected - 1)
        elif key == curses.KEY_DOWN:
            self.article_selected = min(max(0, len(self.articles) - 1), self.article_selected + 1)
        elif key in (10, 13, curses.KEY_ENTER):
            self.open_selected_article()
        elif key == ord("/"):
            result = self.prompt("Search articles", self.article_query)
            if result is not None:
                self.article_query = result.strip()
                self.article_offset = 0
                self.refresh_articles()
        elif key in (ord("n"), curses.KEY_NPAGE):
            self.article_offset += self.article_limit
            self.refresh_articles()
        elif key in (ord("p"), curses.KEY_PPAGE):
            self.article_offset = max(0, self.article_offset - self.article_limit)
            self.refresh_articles()
        elif key in (ord("r"), ord("R")):
            self.refresh_articles()

    def handle_analysis_key(self, key: int) -> None:
        if self.analysis_result:
            if key == curses.KEY_PPAGE:
                self.analysis_scroll = max(0, self.analysis_scroll - 10)
                return
            if key == curses.KEY_NPAGE:
                self.analysis_scroll += 10
                return
            if key in (ord("o"), ord("O")):
                self.save_analysis_result()
                return
            if key in (ord("x"), ord("X")):
                self.analysis_result = ""
                self.analysis_scroll = 0
                return
        self.handle_form_key(key, self.analysis_fields, "analysis")

    def handle_logs_key(self, key: int) -> None:
        if key in (curses.KEY_UP, ord("k")):
            self.log_scroll += 1
        elif key in (curses.KEY_DOWN, ord("j")):
            self.log_scroll = max(0, self.log_scroll - 1)
        elif key == curses.KEY_HOME:
            self.log_scroll = 10_000
        elif key == curses.KEY_END:
            self.log_scroll = 0
        elif key in (ord("c"), ord("C")):
            self.logs.clear()
            self.log_scroll = 0
            self.status = "Logs cleared"

    def edit_field(self, field: FormField) -> None:
        current = str(field.value)
        result = self.prompt(field.label, current)
        if result is None:
            return
        try:
            if field.kind == "int":
                value = int(result)
                if value <= 0:
                    raise ValueError("must be greater than zero")
            elif field.kind == "float":
                value = float(result)
                if value < 0:
                    raise ValueError("must be zero or greater")
            else:
                value = result
            field.value = value
            self.status = f"Updated {field.label}"
        except ValueError as exc:
            self.show_message("Invalid value", f"{field.label}: {exc}", error=True)

    def prompt(self, title: str, initial: str = "") -> str | None:
        height, width = self.screen.getmaxyx()
        box_width = min(max(48, len(title) + 8), width - 4)
        box_height = 7
        y = max(1, (height - box_height) // 2)
        x = max(2, (width - box_width) // 2)
        window = curses.newwin(box_height, box_width, y, x)
        window.keypad(True)
        window.border()
        try:
            window.addnstr(1, 2, title, box_width - 4, curses.A_BOLD)
            window.addnstr(3, 2, initial, box_width - 4)
            window.addnstr(5, 2, "Enter accept • Esc cancel", box_width - 4, self.style("muted"))
        except curses.error:
            pass
        buffer = list(initial)
        cursor = len(buffer)
        curses.curs_set(1)
        while True:
            display = "".join(buffer)
            try:
                window.move(3, 2)
                window.clrtoeol()
                window.addnstr(3, 2, display, box_width - 4)
                window.move(3, min(box_width - 3, 2 + cursor))
            except curses.error:
                pass
            window.refresh()
            key = window.getch()
            if key in (10, 13, curses.KEY_ENTER):
                curses.curs_set(0)
                return "".join(buffer)
            if key == 27:
                curses.curs_set(0)
                return None
            if key in (curses.KEY_BACKSPACE, 127, 8):
                if cursor > 0:
                    del buffer[cursor - 1]
                    cursor -= 1
            elif key == curses.KEY_DC:
                if cursor < len(buffer):
                    del buffer[cursor]
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(buffer), cursor + 1)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = len(buffer)
            elif 32 <= key <= 0x10FFFF:
                try:
                    character = chr(key)
                except ValueError:
                    continue
                buffer.insert(cursor, character)
                cursor += 1

    def show_message(self, title: str, message: str, *, error: bool = False) -> None:
        height, width = self.screen.getmaxyx()
        wrapped = textwrap.wrap(message, max(30, min(72, width - 10))) or [""]
        box_width = min(width - 4, max(48, max(len(line) for line in wrapped) + 4))
        box_height = min(height - 4, len(wrapped) + 6)
        y = max(1, (height - box_height) // 2)
        x = max(2, (width - box_width) // 2)
        window = curses.newwin(box_height, box_width, y, x)
        window.keypad(True)
        window.border()
        attr = self.style("error" if error else "accent", curses.A_BOLD)
        try:
            window.addnstr(1, 2, title, box_width - 4, attr)
            for index, line in enumerate(wrapped[: box_height - 5]):
                window.addnstr(3 + index, 2, line, box_width - 4)
            window.addnstr(box_height - 2, 2, "Press any key", box_width - 4, self.style("muted"))
        except curses.error:
            pass
        window.refresh()
        window.getch()

    def confirm(self, message: str) -> bool:
        result = self.prompt(message + " [y/N]", "")
        return bool(result and result.strip().lower() in {"y", "yes"})

    def field_values(self, fields: list[FormField]) -> dict[str, Any]:
        return {field.key: field.value for field in fields if field.kind != "action"}

    def launch_crawler(self) -> None:
        values = self.field_values(self.crawler_fields)
        if int(values["max_articles"]) <= 0 or int(values["max_discovered"]) <= 0:
            self.show_message("Invalid crawler settings", "Article limits must be greater than zero.", error=True)
            return

        def task() -> dict[str, Any]:
            from crawler import CryptoNewsCrawler

            db = str(Path(str(values["db"])).expanduser())
            Path(db).resolve().parent.mkdir(parents=True, exist_ok=True)
            crawler: Any | None = None
            try:
                crawler = CryptoNewsCrawler(
                    db_path=db,
                    delay=float(values["delay"]),
                    timeout=float(values["timeout"]),
                    max_articles=int(values["max_articles"]),
                    max_discovered=int(values["max_discovered"]),
                    refresh=bool(values["refresh"]),
                )
                saved, skipped, rejected = crawler.run()
                return {
                    "saved": saved,
                    "skipped_existing": skipped,
                    "rejected_or_failed": rejected,
                    "database": db,
                }
            finally:
                if crawler is not None:
                    crawler.close()

        self.start_task("Crawler", task)

    def launch_analysis(self) -> None:
        values = self.field_values(self.analysis_fields)
        try:
            fast = int(values["fast_period"])
            slow = int(values["slow_period"])
            news_limit = int(values["news_limit"])
            risk = float(values["max_risk"])
            min_rr = float(values["min_rr"])
            min_conf = float(values["min_conf"])
            if fast <= 0 or slow <= 0 or fast >= slow:
                raise ValueError("Fast EMA must be positive and lower than slow EMA.")
            if news_limit <= 0:
                raise ValueError("News limit must be greater than zero.")
            if not 0 <= min_conf <= 1:
                raise ValueError("News confidence must be between 0 and 1.")
            if risk < 0 or risk > 100:
                raise ValueError("Max risk must be between 0 and 100 percent.")
            if min_rr <= 0:
                raise ValueError("Minimum reward/risk must be greater than zero.")
            db = str(Path(str(values["db"])).expanduser())
            if bool(values["require_news"]) and not Path(db).exists():
                raise ValueError(f"News database not found: {db}")
        except ValueError as exc:
            self.show_message("Invalid analysis settings", str(exc), error=True)
            return

        def task() -> dict[str, Any]:
            import main as trading
            from data import TradingDeps

            symbol = str(values["symbol"]).strip().upper()
            timeframe = str(values["timeframe"]).strip()
            request = str(values["request"]).strip() or (
                f"Analyze {symbol} on the {timeframe} timeframe. Propose a trade only "
                "when the EMA and recent-news evidence are clear and risk is bounded."
            )
            deps = TradingDeps(
                user_id="tui-user",
                paper_trading=bool(values["paper"]),
                max_risk_per_trade_pct=risk,
                minimum_reward_risk_ratio=min_rr,
                news_db_path=db,
                require_news_for_trade=bool(values["require_news"]),
                minimum_news_confidence=min_conf,
            )
            response, usage = asyncio.run(
                trading.create_trade_plan(
                    request,
                    deps,
                    symbol=symbol,
                    timeframe=timeframe,
                    fast_period=fast,
                    slow_period=slow,
                    news_keywords=parse_keywords([str(values["keywords"])]),
                    news_time_filter=str(values["news_filter"]),
                    news_limit=news_limit,
                )
            )
            return {"result": response, "usage": usage}

        self.start_task("Trade analysis", task)

    def refresh_dashboard(self) -> None:
        db = self.current_db_path()
        if not db.exists():
            self.dashboard = {"error": f"Database not found: {db}"}
            return
        try:
            with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
                ).fetchone()
                if table is None:
                    raise sqlite3.Error("articles table is missing")
                overview = conn.execute(
                    """
                    SELECT COUNT(*) AS article_count,
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
                    LIMIT 8
                    """
                ).fetchall()
            self.dashboard = {
                "database": str(db),
                **dict(overview),
                "top_categories": [dict(row) for row in categories],
            }
        except (sqlite3.Error, OSError) as exc:
            self.dashboard = {"error": f"Could not read {db}: {exc}"}

    def current_db_path(self) -> Path:
        for field in self.crawler_fields:
            if field.key == "db":
                return Path(str(field.value)).expanduser()
        return Path(self.db_path).expanduser()

    def refresh_articles(self) -> None:
        db = self.current_db_path()
        if not db.exists():
            self.articles = []
            self.article_selected = 0
            return
        try:
            with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                fields = (
                    "id, title, published_at, author, category, url, scraped_at, "
                    "substr(description, 1, 220) AS description, "
                    "substr(content, 1, 240) AS excerpt"
                )
                if self.article_query:
                    pattern = f"%{self.article_query}%"
                    sql = (
                        f"SELECT {fields} FROM articles "
                        "WHERE title LIKE ? COLLATE NOCASE OR description LIKE ? COLLATE NOCASE "
                        "OR content LIKE ? COLLATE NOCASE OR author LIKE ? COLLATE NOCASE "
                        "OR tags_json LIKE ? COLLATE NOCASE "
                        "ORDER BY COALESCE(NULLIF(published_at, ''), scraped_at) DESC LIMIT ? OFFSET ?"
                    )
                    params = (pattern, pattern, pattern, pattern, pattern, self.article_limit, self.article_offset)
                else:
                    sql = (
                        f"SELECT {fields} FROM articles "
                        "ORDER BY COALESCE(NULLIF(published_at, ''), scraped_at) DESC LIMIT ? OFFSET ?"
                    )
                    params = (self.article_limit, self.article_offset)
                self.articles = [dict(row) for row in conn.execute(sql, params).fetchall()]
            self.article_selected = min(self.article_selected, max(0, len(self.articles) - 1))
            self.status = f"Loaded {len(self.articles)} articles"
        except (sqlite3.Error, OSError) as exc:
            self.articles = []
            self.status = f"Could not load articles: {exc}"
            self.logs.append(f"ERROR database: {exc}")

    def open_selected_article(self) -> None:
        if not self.articles:
            return
        article_id = self.articles[self.article_selected]["id"]
        db = self.current_db_path()
        try:
            with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
            if row is not None:
                self.article_detail = dict(row)
                self.article_detail_scroll = 0
        except (sqlite3.Error, OSError) as exc:
            self.show_message("Could not open article", str(exc), error=True)

    def save_analysis_result(self) -> None:
        if not self.analysis_result:
            return
        result = self.prompt("Save result to file", "trade_analysis.json")
        if result is None or not result.strip():
            return
        try:
            path = Path(result.strip()).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.analysis_result + "\n", encoding="utf-8")
            self.status = f"Saved {path}"
        except OSError as exc:
            self.show_message("Could not save result", str(exc), error=True)


def run_tui(*, db_path: str = DEFAULT_DB, log_level: str = "INFO") -> int:
    """Launch the application and return its process exit status."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("The terminal UI requires an interactive TTY.", file=sys.stderr)
        return 2
    try:
        return int(curses.wrapper(lambda screen: CryptoTUI(screen, db_path=db_path, log_level=log_level).run()))
    except KeyboardInterrupt:
        return 130
    except curses.error as exc:
        print(f"Could not initialize terminal UI: {exc}", file=sys.stderr)
        return 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive terminal UI for the crypto project.")
    parser.add_argument("--db", default=DEFAULT_DB, help="Default SQLite article database.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("CRYPTO_CLI_LOG_LEVEL", "INFO").upper(),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_tui(db_path=args.db, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
