"""
Crawl public articles from https://crypto.news/ and store them in SQLite.

Install dependencies:
    python -m pip install requests beautifulsoup4

Examples:
    python crypto_news_crawler.py
    python crypto_news_crawler.py --db crypto_news.db --max-articles 100 --delay 1.5
    python crypto_news_crawler.py --refresh --max-articles 25

The crawler:
- respects crypto.news/robots.txt
- discovers URLs through RSS, news sitemaps, the sitemap index, and homepage links
- rate-limits requests and retries temporary failures
- validates pages as articles before storing them
- deduplicates/upserts records by canonical URL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

try:
    import requests
    from bs4 import BeautifulSoup, Tag
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: "
        "python -m pip install requests beautifulsoup4"
    ) from exc


BASE_URL = "https://crypto.news/"
DOMAIN = "crypto.news"
USER_AGENT = "CryptoNewsSQLiteCrawler/1.0 (+personal research crawler)"
DEFAULT_DB = "crypto_news.db"

# These paths are utility, account, search, price, taxonomy, or other non-article pages.
EXCLUDED_PREFIXES = (
    "/wp/",
    "/cdn-cgi/",
    "/search/",
    "/download/",
    "/author/",
    "/tag/",
    "/category/",
    "/prices/",
    "/price/",
    "/events/",
    "/glossary/",
    "/newsletters/",
    "/about/",
    "/contact-us/",
    "/advertise/",
    "/privacy-policy/",
    "/terms-of-service/",
    "/cookie-policy/",
    "/sitemap/",
    "/feed/",
)

STOP_TEXT = {
    "read more about",
    "best crypto platforms",
    "sign up for crypto news and market insights",
    "related news",
    "latest news",
}

SPACE_RE = re.compile(r"\s+")


@dataclass
class Article:
    url: str
    source_url: str
    title: str
    description: str
    author: str
    published_at: str
    modified_at: str
    category: str
    tags: list[str]
    image_url: str
    content: str
    language: str
    json_ld: str
    content_hash: str
    scraped_at: str
    http_status: int


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(clean_text(item) for item in value if item is not None)
    return SPACE_RE.sub(" ", str(value)).strip()


def normalize_url(url: str, base: str = BASE_URL) -> str:
    absolute = urljoin(base, clean_text(url))
    parts = urlsplit(absolute)
    scheme = "https"
    host = (parts.hostname or "").lower()
    host = host.removeprefix("www.")
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and not path.endswith("/"):
        path += "/"
    return urlunsplit((scheme, host, path, "", ""))


def is_same_domain(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    return host == DOMAIN


def is_candidate_url(url: str) -> bool:
    if not is_same_domain(url):
        return False
    path = urlsplit(url).path.lower()
    if path in ("", "/"):
        return False
    if any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if path.endswith((".xml/", ".xml", ".json/", ".json", ".jpg/", ".png/")):
        return False
    return True


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def meta_content(soup: BeautifulSoup, *keys: tuple[str, str]) -> str:
    for attr, value in keys:
        node = soup.find("meta", attrs={attr: value})
        if node and node.get("content"):
            return clean_text(node["content"])
    return ""


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def flatten_json_ld(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        graph = value.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from flatten_json_ld(item)
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten_json_ld(item)


def parse_json_ld(soup: BeautifulSoup) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    objects: list[dict[str, Any]] = []
    article_obj: dict[str, Any] | None = None
    accepted_types = {
        "article",
        "newsarticle",
        "reportagenewsarticle",
        "analysisnewsarticle",
        "opinionnewsarticle",
        "blogposting",
    }

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text("", strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for obj in flatten_json_ld(parsed):
            objects.append(obj)
            types = {clean_text(t).lower() for t in as_list(obj.get("@type"))}
            if types & accepted_types and article_obj is None:
                article_obj = obj

    return objects, article_obj


def extract_person_name(value: Any) -> str:
    names: list[str] = []
    for item in as_list(value):
        if isinstance(item, dict):
            name = first_nonempty(item.get("name"), item.get("alternateName"))
        else:
            name = clean_text(item)
        if name and name not in names:
            names.append(name)
    return ", ".join(names)


def extract_image(value: Any) -> str:
    for item in as_list(value):
        if isinstance(item, str):
            return clean_text(item)
        if isinstance(item, dict):
            candidate = first_nonempty(item.get("url"), item.get("contentUrl"))
            if candidate:
                return candidate
    return ""


def extract_tags(article_obj: dict[str, Any] | None, soup: BeautifulSoup) -> list[str]:
    tags: list[str] = []
    if article_obj:
        for value in as_list(article_obj.get("keywords")):
            if isinstance(value, str):
                candidates = re.split(r"[,;]", value)
            else:
                candidates = [clean_text(value)]
            for candidate in candidates:
                candidate = clean_text(candidate)
                if candidate and candidate.lower() not in {tag.lower() for tag in tags}:
                    tags.append(candidate)

    keyword_meta = meta_content(soup, ("name", "keywords"))
    for candidate in re.split(r"[,;]", keyword_meta):
        candidate = clean_text(candidate)
        if candidate and candidate.lower() not in {tag.lower() for tag in tags}:
            tags.append(candidate)
    return tags


def tag_text_length(node: Tag) -> int:
    return sum(len(clean_text(p.get_text(" ", strip=True))) for p in node.find_all("p"))


def link_text_length(node: Tag) -> int:
    return sum(len(clean_text(a.get_text(" ", strip=True))) for a in node.find_all("a"))


def choose_content_root(soup: BeautifulSoup) -> Tag | None:
    selectors = (
        "[itemprop='articleBody']",
        ".article-content",
        ".article__content",
        ".post-content",
        ".post__content",
        ".entry-content",
        ".single-post-content",
        "article",
        "main",
    )
    candidates: list[Tag] = []
    seen: set[int] = set()
    for selector in selectors:
        for node in soup.select(selector):
            if id(node) not in seen:
                seen.add(id(node))
                candidates.append(node)

    best: Tag | None = None
    best_score = -1.0
    for node in candidates:
        text_len = tag_text_length(node)
        if text_len < 150:
            continue
        links_len = link_text_length(node)
        # Prefer substantial paragraph text and penalize navigation-heavy containers.
        score = text_len - (links_len * 0.65)
        if node.name == "article" or node.get("itemprop") == "articleBody":
            score *= 1.15
        if score > best_score:
            best_score = score
            best = node
    return best


def extract_content(soup: BeautifulSoup, article_obj: dict[str, Any] | None) -> str:
    if article_obj:
        structured_body = clean_text(article_obj.get("articleBody"))
        if len(structured_body) >= 250:
            return structured_body

    root = choose_content_root(soup)
    if root is None:
        return ""

    # Work on a detached copy so cleanup cannot affect other extraction.
    fragment = BeautifulSoup(str(root), "html.parser")
    for unwanted in fragment.select(
        "script, style, noscript, iframe, form, nav, footer, aside, "
        ".advertisement, .ads, .ad, [class*='share'], [class*='social'], "
        "[class*='newsletter'], [class*='related']"
    ):
        unwanted.decompose()

    blocks: list[str] = []
    total_chars = 0
    for node in fragment.find_all(["h2", "h3", "p", "blockquote", "li"]):
        text = clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        lowered = text.lower().rstrip(":")
        if total_chars >= 400 and any(lowered.startswith(stop) for stop in STOP_TEXT):
            break
        if node.name in {"h2", "h3"}:
            text = f"\n{text}\n"
        if not blocks or text != blocks[-1]:
            blocks.append(text)
            total_chars += len(text)

    content = "\n\n".join(blocks)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


def is_article_page(
    soup: BeautifulSoup,
    article_obj: dict[str, Any] | None,
    title: str,
    published_at: str,
    content: str,
) -> bool:
    if article_obj is not None:
        return bool(title and (published_at or len(content) >= 200))
    og_type = meta_content(soup, ("property", "og:type"), ("name", "og:type")).lower()
    return og_type == "article" and bool(title and len(content) >= 200)


class CryptoNewsCrawler:
    def __init__(
        self,
        db_path: str,
        delay: float,
        timeout: float,
        max_articles: int,
        max_discovered: int,
        refresh: bool,
    ) -> None:
        self.db_path = db_path
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.max_articles = max(1, max_articles)
        self.max_discovered = max(self.max_articles, max_discovered)
        self.refresh = refresh
        self.last_request_at = 0.0
        self.session = self._build_session()
        self.robots = self._load_robots()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
                "Cache-Control": "no-cache",
            }
        )
        return session

    def _polite_wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _request(self, url: str, *, check_robots: bool = True) -> requests.Response | None:
        url = normalize_url(url) if not url.endswith(".xml") else url
        if check_robots and self.robots and not self.robots.can_fetch(USER_AGENT, url):
            logging.warning("Blocked by robots.txt: %s", url)
            return None
        self._polite_wait()
        try:
            response = self.session.get(url, timeout=self.timeout)
            self.last_request_at = time.monotonic()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            self.last_request_at = time.monotonic()
            logging.warning("Request failed for %s: %s", url, exc)
            return None

    def _load_robots(self) -> RobotFileParser | None:
        robots_url = urljoin(BASE_URL, "robots.txt")
        self._polite_wait()
        try:
            response = self.session.get(robots_url, timeout=self.timeout)
            self.last_request_at = time.monotonic()
            response.raise_for_status()
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            return parser
        except requests.RequestException as exc:
            logging.warning("Could not load robots.txt (%s); continuing conservatively.", exc)
            return None

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                published_at TEXT NOT NULL DEFAULT '',
                modified_at TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                image_url TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                json_ld TEXT NOT NULL DEFAULT '[]',
                content_hash TEXT NOT NULL DEFAULT '',
                scraped_at TEXT NOT NULL,
                http_status INTEGER NOT NULL DEFAULT 200
            );

            CREATE INDEX IF NOT EXISTS idx_articles_published_at
                ON articles(published_at DESC);
            CREATE INDEX IF NOT EXISTS idx_articles_category
                ON articles(category);
            CREATE INDEX IF NOT EXISTS idx_articles_content_hash
                ON articles(content_hash);
            """
        )
        self.conn.commit()

    def _already_stored(self, url: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
        return row is not None

    def _discover_from_feed(self) -> list[str]:
        response = self._request(urljoin(BASE_URL, "feed/"))
        if response is None:
            return []
        urls: list[str] = []
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            logging.warning("Could not parse RSS feed: %s", exc)
            return []

        for item in root.findall(".//item"):
            link = item.findtext("link")
            if link:
                url = normalize_url(link)
                if is_candidate_url(url) and url not in urls:
                    urls.append(url)
        logging.info("RSS discovered %d candidate URLs", len(urls))
        return urls

    @staticmethod
    def _xml_local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _parse_sitemap(self, sitemap_url: str) -> tuple[list[str], list[tuple[str, str]]]:
        response = self._request(sitemap_url, check_robots=False)
        if response is None:
            return [], []
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            logging.warning("Could not parse sitemap %s: %s", sitemap_url, exc)
            return [], []

        root_type = self._xml_local_name(root.tag)
        article_urls: list[tuple[str, str]] = []
        child_sitemaps: list[str] = []

        if root_type == "sitemapindex":
            for node in root:
                loc = ""
                for child in node:
                    if self._xml_local_name(child.tag) == "loc":
                        loc = clean_text(child.text)
                        break
                if loc:
                    child_sitemaps.append(loc)
        elif root_type == "urlset":
            for node in root:
                loc = ""
                lastmod = ""
                for child in node:
                    name = self._xml_local_name(child.tag)
                    if name == "loc":
                        loc = clean_text(child.text)
                    elif name == "lastmod":
                        lastmod = clean_text(child.text)
                if loc:
                    url = normalize_url(loc)
                    if is_candidate_url(url):
                        article_urls.append((url, lastmod))
        return child_sitemaps, article_urls

    def _discover_from_sitemaps(self) -> list[str]:
        queue = [
            urljoin(BASE_URL, "news-sitemap.xml"),
            urljoin(BASE_URL, "cn_cryptocurrency-sitemap.xml"),
            urljoin(BASE_URL, "sitemap_index.xml"),
        ]
        visited: set[str] = set()
        discovered: dict[str, str] = {}
        sitemap_limit = 16

        while queue and len(visited) < sitemap_limit and len(discovered) < self.max_discovered:
            sitemap_url = queue.pop(0)
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            children, articles = self._parse_sitemap(sitemap_url)
            for url, lastmod in articles:
                discovered[url] = max(discovered.get(url, ""), lastmod)
                if len(discovered) >= self.max_discovered:
                    break

            for child in children:
                lower = child.lower()
                # Prefer post/news sitemaps. Avoid taxonomy, author, page, and image maps.
                if any(token in lower for token in ("post", "news", "cryptocurrency")) and not any(
                    token in lower for token in ("image", "author", "category", "tag", "page-sitemap")
                ):
                    queue.append(child)

        urls = [url for url, _ in sorted(discovered.items(), key=lambda item: item[1], reverse=True)]
        logging.info("Sitemaps discovered %d candidate URLs", len(urls))
        return urls

    def _discover_from_homepage(self) -> list[str]:
        response = self._request(BASE_URL)
        if response is None:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        urls: list[str] = []
        for anchor in soup.find_all("a", href=True):
            url = normalize_url(anchor["href"], response.url)
            if is_candidate_url(url) and url not in urls:
                urls.append(url)
        logging.info("Homepage discovered %d candidate URLs", len(urls))
        return urls

    def discover_urls(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        sources: Iterable[list[str]] = (
            self._discover_from_feed(),
            self._discover_from_sitemaps(),
            self._discover_from_homepage(),
        )
        for source_urls in sources:
            for url in source_urls:
                if url not in seen:
                    seen.add(url)
                    ordered.append(url)
                    if len(ordered) >= self.max_discovered:
                        return ordered
        return ordered

    def parse_article(self, source_url: str) -> Article | None:
        response = self._request(source_url)
        if response is None:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        json_ld_objects, article_obj = parse_json_ld(soup)

        canonical_node = soup.find("link", rel=lambda value: value and "canonical" in value)
        canonical = normalize_url(
            canonical_node.get("href") if canonical_node and canonical_node.get("href") else response.url,
            response.url,
        )
        if not is_same_domain(canonical):
            canonical = normalize_url(response.url)

        title = first_nonempty(
            article_obj.get("headline") if article_obj else "",
            meta_content(soup, ("property", "og:title"), ("name", "twitter:title")),
            soup.h1.get_text(" ", strip=True) if soup.h1 else "",
            soup.title.get_text(" ", strip=True) if soup.title else "",
        )
        description = first_nonempty(
            article_obj.get("description") if article_obj else "",
            meta_content(
                soup,
                ("name", "description"),
                ("property", "og:description"),
                ("name", "twitter:description"),
            ),
        )
        author = first_nonempty(
            extract_person_name(article_obj.get("author")) if article_obj else "",
            meta_content(soup, ("name", "author"), ("property", "article:author")),
        )
        published_at = first_nonempty(
            article_obj.get("datePublished") if article_obj else "",
            meta_content(
                soup,
                ("property", "article:published_time"),
                ("name", "article:published_time"),
                ("name", "date"),
            ),
        )
        modified_at = first_nonempty(
            article_obj.get("dateModified") if article_obj else "",
            meta_content(
                soup,
                ("property", "article:modified_time"),
                ("name", "article:modified_time"),
            ),
        )
        category = first_nonempty(
            article_obj.get("articleSection") if article_obj else "",
            meta_content(soup, ("property", "article:section")),
        )
        image_url = first_nonempty(
            extract_image(article_obj.get("image")) if article_obj else "",
            meta_content(soup, ("property", "og:image"), ("name", "twitter:image")),
        )
        language = first_nonempty(
            article_obj.get("inLanguage") if article_obj else "",
            soup.html.get("lang") if soup.html else "",
        )
        tags = extract_tags(article_obj, soup)
        content = extract_content(soup, article_obj)

        if not is_article_page(soup, article_obj, title, published_at, content):
            logging.debug("Not an article page: %s", source_url)
            return None

        content_hash = hashlib.sha256(
            (title + "\n" + content).encode("utf-8", errors="replace")
        ).hexdigest()
        return Article(
            url=canonical,
            source_url=normalize_url(source_url),
            title=title,
            description=description,
            author=author,
            published_at=published_at,
            modified_at=modified_at,
            category=category,
            tags=tags,
            image_url=image_url,
            content=content,
            language=language,
            json_ld=json.dumps(json_ld_objects, ensure_ascii=False, separators=(",", ":")),
            content_hash=content_hash,
            scraped_at=utc_now(),
            http_status=response.status_code,
        )

    def save_article(self, article: Article) -> None:
        self.conn.execute(
            """
            INSERT INTO articles (
                url, source_url, title, description, author, published_at,
                modified_at, category, tags_json, image_url, content, language,
                json_ld, content_hash, scraped_at, http_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                source_url = excluded.source_url,
                title = excluded.title,
                description = excluded.description,
                author = excluded.author,
                published_at = excluded.published_at,
                modified_at = excluded.modified_at,
                category = excluded.category,
                tags_json = excluded.tags_json,
                image_url = excluded.image_url,
                content = excluded.content,
                language = excluded.language,
                json_ld = excluded.json_ld,
                content_hash = excluded.content_hash,
                scraped_at = excluded.scraped_at,
                http_status = excluded.http_status
            """,
            (
                article.url,
                article.source_url,
                article.title,
                article.description,
                article.author,
                article.published_at,
                article.modified_at,
                article.category,
                json.dumps(article.tags, ensure_ascii=False),
                article.image_url,
                article.content,
                article.language,
                article.json_ld,
                article.content_hash,
                article.scraped_at,
                article.http_status,
            ),
        )
        self.conn.commit()

    def run(self) -> tuple[int, int, int]:
        candidates = self.discover_urls()
        logging.info("Total unique candidates: %d", len(candidates))
        saved = 0
        skipped_existing = 0
        rejected = 0

        for position, url in enumerate(candidates, start=1):
            if saved >= self.max_articles:
                break
            if not self.refresh and self._already_stored(url):
                skipped_existing += 1
                continue

            logging.info("[%d/%d] Crawling %s", position, len(candidates), url)
            article = self.parse_article(url)
            if article is None:
                rejected += 1
                continue

            # A redirect/canonical may point to an already stored URL.
            if not self.refresh and article.url != url and self._already_stored(article.url):
                skipped_existing += 1
                continue

            self.save_article(article)
            saved += 1
            logging.info("Saved: %s", article.title)

        return saved, skipped_existing, rejected

    def close(self) -> None:
        self.conn.close()
        self.session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl crypto.news articles into a SQLite database."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum number of new/updated articles to save (default: 50)",
    )
    parser.add_argument(
        "--max-discovered",
        type=int,
        default=500,
        help="Maximum candidate URLs retained from discovery (default: 500)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Minimum seconds between HTTP requests (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="Per-request timeout in seconds (default: 25)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch and upsert URLs already present in the database",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    crawler: CryptoNewsCrawler | None = None
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
        print(
            f"Done. saved={saved} skipped_existing={skipped} "
            f"rejected_or_failed={rejected} db={args.db}"
        )
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 130
    except sqlite3.Error as exc:
        logging.error("SQLite error: %s", exc)
        return 2
    finally:
        if crawler is not None:
            crawler.close()


if __name__ == "__main__":
    sys.exit(main())
