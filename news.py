def get_latest_news(
    db_path="crypto_news.db",
    keywords=None,
    time_filter="24h",
    limit=20,
    match_all=False,
):
    """
    Read the latest matching news articles from the crawler's SQLite database.

    Args:
        db_path:
            SQLite database path.

        keywords:
            A string, comma-separated string, or list of strings.
            Examples:
                "bitcoin"
                "bitcoin, ETF"
                ["bitcoin", "ETF"]

        time_filter:
            Relative time:
                "30m", "6h", "24h", "7d", "2w"

            Or an ISO date/time:
                "2026-07-31"
                "2026-07-31T10:00:00Z"

            Use None to disable the time filter.

        limit:
            Maximum number of articles returned.

        match_all:
            False: match any keyword.
            True: match every keyword.

    Returns:
        A list of dictionaries ordered newest first.
    """
    import re
    import sqlite3
    from datetime import datetime, timedelta, timezone

    def parse_time(value):
        if value is None:
            return None

        value = str(value).strip()

        relative_match = re.fullmatch(
            r"(\d+)\s*(m|h|d|w)",
            value,
            flags=re.IGNORECASE,
        )

        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2).lower()

            units = {
                "m": timedelta(minutes=amount),
                "h": timedelta(hours=amount),
                "d": timedelta(days=amount),
                "w": timedelta(weeks=amount),
            }

            return datetime.now(timezone.utc) - units[unit]

        normalized = value.replace("Z", "+00:00")

        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                "time_filter must be like '30m', '24h', '7d', "
                "'2026-07-31', or an ISO timestamp"
            ) from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    if isinstance(keywords, str):
        keyword_list = [
            keyword.strip()
            for keyword in keywords.split(",")
            if keyword.strip()
        ]
    else:
        keyword_list = [
            str(keyword).strip()
            for keyword in (keywords or [])
            if str(keyword).strip()
        ]

    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")

    start_time = parse_time(time_filter)

    searchable_text = """
        (
            title || ' ' ||
            description || ' ' ||
            content || ' ' ||
            category || ' ' ||
            author || ' ' ||
            tags_json
        )
    """

    conditions = []
    parameters = []

    if start_time:
        conditions.append(
            """
            datetime(
                COALESCE(NULLIF(published_at, ''), scraped_at)
            ) >= datetime(?)
            """
        )
        parameters.append(
            start_time.replace(microsecond=0).isoformat()
        )

    if keyword_list:
        keyword_conditions = []

        for keyword in keyword_list:
            keyword_conditions.append(
                f"LOWER({searchable_text}) LIKE LOWER(?)"
            )
            parameters.append(f"%{keyword}%")

        operator = " AND " if match_all else " OR "
        conditions.append(
            "(" + operator.join(keyword_conditions) + ")"
        )

    where_clause = (
        "WHERE " + " AND ".join(conditions)
        if conditions
        else ""
    )

    query = f"""
        SELECT
            id,
            title,
            description,
            author,
            published_at,
            modified_at,
            category,
            tags_json,
            image_url,
            url,
            content,
            scraped_at
        FROM articles
        {where_clause}
        ORDER BY datetime(
            COALESCE(NULLIF(published_at, ''), scraped_at)
        ) DESC
        LIMIT ?
    """

    parameters.append(limit)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


# Example:
news = get_latest_news(
    db_path="crypto_news.db",
    keywords=["bitcoin", "ETF"],
    time_filter="24h",
    limit=10,
    match_all=False,
)

for article in news:
    print(article["published_at"])
    print(article["title"])
    print(article["url"])
    print()
