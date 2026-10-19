"""Bounded local research crawling; remote content is untrusted source material.

This product includes software developed by UncleCode (https://x.com/unclecode)
as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).

Adapted from strmt7/FlitzZip tools/crawl4ai_research.py (MIT).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from tools.crawl4ai_tool import ATTRIBUTION, ROOT, VERSION, text_identity

MAX_MARKDOWN_BYTES = 2 * 1024 * 1024
SITES = ROOT / "tools/crawl4ai_sites.json"


def crawler():
    """Purpose: Construct the pinned upstream crawler
    Inputs: Installed package
    Outputs: Crawler owner."""
    if importlib.metadata.version("crawl4ai") != VERSION:
        raise ValueError("Crawler version does not match the reviewed integration")
    from crawl4ai import AsyncWebCrawler

    return AsyncWebCrawler()


def inspect_result(result, marker: str = "") -> dict:
    """Purpose: Reject empty/error/challenge extraction
    Inputs: Native result and marker
    Outputs: Metadata."""
    markdown = str(result.markdown.raw_markdown) if result.markdown else ""
    lowered = markdown.casefold()
    status = int(result.status_code or 0)
    valid = (
        bool(result.success)
        and 200 <= status < 300
        and 200 <= len(markdown.encode("utf-8")) <= MAX_MARKDOWN_BYTES
        and (not marker or marker.casefold() in lowered)
        and not any(
            text in lowered[:2000]
            for text in ("verify you are human", "checking your browser", "just a moment...")
        )
    )
    return {
        "success": valid,
        "native_success": bool(result.success),
        "http_status": status,
        "markdown_bytes": len(markdown.encode("utf-8")),
        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "expected_content": not marker or marker.casefold() in lowered,
        "error": str(result.error_message or "")[:500] if not valid else "",
    }


async def fetch(owner, url: str, marker: str = "") -> dict:
    """Purpose: Crawl one page with finite lifetime
    Inputs: Owner/URL/marker
    Outputs: Honest result metadata."""
    from crawl4ai import CacheMode, CrawlerRunConfig

    started = time.monotonic()
    row = {"url": url, "success": False}
    try:
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, check_robots_txt=True)
        result = await asyncio.wait_for(owner.arun(url=url, config=config), timeout=45)
        row.update(inspect_result(result, marker))
    except (ValueError, OSError, TimeoutError) as error:
        row["error"] = str(error)[:500]
    row["seconds"] = round(time.monotonic() - started, 3)
    return row


def source_identity() -> dict[str, str]:
    """Purpose: Bind qualification to integration bytes
    Inputs: Repository sources
    Outputs: Exact source digests."""
    paths = (
        "tools/crawl4ai_tool.py",
        "tools/crawl4ai_research.py",
        "tools/requirements/crawl4ai.txt",
        "tools/crawl4ai_sites.json",
    )
    return {path: text_identity(ROOT / path) for path in paths}


async def execute(args) -> int:
    """Purpose: Run the requested local lane
    Inputs: Parsed CLI
    Outputs: Exit status and bounded evidence."""
    if args.command == "doctor":
        from crawl4ai import CacheMode, CrawlerRunConfig

        async with crawler() as owner:
            result = await asyncio.wait_for(
                owner.arun(
                    url=(
                        "raw:<html><body><h1>model-protocol-lab crawler contract</h1>"
                        "<p>Local extraction.</p></body></html>"
                    ),
                    config=CrawlerRunConfig(cache_mode=CacheMode.DISABLED, word_count_threshold=0),
                ),
                timeout=45,
            )
        passed = result.success and "model-protocol-lab crawler contract" in str(result.markdown)
        print(
            json.dumps(
                {
                    "doctor_passed": bool(passed),
                    "version": VERSION,
                    "browser_configuration": "upstream-default",
                }
            )
        )
        return 0 if passed else 1
    sites = json.loads(SITES.read_text(encoding="utf-8"))["sites"]
    rows = []
    before = source_identity()
    async with crawler() as owner:
        for site in sites:
            row = await fetch(owner, site["url"], site["marker"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    successful_domains = {urlsplit(row["url"]).hostname for row in rows if row["success"]}
    qualified = len(successful_domains) >= 20 and before == source_identity()
    receipt = {
        "schema": 1,
        "completed_utc": datetime.now(UTC).isoformat(),
        "version": VERSION,
        "browser_configuration": "upstream-default",
        "robots_checked": True,
        "custom_browser_overrides": False,
        "source_identity": before,
        "attempted": len(rows),
        "successful_domains": len(successful_domains),
        "qualified": qualified,
        "results": rows,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(
        json.dumps({key: receipt[key] for key in ("attempted", "successful_domains", "qualified")}),
        flush=True,
    )
    return 0 if qualified else 1


def main() -> int:
    """Purpose: Parse explicit crawler lanes. Inputs: CLI. Outputs: Selected lane status."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=ATTRIBUTION)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    qualify = commands.add_parser("qualify")
    qualify.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="New local JSON evidence file; never overwrite",
    )
    return asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
