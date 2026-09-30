#!/usr/bin/env python3
"""Download every PDF linked from the CLEF working-notes page (stdlib only)."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
from urllib.request import Request, urlopen


DEFAULT_URL = "https://clef-staging.pages.dev/"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"


def open_url(url):
    # The site rejects Python's default user agent with HTTP 403.
    return urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60)


class PDFLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href")
        if tag == "a" and href and urlsplit(href).path.lower().endswith(".pdf"):
            self.links.append(href)


def find_pdfs(url):
    with open_url(url) as response:
        parser = PDFLinks()
        parser.feed(response.read().decode(response.headers.get_content_charset() or "utf-8"))
        base_url = response.geturl()
    return list(dict.fromkeys(urldefrag(urljoin(base_url, link))[0] for link in parser.links))


def is_pdf(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(5) == b"%PDF-"


def download(url, destination):
    if is_pdf(destination):
        return "Skipped"
    temporary = destination.with_suffix(".pdf.part")
    for attempt in range(3):
        try:
            with open_url(url) as response, temporary.open("wb") as output:
                prefix = response.read(5)
                if prefix != b"%PDF-":
                    raise ValueError(f"Response is not a PDF: {url}")
                output.write(prefix)
                shutil.copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
                if expected is not None and output.tell() != int(expected):
                    raise ValueError(f"Incomplete download: {url}")
            temporary.replace(destination)
            return "Downloaded"
        except (OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise error
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Page containing PDF links")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        urls = find_pdfs(args.url)
        if not urls:
            raise ValueError("No PDF links found on the page")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        names = set()
        for index, url in enumerate(urls, 1):
            name = Path(unquote(urlsplit(url).path)).name
            while name in names:
                name = f"{index}_{name}"
            names.add(name)
            jobs.append((url, args.output_dir / name))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Found {len(jobs)} PDFs. Saving to {args.output_dir.resolve()}", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, url, path): path for url, path in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                status = future.result()
                print(f"[{count}/{len(jobs)}] {status}: {path.name}", flush=True)
            except Exception as error:
                failed += 1
                print(f"[{count}/{len(jobs)}] Failed: {path.name}: {error}", file=sys.stderr)
    print(f"Finished: {len(jobs) - failed} PDFs available, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
