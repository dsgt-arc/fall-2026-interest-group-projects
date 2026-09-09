"""Read the CEUR catalog and perform exact, parameterized metadata lookups."""

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from clef_rag.config import CATALOG_URL, RagError


def normalized(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text).casefold()
        if not unicodedata.combining(c)
    )


def read_catalog(index_dir, html_file=None, refresh=False):
    snapshots = index_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    previous = sorted(snapshots.glob("*.html"))
    if html_file:
        html = html_file.read_bytes()
    elif previous and not refresh:
        path = previous[-1]
        return parse_catalog(path.read_bytes()), path
    else:
        try:
            with urlopen(
                Request(CATALOG_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60
            ) as response:
                html = response.read()
        except OSError as exc:
            raise RagError(
                "Cannot fetch the catalog. Use --catalog-html with a saved page, or retry."
            ) from exc
    documents = parse_catalog(html)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = snapshots / f"{stamp}-{hashlib.sha256(html).hexdigest()[:10]}.html"
    path.write_bytes(html)
    return documents, path


def parse_catalog(html):
    soup = BeautifulSoup(html, "html.parser")
    toc = soup.select_one(".CEURTOC")
    if toc is None:
        raise RagError("The catalog has no CEUR table of contents.")
    track = "Supporting documents"
    documents = []
    seen = set()
    for element in toc.select(".CEURSESSION, li"):
        if "CEURSESSION" in element.get("class", []):
            track = element.get_text(" ", strip=True)
            continue
        link = next(
            (
                a
                for a in element.select("a[href]")
                if urlsplit(a["href"]).path.lower().endswith(".pdf")
            ),
            None,
        )
        if link is None:
            continue
        url = urljoin(CATALOG_URL, link["href"])
        filename = Path(unquote(urlsplit(url).path)).name
        doc_id = element.get("id") or Path(filename).stem
        if doc_id in seen:
            raise RagError(f"Duplicate catalog document ID: {doc_id}")
        seen.add(doc_id)
        title = element.select_one(".CEURTITLE")
        authors = [a.get_text(" ", strip=True) for a in element.select(".CEURAUTHOR")]
        pages = element.select_one(".CEURPAGES")
        documents.append(
            {
                "id": doc_id,
                "title": (title or link).get_text(" ", strip=True),
                "authors": authors,
                "author_search": normalized("; ".join(authors)),
                "track": track,
                "url": url,
                "filename": filename,
                "kind": "paper" if title else "supporting",
                "proceedings_pages": pages.get_text(strip=True) if pages else "",
            }
        )
    if not documents or not any(d["kind"] == "paper" for d in documents):
        raise RagError("No papers found in the CEUR catalog.")
    return documents


def list_documents(db, track=None, author=None, title=None, paper_ids=None):
    clauses, values = [], []
    for column, value in (
        ("track", track),
        ("author_search", normalized(author) if author else None),
        ("title", title),
    ):
        if value:
            # instr treats user input literally, including quotes, '%' and '_'.
            clauses.append(f"instr(lower({column}), lower(?)) > 0")
            values.append(value)
    if paper_ids:
        clauses.append("id IN (" + ",".join("?" for _ in paper_ids) + ")")
        values.extend(paper_ids)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = db.execute(
        "SELECT * FROM documents" + where + " ORDER BY rowid", values
    ).fetchall()
    return [dict(row) for row in rows]


def resolve_papers(db, references):
    ids = []
    for reference in references:
        ref = re.sub(r"\.pdf$", "", reference.strip(), flags=re.IGNORECASE)
        ref = re.sub(r"^paper\s+(\d+)$", r"paper\1", ref, flags=re.IGNORECASE)
        exact = db.execute(
            "SELECT id FROM documents WHERE lower(id)=lower(?) OR lower(title)=lower(?)",
            (ref, ref),
        ).fetchall()
        matches = [row["id"] for row in exact] or [
            d["id"] for d in list_documents(db, title=ref)
        ]
        if not matches:
            raise RagError(f"No catalog paper matches {reference!r}.")
        if len(matches) > 1:
            raise RagError(
                f"Ambiguous paper {reference!r}; use one of these IDs: {', '.join(matches[:12])}"
            )
        if matches[0] not in ids:
            ids.append(matches[0])
    return ids


def stats(db, track=None, author=None, title=None, paper_ids=None):
    docs = list_documents(
        db, track=track, author=author, title=title, paper_ids=paper_ids
    )
    papers = [d for d in docs if d["kind"] == "paper"]
    supporting = [d for d in docs if d["kind"] != "paper"]
    by_track = {}
    for d in papers:
        counts = by_track.setdefault(
            d["track"], {"listed": 0, "downloaded": 0, "indexed": 0}
        )
        counts["listed"] += 1
        counts["downloaded"] += d["available"]
        counts["indexed"] += d["status"] in ("indexed", "partial")
    return {
        "listed_papers": len(papers),
        "downloaded_papers": sum(d["available"] for d in papers),
        "indexed_papers": sum(d["status"] in ("indexed", "partial") for d in papers),
        "supporting_documents": len(supporting),
        "downloaded_supporting_documents": sum(d["available"] for d in supporting),
        "indexed_supporting_documents": sum(
            d["status"] in ("indexed", "partial") for d in supporting
        ),
        "missing_papers": [d["id"] for d in papers if not d["available"]],
        "extraction_issues": [
            {
                "id": d["id"],
                "status": d["status"],
                "error": d["error"],
                "warnings": json.loads(d["warnings"]),
            }
            for d in docs
            if d["error"] or json.loads(d["warnings"])
        ],
        "tracks": by_track,
    }
