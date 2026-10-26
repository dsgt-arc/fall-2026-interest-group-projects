"""LangChain PDF ingestion, splitting and caching into persistent Chroma."""

import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from clef_rag.catalog import read_catalog
from clef_rag.config import EMBED_MODEL, RagError
from clef_rag.index import (
    COLLECTION,
    INDEX_SCHEMA_VERSION,
    cached_embeddings,
    chroma_client,
)
from clef_rag.models import EMBED_INPUT_BYTES

EXTRACTION_VERSION = "langchain-pypdf-pages-v1"
CHUNK_BYTES = 1400
OVERLAP_BYTES = 250


def content_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def chunks_for_page(doc, page, text):
    prefix = f"title: {doc['title']} | text: Track: {doc['track']}\n"
    capacity = min(CHUNK_BYTES, EMBED_INPUT_BYTES - len(prefix.encode("utf-8")))
    if capacity < 80:
        raise RagError(f"Catalog title/track is too long to embed: {doc['id']}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=capacity,
        chunk_overlap=min(OVERLAP_BYTES, capacity // 5),
        length_function=lambda value: len(value.encode("utf-8")),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    for chunk in splitter.create_documents([text]):
        start = chunk.metadata["start_index"]
        key = content_hash(f"{doc['id']}:{page}:{start}:{chunk.page_content}")
        yield Document(
            id=key,
            page_content=prefix + chunk.page_content,
            metadata={
                "chunk_id": key,
                "document_id": doc["id"],
                "page": page,
                "start_char": start,
                "end_char": start + len(chunk.page_content),
                "title": doc["title"],
                "track": doc["track"],
                "url": doc["url"],
                "filename": doc["filename"],
                "proceedings_pages": doc["proceedings_pages"],
            },
        )


def extract_pdf(path):
    pages, warnings, error = [], [], ""
    try:
        for document in PyPDFLoader(
            path, mode="page", extract_images=False
        ).lazy_load():
            number = document.metadata["page"] + 1
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", document.page_content)
            text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 80:
                warnings.append(f"PDF page {number}: little or no extractable text")
            pages.append({"page": number, "text": text})
    except Exception as exc:  # noqa: BLE001 - retain readable pages and record malformed PDFs
        error = str(exc)
        warnings.append(f"PDF loading stopped after {len(pages)} pages: {exc}")
    return {"pages": pages, "warnings": warnings, "error": error}


@contextmanager
def build_lock(index_dir):
    import fcntl

    with (index_dir / "build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RagError("Another ingestion is running for this index.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


SCHEMA = """
CREATE TABLE documents (
 id TEXT PRIMARY KEY, title TEXT, authors TEXT, author_search TEXT, track TEXT,
 url TEXT, filename TEXT, kind TEXT, proceedings_pages TEXT, available INTEGER,
 sha256 TEXT, status TEXT, error TEXT, page_count INTEGER, warnings TEXT
);
"""


def ingest(
    data_dir,
    index_dir,
    models,
    *,
    html_file=None,
    refresh=False,
    batch_size=16,
    progress=print,
):
    if not data_dir.is_dir():
        raise RagError(
            f"PDF directory does not exist: {data_dir}. Run download_papers.py first."
        )
    index_dir.mkdir(parents=True, exist_ok=True)
    with build_lock(index_dir):
        return _ingest(
            data_dir, index_dir, models, html_file, refresh, batch_size, progress
        )


def _ingest(data_dir, index_dir, models, html_file, refresh, batch_size, progress):
    started = time.perf_counter()
    digest = models.digest(EMBED_MODEL)
    documents, snapshot = read_catalog(index_dir, html_file, refresh)
    generation = f"build-{uuid.uuid4().hex}"
    build_dir = index_dir / generation
    build_dir.mkdir()
    extraction_cache = LocalFileStore(index_dir / "extraction-cache")
    embeddings = cached_embeddings(index_dir, models)
    namespace = f"embeddinggemma-v2-{digest}-"
    before = sum(
        1
        for _ in LocalFileStore(index_dir / "embedding-cache").yield_keys()
        if _.startswith(namespace)
    )
    client = chroma_client(build_dir / "chroma")
    store = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    db = sqlite3.connect(build_dir / "catalog.sqlite")
    try:
        db.executescript(SCHEMA)
        extracted = reused = chunks_total = 0
        for number, doc in enumerate(documents, 1):
            path = data_dir / doc["filename"]
            exists = path.is_file()
            file_hash, status = "", "missing"
            result = {"pages": [], "warnings": [], "error": ""}
            if exists:
                with path.open("rb") as stream:
                    file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                key = content_hash(EXTRACTION_VERSION + file_hash)
                cached = extraction_cache.mget([key])[0]
                if cached:
                    result = json.loads(cached)
                    reused += 1
                else:
                    result = extract_pdf(path)
                    if not result["error"]:
                        extraction_cache.mset([(key, json.dumps(result).encode())])
                    extracted += 1
                status = "error" if result["error"] else "no_text"
            chunks = [
                chunk
                for page in result["pages"]
                for chunk in chunks_for_page(doc, page["page"], page["text"])
            ]
            if chunks:
                status = "partial" if result["warnings"] else "indexed"
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc["id"],
                    doc["title"],
                    json.dumps(doc["authors"], ensure_ascii=False),
                    doc["author_search"],
                    doc["track"],
                    doc["url"],
                    doc["filename"],
                    doc["kind"],
                    doc["proceedings_pages"],
                    int(exists),
                    file_hash,
                    status,
                    result["error"],
                    len(result["pages"]),
                    json.dumps(result["warnings"]),
                ),
            )
            for offset in range(0, len(chunks), batch_size):
                store.add_documents(chunks[offset : offset + batch_size])
            chunks_total += len(chunks)
            db.commit()
            progress(
                f"[{number}/{len(documents)}] {doc['id']}: {status}, {len(chunks)} passages; {chunks_total} total"
            )
        if not chunks_total:
            raise RagError(
                "No passages were extracted; the previous index was preserved."
            )
        after = sum(
            1
            for _ in LocalFileStore(index_dir / "embedding-cache").yield_keys()
            if _.startswith(namespace)
        )
        collection = client.get_collection(COLLECTION, embedding_function=None)
        if collection.count() != chunks_total:
            raise RagError(
                "Chroma passage count differs from ingestion; the previous index was preserved."
            )
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "generation": generation,
            "created_at": datetime.now(UTC).isoformat(),
            "catalog_snapshot": str(snapshot.resolve()),
            "data_dir": str(data_dir.resolve()),
            "embedding_model": EMBED_MODEL,
            "embedding_digest": digest,
            "vector_store": "chromadb",
            "collection": COLLECTION,
            "chunks": chunks_total,
            "documents": len(documents),
            "extraction_version": EXTRACTION_VERSION,
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_bytes": CHUNK_BYTES,
            "overlap_bytes": OVERLAP_BYTES,
            "extracted_documents": extracted,
            "reused_extractions": reused,
            "new_embeddings": after - before,
            "reused_embeddings": chunks_total - (after - before),
            "ingestion_seconds": round(time.perf_counter() - started, 2),
        }
        (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        pointer = index_dir / "current.tmp"
        pointer.write_text(json.dumps({"generation": generation}) + "\n")
        pointer.replace(index_dir / "current.json")
        return manifest
    finally:
        db.close()
        client.close()
