"""Persistent Chroma vector stores and immutable catalog generations."""

import json
import sqlite3

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document

from clef_rag.config import EMBED_MODEL, RagError

COLLECTION = "clef_papers"
INDEX_SCHEMA_VERSION = 2


def chroma_client(directory):
    return chromadb.PersistentClient(
        path=str(directory.resolve()), settings=Settings(anonymized_telemetry=False)
    )


def cached_embeddings(directory, models):
    digest = models.digest(EMBED_MODEL)
    return CacheBackedEmbeddings.from_bytes_store(
        models.embeddings(),
        LocalFileStore(directory / "embedding-cache"),
        namespace=f"embeddinggemma-v2-{digest}-",
        key_encoder="sha256",
        query_embedding_cache=LocalFileStore(directory / "query-cache"),
    )


def as_documents(result):
    return [
        Document(id=key, page_content=text, metadata=metadata)
        for key, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    ]


class Index:
    def __init__(self, directory):
        self.db = self.client = None
        self.directory = directory
        try:
            generation = json.loads((directory / "current.json").read_text())[
                "generation"
            ]
            if (
                not generation.startswith("build-")
                or "/" in generation
                or "\\" in generation
            ):
                raise ValueError("invalid generation")
            root = directory / generation
            self.manifest = json.loads((root / "manifest.json").read_text())
            if self.manifest["schema_version"] != INDEX_SCHEMA_VERSION:
                raise ValueError(
                    "legacy index format; rebuild for LangChain and ChromaDB"
                )
            if self.manifest["embedding_model"] != EMBED_MODEL:
                raise ValueError("unsupported embedding model")
            self.db = sqlite3.connect(
                (root / "catalog.sqlite").resolve().as_uri() + "?mode=ro", uri=True
            )
            self.db.row_factory = sqlite3.Row
            if not (root / "chroma" / "chroma.sqlite3").is_file():
                raise ValueError("Chroma database is missing")
            self.client = chroma_client(root / "chroma")
            self.collection = self.client.get_collection(
                COLLECTION, embedding_function=None
            )
            if self.collection.count() != self.manifest["chunks"]:
                raise ValueError("Chroma document count does not match the manifest")
        except (OSError, ValueError, KeyError, sqlite3.Error, ChromaError) as exc:
            self.close()
            raise RagError(
                f"Cannot open index at {directory}: {exc}. Run: clef-rag ingest"
            ) from exc

    def vectorstore(self, models):
        if models.digest(EMBED_MODEL) != self.manifest["embedding_digest"]:
            raise RagError(
                "The embedding model changed since ingestion. Run clef-rag ingest to rebuild the index."
            )
        return Chroma(
            client=self.client,
            collection_name=COLLECTION,
            embedding_function=cached_embeddings(self.directory, models),
            create_collection_if_not_exists=False,
        )

    def documents(self, paper_ids=None):
        where = {"document_id": {"$in": paper_ids}} if paper_ids else None
        return as_documents(
            self.collection.get(where=where, include=["documents", "metadatas"])
        )

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
