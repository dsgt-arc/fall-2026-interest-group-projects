import ast
import json
import math
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from clef_rag.answer import generate_answer, route_question
from clef_rag.catalog import list_documents, parse_catalog, resolve_papers, stats
from clef_rag.config import EMBED_MODEL, RagError
from clef_rag.index import Index
from clef_rag.ingest import chunks_for_page, extract_pdf, ingest
from clef_rag.models import EMBED_INPUT_BYTES, LocalModels, RetrievalEmbeddings
from clef_rag.retrieve import Retriever, lexical_tokens, source_record

HTML = """<div class="CEURTOC"><ul><li id="preface"><a href="preface.pdf">Preface</a></li></ul>
<h3><span class="CEURSESSION">BioASQ: Biomedical QA</span></h3><ul>
<li id="paper1"><a href="paper1.pdf"><span class="CEURTITLE">Alpha biomedical retrieval</span></a>
<span class="CEURAUTHOR">José Example</span><span class="CEURPAGES">1-2</span></li>
<li id="paper2"><a href="paper2.pdf"><span class="CEURTITLE">Beta biomedical methods</span></a>
<span class="CEURAUTHOR">Jane Example</span><span class="CEURPAGES">3-4</span></li></ul>
<h3><span class="CEURSESSION">PAN: Text Forensics</span></h3><ul>
<li id="paper3"><a href="paper3.pdf"><span class="CEURTITLE">Gamma missing paper</span></a></li></ul></div>"""


class FakeModels(Embeddings):
    chat_model = "gemma3:4b"

    def __init__(self):
        self.calls = 0
        self.identity = "test-digest"
        self.answer = None

    def digest(self, _):
        return self.identity

    def embeddings(self):
        return self

    def embed_documents(self, texts):
        self.calls += len(texts)
        vectors = [
            [
                text.lower().count("retrieval") + 1,
                text.lower().count("classification") + 1,
                1,
            ]
            for text in texts
        ]
        return [[value / math.hypot(*vector) for value in vector] for vector in vectors]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def chat(self, *_args, **_kwargs):
        if self.answer is None:
            raise AssertionError("This operation should not need an LLM")
        if "reason" in _kwargs.get("schema", {}).get("required", []):
            return {
                "supported": self.answer["supported"],
                "reason": "Test evidence assessment",
                "unsupported_sources": [],
            }
        return self.answer


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("paper1", "paper2", "preface"):
        (data / f"{name}.pdf").write_bytes(b"%PDF-1.7\n" + name.encode())
    html = tmp_path / "catalog.html"
    html.write_text(HTML)

    def extraction(path):
        text = (
            "Alpha uses sparse retrieval and reranking. "
            if path.stem == "paper1"
            else "Beta uses classification of biomedical entities. "
        )
        return {
            "pages": [
                {"page": 1, "text": text * 90},
                {"page": 2, "text": "Conclusion. " * 60},
            ],
            "warnings": [],
            "error": "",
        }

    monkeypatch.setattr("clef_rag.ingest.extract_pdf", extraction)
    models = FakeModels()
    directory = tmp_path / "index"
    manifest = ingest(data, directory, models, html_file=html, progress=lambda _: None)
    return data, directory, html, models, manifest


def test_catalog_tracks_authors_and_invalid_page():
    docs = parse_catalog(HTML)
    assert len(docs) == 4
    assert docs[1]["authors"] == ["José Example"]
    assert docs[1]["track"] == "BioASQ: Biomedical QA"
    assert docs[3]["track"] == "PAN: Text Forensics"
    assert docs[0]["kind"] == "supporting"
    with pytest.raises(RagError, match="table of contents"):
        parse_catalog("<html>Error page</html>")


def test_langchain_splitter_preserves_text_and_page_provenance():
    text = " ".join(f"word{i}." for i in range(905))
    doc = parse_catalog(HTML)[1]
    chunks = list(chunks_for_page(doc, 7, text))
    covered = set()
    for chunk in chunks:
        c = source_record(chunk)
        assert c["page"] == 7
        assert c["text"] == text[c["start_char"] : c["end_char"]]
        assert len(chunk.page_content.encode("utf-8")) <= EMBED_INPUT_BYTES
        covered.update(range(c["start_char"], c["end_char"]))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    assert len({c.id for c in chunks}) == len(chunks)


def test_real_pdf_extraction_keeps_pdf_page_numbers(tmp_path):
    writer = PdfWriter()
    for text in ("Biomedical retrieval evidence.", "Second page results."):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "paper.pdf"
    writer.write(path)
    extracted = extract_pdf(path)
    assert not extracted["error"]
    assert extracted["pages"] == [
        {"page": 1, "text": "Biomedical retrieval evidence."},
        {"page": 2, "text": "Second page results."},
    ]


def test_exact_counts_filters_and_resume(corpus):
    data, directory, html, models, first = corpus
    with Index(directory) as index:
        coverage = stats(index.db)
        assert coverage["listed_papers"] == 3
        assert coverage["downloaded_papers"] == coverage["indexed_papers"] == 2
        assert coverage["missing_papers"] == ["paper3"]
        assert coverage["supporting_documents"] == 1
        assert len(list_documents(index.db, author="Jose")) == 1
        assert list_documents(index.db, author="%' OR 1=1 --") == []
        assert resolve_papers(index.db, ["paper 1", "Alpha biomedical"]) == ["paper1"]
        with pytest.raises(RagError, match="Ambiguous"):
            resolve_papers(index.db, ["biomedical"])
    calls = models.calls
    again = ingest(data, directory, models, html_file=html, progress=lambda _: None)
    assert models.calls == calls
    assert again["new_embeddings"] == 0
    assert again["reused_extractions"] == 3
    assert again["chunks"] == first["chunks"]


def test_failed_build_preserves_previous_generation_and_cache(corpus, monkeypatch):
    data, directory, html, models, first = corpus
    (data / "paper1.pdf").write_bytes(b"%PDF-changed")
    monkeypatch.setattr(
        "clef_rag.ingest.extract_pdf",
        lambda _: {
            "pages": [{"page": 1, "text": "Changed text " * 300}],
            "warnings": [],
            "error": "",
        },
    )
    monkeypatch.setattr(
        models,
        "embed_documents",
        lambda _: (_ for _ in ()).throw(RagError("Simulated interruption")),
    )
    with pytest.raises(RagError, match="interruption"):
        ingest(data, directory, models, html_file=html, progress=lambda _: None)
    with Index(directory) as index:
        assert index.manifest["generation"] == first["generation"]
        assert index.collection.count() == first["chunks"]


def test_changed_and_removed_papers_replace_old_search_content(corpus, monkeypatch):
    data, directory, html, models, _ = corpus
    (data / "paper2.pdf").unlink()
    (data / "paper1.pdf").write_bytes(b"%PDF-1.7 revised content")
    monkeypatch.setattr(
        "clef_rag.ingest.extract_pdf",
        lambda _: {
            "pages": [{"page": 1, "text": "New revised retrieval method."}],
            "warnings": [],
            "error": "",
        },
    )
    manifest = ingest(data, directory, models, html_file=html, progress=lambda _: None)
    assert manifest["new_embeddings"] == 1
    assert manifest["reused_extractions"] == 1
    with Index(directory) as index:
        assert stats(index.db)["indexed_papers"] == 1
        assert index.documents(["paper2"]) == []
        texts = [source_record(doc)["text"] for doc in index.documents(["paper1"])]
        assert texts == ["New revised retrieval method."]


def test_hybrid_search_filters_comparison_and_model_mismatch(corpus):
    _, directory, _, models, _ = corpus
    with Index(directory) as index:
        retrieval = Retriever(index, models)
        hits = retrieval.search("sparse retrieval", paper_ids=["paper1"])
        assert hits and all(h["document_id"] == "paper1" for h in hits)
        hits = retrieval.search(
            "compare retrieval and classification", paper_ids=["paper1", "paper2"]
        )
        assert {h["document_id"] for h in hits} == {"paper1", "paper2"}
        with pytest.raises(RagError, match="unavailable"):
            retrieval.search("method", paper_ids=["paper3"])
        models.identity = "changed"
        with pytest.raises(RagError, match="model changed"):
            retrieval.search("retrieval")


def test_counts_do_not_use_top_k_or_llm(corpus):
    _, directory, _, models, _ = corpus
    with Index(directory) as index:
        assert (
            route_question("How many papers are in BioASQ?", index.db, models)[
                "operation"
            ]
            == "stats"
        )
        assert (
            route_question("How many papers are downloaded?", index.db, models)[
                "operation"
            ]
            == "stats"
        )
        assert (
            route_question("How many papers use RAG?", index.db, models)["operation"]
            == "unsupported_count"
        )
        assert (
            route_question("Who wrote paper1?", index.db, models)["operation"]
            == "papers"
        )


def test_citation_validation_and_abstention(corpus):
    _, directory, _, models, _ = corpus
    with Index(directory) as index:
        sources = Retriever(index, models).search("retrieval", paper_ids=["paper1"])
    models.answer = {
        "supported": True,
        "claims": [{"text": "Uses sparse retrieval.", "sources": ["S999"]}],
    }
    with pytest.raises(RagError, match="invalid citations"):
        generate_answer("What method?", sources, models)
    models.answer = {
        "supported": True,
        "claims": [{"text": "Uses sparse retrieval.", "sources": ["S1"]}],
    }
    result = generate_answer("What method?", sources, models)
    assert len(result["sources"]) == 1
    models.answer = {
        "supported": True,
        "claims": [{"text": "Uses sparse retrieval.", "sources": ["S1", "S2"]}],
    }
    assert "[S1][S2]" in generate_answer("What method?", sources, models)["answer"]
    models.answer = {"supported": False, "claims": []}
    result = generate_answer("Who won an unrelated event?", sources, models)
    assert result["supported"] is False and "guess" not in result["answer"]


def test_multibyte_chunks_fit_embedding_context_and_oversize_queries_fail():
    doc = parse_catalog(HTML)[1]
    chunks = list(chunks_for_page(doc, 3, "αβγδεζηθ " * 600))
    assert len(chunks) > 1
    assert all(len(c.page_content.encode("utf-8")) <= EMBED_INPUT_BYTES for c in chunks)
    with pytest.raises(RagError, match="shorten"):
        RetrievalEmbeddings(FakeModels()).embed_query("α" * EMBED_INPUT_BYTES)


def test_unsupported_attribution_is_withheld_after_one_repair(corpus, monkeypatch):
    _, directory, _, models, _ = corpus
    with Index(directory) as index:
        sources = Retriever(index, models).search("retrieval", paper_ids=["paper1"])
    models.answer = {
        "supported": True,
        "claims": [{"text": "Unsupported attribution.", "sources": ["S1"]}],
    }
    original = models.chat
    reviews = []

    def chat(system, question, **kwargs):
        if system.startswith("Check each claim"):
            reviews.append(question)
            return {
                "reason": "This passage describes prior work, not the team's method.",
                "supported": False,
            }
        return original(system, question, **kwargs)

    monkeypatch.setattr(models, "chat", chat)
    result = generate_answer("What method does this paper use?", sources, models)
    assert result["supported"] is False
    assert "Unsupported attribution" not in result["answer"]
    assert len(reviews) == 2


def test_citation_repair_excludes_rejected_evidence(corpus, monkeypatch):
    _, directory, _, models, _ = corpus
    with Index(directory) as index:
        sources = Retriever(index, models).search("retrieval", paper_ids=["paper1"])
    attempts = []

    def chat(system, question, **_kwargs):
        if system.startswith("Check each claim"):
            return {
                "supported": len(attempts) == 2,
                "reason": "S1 does not support the attributed method.",
                "unsupported_sources": ["S1"] if len(attempts) == 1 else [],
            }
        if system.startswith("Decide whether"):
            return {"supported": True, "reason": "The method is present."}
        evidence = json.loads(question.split("\nEvidence:\n", 1)[1])
        available = {source["source_id"] for source in evidence}
        attempts.append(available)
        return {
            "supported": True,
            "claims": [{"text": "The supported method.", "sources": [min(available)]}],
        }

    monkeypatch.setattr(models, "chat", chat)
    result = generate_answer("What method does this paper use?", sources, models)
    assert len(attempts) == 2
    assert "S1" in attempts[0] and "S1" not in attempts[1]
    assert result["supported"] is True
    assert "[S1]" not in result["answer"]
    assert all(source["source_id"] != "S1" for source in result["sources"])


def test_model_policy_and_literal_keyword_queries():
    with pytest.raises(RagError, match="not allowed"):
        LocalModels("qwen3.6:35b")
    with pytest.raises(RagError, match="not allowed"):
        LocalModels().digest("bge-m3")
    assert lexical_tokens('" OR 1=1; DROP TABLE documents; --') == [
        "or",
        "1",
        "1",
        "drop",
        "table",
        "documents",
    ]
    assert EMBED_MODEL == "embeddinggemma:300m"


def test_legacy_index_requires_explicit_rebuild(tmp_path):
    build = tmp_path / "build-old"
    build.mkdir()
    (tmp_path / "current.json").write_text(json.dumps({"generation": "build-old"}))
    (build / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(RagError, match="legacy index"):
        Index(tmp_path)


def test_project_uses_only_absolute_imports():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("tests").rglob("*.py"),
        root / "download_papers.py",
    ]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            assert not (isinstance(node, ast.ImportFrom) and node.level), str(path)
