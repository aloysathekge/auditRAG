from pathlib import Path
from unittest.mock import MagicMock, patch

from auditrag.ingestion import (
    chunk_pages,
    embed_chunks,
    extract_text,
    download_pdf,
    upsert_chunks_to_qdrant,
)


class TestChunkPages:
    def test_basic_chunking(self):
        pages = [
            {"page": 1, "text": " ".join(f"word{i}" for i in range(100))},
        ]
        chunks = chunk_pages(pages, "test_doc", chunk_size=30, chunk_overlap=5)

        assert len(chunks) > 1
        assert chunks[0]["doc_name"] == "test_doc"
        assert chunks[0]["chunk_id"] == "default_test_doc_chunk_0"
        assert chunks[0]["page"] == 1
        assert chunks[0]["knowledge_base"] == "default"

    def test_chunk_with_custom_knowledge_base(self):
        pages = [{"page": 1, "text": "hello world this is a test"}]
        chunks = chunk_pages(pages, "doc", chunk_size=100, chunk_overlap=0, knowledge_base="my_kb")

        assert chunks[0]["knowledge_base"] == "my_kb"
        assert chunks[0]["chunk_id"].startswith("my_kb_")

    def test_chunk_text_not_empty(self):
        pages = [{"page": 1, "text": "hello world this is a test"}]
        chunks = chunk_pages(pages, "doc", chunk_size=3, chunk_overlap=1)

        for chunk in chunks:
            assert len(chunk["text"].strip()) > 0

    def test_overlap_produces_more_chunks(self):
        pages = [
            {"page": 1, "text": " ".join(f"w{i}" for i in range(100))},
        ]
        no_overlap = chunk_pages(pages, "doc", chunk_size=20, chunk_overlap=0)
        with_overlap = chunk_pages(pages, "doc", chunk_size=20, chunk_overlap=5)

        assert len(with_overlap) > len(no_overlap)

    def test_single_page_metadata(self):
        pages = [{"page": 5, "text": "some financial data here"}]
        chunks = chunk_pages(pages, "report", chunk_size=100, chunk_overlap=0)

        assert len(chunks) == 1
        assert chunks[0]["page"] == 5

    def test_multi_page_tracks_page_numbers(self):
        pages = [
            {"page": 1, "text": " ".join(f"a{i}" for i in range(50))},
            {"page": 2, "text": " ".join(f"b{i}" for i in range(50))},
        ]
        chunks = chunk_pages(pages, "doc", chunk_size=30, chunk_overlap=0)

        assert any(c["page"] == 1 for c in chunks)
        assert any(c["page"] == 2 for c in chunks)

    def test_empty_pages_produce_no_chunks(self):
        pages = []
        chunks = chunk_pages(pages, "empty_doc", chunk_size=50, chunk_overlap=10)
        assert chunks == []


class TestExtractText:
    def test_extracts_pages_from_cached_pdf(self):
        pdf_path = Path("data/financebench/pdfs/3M_2018_10K.pdf")
        if not pdf_path.exists():
            return  # skip if PDF not downloaded yet

        pages = extract_text(pdf_path)

        assert len(pages) > 0
        assert "page" in pages[0]
        assert "text" in pages[0]
        assert isinstance(pages[0]["text"], str)
        assert len(pages[0]["text"]) > 0


class TestEmbedChunks:
    def test_local_embedding_adds_vectors(self):
        chunks = [
            {"chunk_id": "doc_chunk_0", "doc_name": "doc", "page": 1, "text": "revenue was 100 million", "knowledge_base": "default"},
            {"chunk_id": "doc_chunk_1", "doc_name": "doc", "page": 1, "text": "net income increased", "knowledge_base": "default"},
        ]
        fake_vectors = [[0.1] * 384, [0.2] * 384]
        fake_model = MagicMock()
        fake_model.encode.return_value = MagicMock(tolist=lambda: fake_vectors)

        with (
            patch("auditrag.ingestion.embedder.get_settings") as mock_settings,
            patch("auditrag.ingestion.embedder._get_local_model", return_value=fake_model),
        ):
            mock_settings.return_value.embedding_provider = "local"
            result = embed_chunks(chunks)

        assert len(result) == 2
        assert "embedding" in result[0]
        assert isinstance(result[0]["embedding"], list)
        assert len(result[0]["embedding"]) == 384
        assert all(isinstance(v, float) for v in result[0]["embedding"])

    def test_openai_embedding_calls_api(self):
        chunks = [
            {"chunk_id": "doc_chunk_0", "doc_name": "doc", "page": 1, "text": "test text", "knowledge_base": "default"},
        ]
        fake_embedding = [0.1] * 1536
        fake_item = MagicMock()
        fake_item.embedding = fake_embedding
        fake_response = MagicMock()
        fake_response.data = [fake_item]

        with (
            patch("auditrag.ingestion.embedder.get_settings") as mock_settings,
            patch("openai.OpenAI") as mock_openai_cls,
        ):
            mock_settings.return_value.embedding_provider = "openai"
            mock_settings.return_value.openai_api_key = "fake-key"
            mock_openai_cls.return_value.embeddings.create.return_value = fake_response

            result = embed_chunks(chunks)

        assert "embedding" in result[0]
        assert result[0]["embedding"] == fake_embedding


class TestUpsertChunksToQdrant:
    def test_raises_when_chunks_not_embedded(self):
        chunks = [
            {"chunk_id": "a", "doc_name": "d", "page": 1, "text": "x"},
        ]
        with patch("auditrag.ingestion.embedder.get_settings"):
            import pytest
            with pytest.raises(ValueError, match="embedded first"):
                upsert_chunks_to_qdrant(chunks)

    def test_upserts_points_with_mocked_client(self):
        chunks = [
            {
                "chunk_id": "doc_chunk_0",
                "doc_name": "doc",
                "page": 1,
                "text": "revenue was high",
                "embedding": [0.1] * 384,
                "knowledge_base": "default",
            },
            {
                "chunk_id": "doc_chunk_1",
                "doc_name": "doc",
                "page": 2,
                "text": "net income",
                "embedding": [0.2] * 384,
                "knowledge_base": "default",
            },
        ]
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)

        with patch("auditrag.ingestion.embedder.get_settings") as mock_settings:
            mock_settings.return_value.qdrant_url = "http://localhost:6333"
            mock_settings.return_value.qdrant_api_key = ""
            with patch("auditrag.ingestion.embedder.QdrantClient", return_value=mock_client):
                n = upsert_chunks_to_qdrant(chunks)

        assert n == 2
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()
        call_kw = mock_client.upsert.call_args[1]
        assert call_kw["collection_name"] == "auditrag_chunks"
        assert len(call_kw["points"]) == 2
        p = call_kw["points"][0]
        assert p.payload["chunk_id"] == "doc_chunk_0"
        assert p.payload["doc_name"] == "doc"
        assert p.payload["page"] == 1
        assert p.payload["text"] == "revenue was high"
        assert p.payload["knowledge_base"] == "default"
        assert len(p.vector) == 384

    def test_skips_upsert_when_doc_already_ingested(self):
        chunks = [
            {"chunk_id": "doc_chunk_0", "doc_name": "doc", "page": 1, "text": "x", "embedding": [0.1] * 384, "knowledge_base": "default"},
        ]
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([MagicMock()], None)

        with patch("auditrag.ingestion.embedder.get_settings") as mock_settings:
            mock_settings.return_value.qdrant_url = "http://localhost:6333"
            mock_settings.return_value.qdrant_api_key = ""
            with patch("auditrag.ingestion.embedder.QdrantClient", return_value=mock_client):
                n = upsert_chunks_to_qdrant(chunks, skip_if_doc_exists=True)

        assert n == 0
        mock_client.upsert.assert_not_called()


class TestDownloadPdf:
    def test_skips_download_if_cached(self, tmp_path):
        pdf_file = tmp_path / "test_doc.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        with patch("auditrag.ingestion.loader.DATA_DIR", tmp_path):
            result = download_pdf("http://example.com/fake.pdf", "test_doc")

        assert result == pdf_file

    def test_downloads_when_not_cached(self, tmp_path):
        fake_response = MagicMock()
        fake_response.content = b"%PDF-fake-content"
        fake_response.raise_for_status = MagicMock()

        with (
            patch("auditrag.ingestion.loader.DATA_DIR", tmp_path),
            patch("httpx.get", return_value=fake_response) as mock_get,
        ):
            result = download_pdf("http://example.com/doc.pdf", "new_doc")

        mock_get.assert_called_once()
        assert result.exists()
        assert result.read_bytes() == b"%PDF-fake-content"
