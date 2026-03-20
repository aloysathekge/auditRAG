from datetime import datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_name: str = Field(..., description="Document identifier (e.g. 3M_2018_10K)")
    doc_link: str = Field(..., description="URL to the PDF")
    knowledge_base: str = Field("default", description="Knowledge base to ingest into")


class UploadResponse(BaseModel):
    status: str
    doc_id: str
    doc_name: str
    knowledge_base: str
    chunks_created: int
    chunks_upserted: int


class DocumentInfo(BaseModel):
    doc_name: str
    knowledge_base: str
    chunks_count: int
    created_at: datetime | None = None


class DocumentListResponse(BaseModel):
    knowledge_base: str
    documents: list[DocumentInfo]
    total: int
