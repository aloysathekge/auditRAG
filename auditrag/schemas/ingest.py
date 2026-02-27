from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_name: str = Field(..., description="Document identifier (e.g. 3M_2018_10K)")
    doc_link: str = Field(..., description="URL to the PDF")
