from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to return")
    generate_answer: bool = Field(
        True,
        description="If true, return LLM answer + sources (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)",
    )
    knowledge_base: str | None = Field(
        None,
        description="Knowledge base to search within. None searches all.",
    )
    doc_name: str | None = Field(
        None,
        description="Filter retrieval to a specific document. None searches all docs.",
    )
