from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source object representing a URL."""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """A response object containing the main response and sources."""

    answer: str = Field(description="The main response from the agent")
    sources: List[Source] = Field(
        default_factory=list, description="A list of source objects"
    )
