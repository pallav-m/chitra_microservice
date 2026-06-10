from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioInput(BaseModel):
    """Audio supplied as inline base64 OR a remote URL — exactly one."""

    model_config = ConfigDict(populate_by_name=True)

    base64: str | None = Field(default=None, alias="base64")
    audio_uri: str | None = Field(default=None, alias="audioUri")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "AudioInput":
        provided = [self.base64 is not None, self.audio_uri is not None]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of 'base64' or 'audioUri'")
        return self


class SeparationRequest(AudioInput):
    response_format: Literal["json", "wav"] = Field(
        default="json", alias="responseFormat"
    )
    # Only used when response_format == "wav": which stem to stream.
    stem: Literal["vocals", "no_vocals"] = "vocals"


class SeparationResponse(BaseModel):
    vocals: str  # base64-encoded WAV
    no_vocals: str  # base64-encoded WAV
    sample_rate: int


class EmbeddingResponse(BaseModel):
    embedding: list[float]  # raw, un-normalized
    dim: int
