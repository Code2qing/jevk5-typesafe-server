"""Pydantic schemas matching the TypeSafe AI System One specification (https://docs.typesafe.ai/api)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator


# -----------------------------------------------------------------------------
# Questions
# -----------------------------------------------------------------------------


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: Any = Field(
        ...,
        description="The yes/no question to evaluate. Can be string, object, or array.",
    )
    criteria: dict[str, Any] | None = Field(
        default=None,
        description="Optional descriptions for 'true' and 'false'.",
    )


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: Any = Field(
        ...,
        description="What the model should decide.",
    )
    criteria: dict[str, Any] | list[str] = Field(
        ...,
        description="A map of options to descriptions (2-255 options) or list of option keys.",
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: dict[str, Any] | list[str]) -> dict[str, Any] | list[str]:
        count = len(v)
        if count < 2:
            raise ValueError("choice criteria must name at least 2 options")
        if count > 255:
            raise ValueError("choice criteria cannot exceed 255 options")
        return v


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: Any = Field(
        ...,
        description="What the model should rate.",
    )
    criteria: list[Any] = Field(
        ...,
        description="An ordered array of 2 to 10 level descriptions.",
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: list[Any]) -> list[Any]:
        count = len(v)
        if count < 2:
            raise ValueError("score criteria must list at least 2 levels")
        if count > 10:
            raise ValueError("score criteria cannot exceed 10 levels")
        return v


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Request
# -----------------------------------------------------------------------------


class SystemOneRequest(BaseModel):
    state: Any = Field(
        ...,
        description="The content to evaluate. String, object, or array.",
    )
    model: str = Field(
        default="jev-latest",
        description="The model or alias that handles the request.",
    )
    questions: dict[str, Question] = Field(
        ...,
        description="A map of typed Question objects.",
    )


# -----------------------------------------------------------------------------
# Answers & Responses
# -----------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(..., description="The yes/no probability between 0 and 1.")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="The highest-probability option.")
    probabilities: dict[str, float] = Field(
        ...,
        description="Every option mapped to its probability.",
    )
    confidence: float = Field(
        ...,
        description="How certain the model is, derived from probabilities.",
    )


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float = Field(
        ...,
        description="The probability-weighted answer across levels.",
    )
    legend: dict[str, str] = Field(
        ...,
        description="Each level number mapped back to its description.",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Each level mapped to its probability.",
    )
    confidence: float = Field(
        ...,
        description="How certain the model is, derived from probabilities.",
    )


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int = Field(default=0, description="Tokens evaluated for prompt.")
    output_tokens: int = Field(default=0, description="Generated tokens (always minimal/0).")


class SystemOneResponse(BaseModel):
    model: str = Field(..., description="The model that performed the evaluation.")
    answers: dict[str, Answer] = Field(..., description="Answers keyed by question ID.")
    usage: Usage = Field(..., description="Token usage for the request.")


# -----------------------------------------------------------------------------
# Models endpoint
# -----------------------------------------------------------------------------


class ModelCard(BaseModel):
    name: str
    description: str
    release_date: str


class ModelsResponse(BaseModel):
    models: list[ModelCard]
