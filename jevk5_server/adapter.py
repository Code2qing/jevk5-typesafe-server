"""Adapter wrapping jevk5.JevK5GGUF to provide TypeSafe AI format answers."""

from __future__ import annotations

import asyncio
import urllib.request
from typing import Any
from jevk5 import JevK5GGUF

from .confidence import calculate_confidence
from .schemas import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)


class JevK5TypeSafeAdapter:
    """Wraps JevK5GGUF to process requests and output TypeSafe System One responses."""

    def __init__(
        self,
        model: JevK5GGUF,
        default_model_name: str = "jevk5-4b-v0.3-Q8_0",
    ) -> None:
        self.model = model
        self.default_model_name = default_model_name

    def check_health(self) -> bool:
        """Check if backend llama-server is healthy."""
        try:
            req = urllib.request.Request(f"{self.model.url}/health")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        answers: dict[str, Answer] = {}
        total_input_tokens = 0

        for qid, q in request.questions.items():
            q_dict = q.model_dump()
            # JevK5GGUF.probabilities handles state and question dict via llama-server
            probs, tokens = await asyncio.to_thread(
                self.model.probabilities, request.state, q_dict
            )
            total_input_tokens += tokens

            if q.type == "noul":
                noul_prob = round(probs.get("true", 0.0), 4)
                answers[qid] = NoulAnswer(type="noul", noul=noul_prob)

            elif q.type == "choice":
                top_choice = max(probs, key=probs.get)
                conf = calculate_confidence(probs.values())
                answers[qid] = ChoiceAnswer(
                    type="choice",
                    choice=top_choice,
                    probabilities={k: round(v, 4) for k, v in probs.items()},
                    confidence=round(conf, 4),
                )

            elif q.type == "score":
                crit_list = q.criteria
                legend = {str(i): str(level) for i, level in enumerate(crit_list)}
                score_val = sum(int(k) * v for k, v in probs.items())
                conf = calculate_confidence(probs.values())
                answers[qid] = ScoreAnswer(
                    type="score",
                    score=round(score_val, 4),
                    legend=legend,
                    probabilities={k: round(v, 4) for k, v in probs.items()},
                    confidence=round(conf, 4),
                )

        return SystemOneResponse(
            model=request.model or self.default_model_name,
            answers=answers,
            usage=Usage(input_tokens=total_input_tokens, output_tokens=0),
        )
