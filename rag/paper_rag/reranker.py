import json
import os
from typing import Iterable, List, Mapping, Optional


class QwenReranker:
    """Thin rerank module around the fine-tuned Qwen selector model."""

    def __init__(self, model_path: Optional[str], prompt_path: Optional[str] = None):
        self.model_path = model_path
        self.agent = None
        if model_path:
            from coding.models import Agent

            self.agent = Agent(model_path)
        self.prompts = self._load_prompts(prompt_path)

    @property
    def available(self) -> bool:
        return self.agent is not None

    def _load_prompts(self, prompt_path: Optional[str]) -> dict:
        if prompt_path is None:
            prompt_path = os.path.join(os.getcwd(), "agent_prompt.json")

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as file:
                return json.load(file)

        return {
            "get_selected": (
                "Given the user query and paper, answer True if the paper is relevant, "
                "otherwise answer False.\n\n"
                "User query: {user_query}\n"
                "Title: {title}\n"
                "Abstract: {abstract}"
            )
        }

    def build_prompts(self, results: Iterable[Mapping], user_query: str) -> List[str]:
        template = self.prompts["get_selected"]
        prompts = []
        for result in results:
            paper = result["paper"]
            prompts.append(
                template.format(
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    user_query=user_query,
                )
            )
        return prompts

    def score(self, results: List[Mapping], user_query: str) -> List[float]:
        if not self.available or not results:
            return [0.0] * len(results)
        return self.agent.infer_score(self.build_prompts(results, user_query))
