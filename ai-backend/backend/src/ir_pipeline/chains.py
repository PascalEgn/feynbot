from typing import Any, Type

from backend.src.ir_pipeline.schema import LLMPaperResponse, LLMResponse, Terms
from backend.src.utils.langfuse import get_prompt
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableConfig


class RobustPydanticOutputParser(BaseOutputParser):
    pydantic_object: Type[Any]

    def parse(self, text: str) -> Any:
        stripped = text.strip()

        try:
            return self.pydantic_object.model_validate_json(stripped)
        except Exception:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return self.pydantic_object.model_validate_json(
                    stripped[start : end + 1]
                )
            except Exception:
                pass

        raise OutputParserException(f"Invalid json output: {text}")

    def get_format_instructions(self) -> str:
        return PydanticOutputParser(
            pydantic_object=self.pydantic_object
        ).get_format_instructions()


def _make_chain(llm: BaseLanguageModel, prompt_template, schema: Type[Any]):
    if isinstance(llm, BaseChatModel):
        return prompt_template | llm.with_structured_output(schema, strict=True)
    return prompt_template | llm | RobustPydanticOutputParser(pydantic_object=schema)


def create_query_expansion_chain(llm: BaseLanguageModel):
    prompt_template, langfuse_prompt = get_prompt("expand-query")
    config = RunnableConfig(
        run_name="expand-query", metadata={"langfuse_prompt": langfuse_prompt}
    )
    return _make_chain(llm, prompt_template, Terms).with_config(config)


def create_answer_generation_chain(
    llm: BaseLanguageModel, prompt_name: str = "generate-answer"
):
    prompt_template, langfuse_prompt = get_prompt(prompt_name)
    config = RunnableConfig(
        run_name=prompt_name, metadata={"langfuse_prompt": langfuse_prompt}
    )
    return _make_chain(llm, prompt_template, LLMResponse).with_config(config)


def create_rag_answer_generation_chain(llm: BaseLanguageModel):
    prompt_template, langfuse_prompt = get_prompt("rag-query")
    config = RunnableConfig(
        run_name="rag-query", metadata={"langfuse_prompt": langfuse_prompt}
    )
    return _make_chain(llm, prompt_template, LLMResponse).with_config(config)


def create_rag_paper_answer_generation_chain(llm: BaseLanguageModel):
    prompt_template, langfuse_prompt = get_prompt("rag-paper-query")
    config = RunnableConfig(
        run_name="rag-paper-query", metadata={"langfuse_prompt": langfuse_prompt}
    )
    return _make_chain(llm, prompt_template, LLMPaperResponse).with_config(config)
