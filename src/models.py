from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types as gtypes
import json
from chess import Board, Move

CHESS_SCHEMA = {
    "type": "object",
    "properties": {
        "chess_move_SAN": {"type": "string"}
    },
    "required": ["chess_move_SAN"],
    "additionalProperties": False
}

CHESS_SCHEMA_WITH_REASONING = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "chess_move_SAN": {"type": "string"}
    },
    "required": ["chess_move_SAN", "reasoning"],
    "additionalProperties": False
}

HIGH_THINKING: int = 5000
MEDIUM_THINKING: int = 2000
LOW_THINKING: int = 1024

class ProviderType(Enum):
    OPENAI = 'OpenAI'
    ANTHROPIC = 'Anthropic'
    GEMINI = 'Gemini'

@dataclass
class ModelConfig:
    provider: ProviderType
    api_name: str
    label: str
    instructions: str
    is_reasoning: bool = False
    is_COT: bool = False
    elo: float = 400
    max_tokens: int = 2000
    thinking_effort: str = 'medium'

    def __post_init__(self):
        if self.is_COT and self.is_reasoning:
            raise AssertionError('Models must be reasoning XOR CoT')

        if self.provider == ProviderType.OPENAI:
            if self.max_tokens < 2000:
                self.thinking_effort = 'low'
            elif self.max_tokens >=5000:
                self.thinking_effort = 'high'

    def __hash__(self) -> int:
        return hash((self.provider.value, self.api_name, self.is_reasoning, self.is_COT, self.instructions, self.max_tokens))


class ModelProvider(ABC):
    @abstractmethod
    def call(self, context: str) -> Tuple[str, str]:
        pass


class OpenAIProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        self.client = OpenAI()
        self.config = config
        assert (config.provider == ProviderType.OPENAI)

    def call(self, context: str) -> Tuple[str, str]:
        kwargs = {
            "model": self.config.api_name,
            "input": context,
            "instructions": self.config.instructions,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "chess_schema",
                    "schema": {},
                    "strict": True
                }
            }
        }

        if self.config.is_reasoning:
            kwargs["reasoning"] = {'effort': self.config.thinking_effort}
            kwargs['text']['format']['schema'] = CHESS_SCHEMA

        elif self.config.is_COT:
            kwargs['text']['format']['schema'] = CHESS_SCHEMA_WITH_REASONING

        else:
            kwargs['text']['format']['schema'] = CHESS_SCHEMA

        response = self.client.responses.create(**kwargs)
        parsed_response = json.loads(response.output_text)
        move = parsed_response['chess_move_SAN']
        reasoning = parsed_response['reasoning'] if self.config.is_COT else response.reasoning.summary if self.config.is_reasoning else 'No reasoning'

        return move, reasoning


class AnthropicProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        self.client = Anthropic()
        self.config = config

    def call(self, context: str) -> Tuple[str, str]:
        kwargs = {
            'model': self.config.api_name,
            'system': self.config.instructions,
            'messages': [
                {'role': 'user', 'content': ''},
            ],
            'max_tokens': self.config.max_tokens + 6
        }

        user_message = f'{context}\n\nFormat your response as valid JSON matching this schema. Respond only with JSON: '

        if self.config.is_reasoning:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': self.config.max_tokens}
            user_message += json.dumps(CHESS_SCHEMA)

        elif self.config.is_COT:
            user_message += json.dumps(CHESS_SCHEMA_WITH_REASONING)

        else:
            user_message += json.dumps(CHESS_SCHEMA)

        kwargs["messages"][0]['content'] = user_message

        response = self.client.messages.create(**kwargs)
        text_content = ''
        for block in response.content:
            if block.type == 'text':
                text_content = block.text
                break
        parsed_response = json.loads(text_content)

        move = parsed_response['chess_move_SAN']
        reasoning = (
                parsed_response['reasoning'] if self.config.is_COT
                else "".join(block.thinking for block in response.content if block.type == "thinking") if self.config.is_reasoning
                else 'None'
            )
        return move, reasoning

class GeminiProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        self.client = genai.Client()
        self.config = config

    def call(self, context: str) -> Tuple[str, str]:
        kwargs = {
            'model': self.config.api_name,
            'contents': context,
            'config': gtypes.GenerateContentConfig(
                system_instruction=self.config.instructions,
                response_mime_type="application/json",
                max_output_tokens=self.config.max_tokens
            )
        }

        if self.config.is_reasoning:
            kwargs['config'].thinking_config = gtypes.ThinkingConfig(
                thinking_budget=self.config.max_tokens,
                include_thoughts=True
            )
            kwargs['config'].response_schema = CHESS_SCHEMA.copy()

        elif self.config.is_COT:
            kwargs['config'].response_schema = CHESS_SCHEMA_WITH_REASONING.copy()

        else:
            kwargs['config'].response_schema = CHESS_SCHEMA.copy()

        kwargs['config'].response_schema.pop('additionalProperties')
        response = self.client.models.generate_content(**kwargs)

        text_content = ''
        thought_content = ''


        if response.candidates[0].finish_reason != 'STOP': #type: ignore
            raise ValueError('Gemini refusal')

        for part in response.candidates[0].content.parts: #type: ignore
            if not part.text:
                continue
            if hasattr(part, 'thought') and part.thought:
                thought_content += part.text
            else:
                text_content += part.text

        parsed_response = json.loads(text_content)
        move = parsed_response['chess_move_SAN']
        reasoning = (
            parsed_response['reasoning'] if self.config.is_COT
            else thought_content if self.config.is_reasoning and thought_content
            else 'No reasoning'
        )
        return move, reasoning

def build_provider(cfg: ModelConfig):
    if cfg.provider == ProviderType.ANTHROPIC:
        return AnthropicProvider(cfg)
    elif cfg.provider == ProviderType.OPENAI:
        return OpenAIProvider(cfg)
    elif cfg.provider == ProviderType.GEMINI:
        return GeminiProvider(cfg)
    else:
        raise ValueError('Unsupported ProviderTypek')

if __name__ == '__main__':
    gemini_thinking_cfg = ModelConfig(
        provider= ProviderType.GEMINI,
        api_name= 'gemini-2.5-flash',
        label= 'gemini-2.5-flash thinking',
        is_reasoning= True,
        is_COT = False,
        instructions='Analyse the chess position and provide the best move.'
    )
    gemini_thinking = build_provider(gemini_thinking_cfg)
    print(gemini_thinking.call('r1bqk2r/ppppbppp/2n2n2/1B2p3/3P4/5N2/PPP2PPP/RNBQR1K1 w kq - 1 7'))
