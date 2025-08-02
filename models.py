from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
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

class ProviderType(Enum):
    OPENAI = 'OpenAI'
    ANTHROPIC = 'Anthropic'

@dataclass
class ModelConfig:
    provider: ProviderType
    name: str
    is_reasoning: bool
    is_COT: bool

    def __post_init__(self):
        if self.is_COT and self.is_reasoning:
            raise AssertionError('Models must be reasoning XOR CoT')

class ModelProvider(ABC):
    @abstractmethod
    def call(self, context: str, instructions: Optional[str]) -> Tuple[str, str]:
        pass


class OpenAIProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        self.client = OpenAI()
        self.config = config

    def call(self, context: str, instructions: Optional[str]) -> Tuple[str, str]:
        instructions = f"{instructions or ''}\n\nThink step by step about the chess position and explain your reasoning before making your move."

        kwargs = {
            "model": self.config.name,
            "input": context,
            "instructions": instructions,
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
            kwargs["reasoning"] = {'effort': 'low'}
            kwargs['text']['format']['schema'] = CHESS_SCHEMA

        elif self.config.is_COT:
            kwargs['text']['format']['schema'] = CHESS_SCHEMA_WITH_REASONING

        else:
            kwargs['text']['format']['schema'] = CHESS_SCHEMA

        response = self.client.responses.create(**kwargs)
        parsed_response = json.loads(response.output_text)
        move = parsed_response['chess_move_SAN']
        reasoning = parsed_response['reasoning'] if self.config.is_COT else response['reasoning']['summary'] if self.config.is_reasoning else 'None'

        return move, reasoning


class AnthropicProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        self.client = Anthropic()
        self.config = config

    def call(self, context: str, instructions: Optional[str]) -> Tuple[str, str]:
        instructions = f"{instructions or ''}\n\nThink step by step about the chess position and explain your reasoning before making your move."
        schema = {}


        kwargs = {
            'model': self.config.name,
            'system': instructions,
            'messages': [
                {'role': 'user', 'content': ''},
            ],
            'max_tokens': 1030
        }

        user_message = f'{context}\n\nFormat your response as valid JSON matching this schema. Respond only with JSON: '

        if self.config.is_reasoning:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': 1024}
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
