from abc import ABC, abstractmethod
from openai import OpenAI
from openai._types import NOT_GIVEN
from llamaapi import LlamaAPI
from google import genai
from google.genai import types
import anthropic
import json


class LLMWrapper(ABC):
    @abstractmethod
    def generate_response(self, prompt):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass


class OpenRouterApi(LLMWrapper):
    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
    ):
        self.system_role = system_role
        self.model = model
        self.temperature = temperature

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.messages = []

    def generate_response(self, prompt, history=[], stream=False):
        messages = history + [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=stream,
        )

        if stream:
            return completion
        else:
            return completion.choices[0].message.content


class GptApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
    ):
        self.system_role = system_role
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.messages = []
        self.temperature = temperature

    def generate_response(self, prompt):

        messages = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )

        return completion.choices[0].message.content


class ClaudeApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
    ):

        self.system_role = system_role
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages = []
        self.temperature = temperature

    def generate_response(self, prompt):

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            system=self.system_role,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        return message.content[0].text


class GeminiApi(LLMWrapper):
    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
    ):

        self.system_role = system_role
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature

    def generate_response(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_role, temperature=self.temperature
            ),
            contents=prompt,
        )

        return response.text


class DeepSeekApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
        base_url="https://api.deepseek.com/v1",
    ):
        self.system_role = system_role
        self.model = model
        self.messages = []
        self.temperature = temperature
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def generate_response(self, prompt):

        messages = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )

        return completion.choices[0].message.content
