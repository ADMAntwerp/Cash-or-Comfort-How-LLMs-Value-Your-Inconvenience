from abc import ABC, abstractmethod
from httpx import options, request
from joblib import logger
from openai import OpenAI, APIConnectionError, APIStatusError, NotFoundError
from openai._types import NOT_GIVEN
from llamaapi import LlamaAPI
from google import genai
from google.genai import types
import anthropic
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger(__name__)


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
        providers=None,
        quantization_tiers=None,
        allow_provider_fallbacks=False,
    ):
        self.system_role = system_role
        self.model = model
        self.temperature = temperature
        self.providers = providers
        self.quantization_tiers = ([list(t) for t in quantization_tiers] if quantization_tiers else [None])
        self.allow_provider_fallbacks = allow_provider_fallbacks

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.messages = []

    def _provider_options(self, quantizations):
        options = {}
        if quantizations:
            options["quantizations"] = quantizations
        if self.providers:
            options["order"] = self.providers
            options["allow_fallbacks"] = self.allow_provider_fallbacks
        return options

    @staticmethod
    def _is_retryable(error):
        """Errors where trying the next quantization tier makes sense."""
        if isinstance(error, (NotFoundError, APIConnectionError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code == 429 or error.status_code >= 500
        return False

    def generate_response(self, prompt, history=None, stream=False):
        history = history or []
        messages = (
            [{"role": "system", "content": self.system_role}]
            + history
            + [{"role": "user", "content": prompt}]
        )

        last_error = None
        for quantizations in self.quantization_tiers:
            options = self._provider_options(quantizations)
            kwargs = {"extra_body": {"provider": options}} if options else {}

            try:
                original_send = self.client._client.send

                def inspect_send(request, *args, **kwargs):

                    auth = request.headers.get("authorization", "")

                    scheme, separator, credential = auth.partition(" ")

                    print("\n--- AUTHENTICATION INSPECTION ---")
                    print("Authorization header exists:", bool(auth))
                    print("Authentication scheme:", scheme)
                    print("Bearer separator present:", bool(separator))
                    print("Credential present:", bool(credential))
                    print("Credential length:", len(credential))
                    print("Starts with sk-or-:", credential.startswith("sk-or-"))
                    print("Contains whitespace:", any(c.isspace() for c in credential))
                    print("Contains Bearer prefix:", credential.lower().startswith("bearer "))
                    print("---------------------------------\n")

                    return original_send(request, *args, **kwargs)

                self.client._client.send = inspect_send

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:
                if not self._is_retryable(e):
                    raise
                log.warning(
                    "Request failed for quantizations=%s. Trying next tier.",
                    quantizations,
                )
                last_error = e
                continue

            if not stream:
                log.info(
                    "Served by provider=%s model=%s quantizations=%s)",
                    getattr(completion, "provider", None),
                    completion.model,
                    quantizations,
                )
                return completion.choices[0].message.content
            return completion

        raise last_error


class GptApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
        providers=None,
        quantization_tiers=None,
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

        log.info(
            "Served by provider=openai requested_model=%s served_model=%s id=%s",
            self.model,
            completion.model,
            completion.id,
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
