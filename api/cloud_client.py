"""
Async HTTP client for calling external LLM APIs.

This module is the adapter for cloud LLMs — same role as inference.py
is for your local model. The API layer calls simple methods here;
all HTTP complexity is hidden.
"""
import httpx
from typing import Optional
from api.config import settings


class CloudLLMClient:
    """
    Wraps calls to Anthropic or OpenAI APIs.
    
    Same interface as ModelInference.generate():
      - Takes a prompt (string)
      - Returns generated text (string)
      - Raises exceptions for errors
    
    This symmetry is deliberate — it's why we can route between them
    with a simple if/else in the endpoint.
    """
    
    def __init__(self):
        self.provider = settings.cloud_provider
        self.model = settings.cloud_model
        self.timeout = 30.0  # seconds before giving up
    
    def is_configured(self) -> bool:
        if self.provider == "anthropic":
            return settings.anthropic_api_key is not None
        elif self.provider == "openai":
            return settings.openai_api_key is not None
        elif self.provider == "mistral":  # NEW
            return settings.mistral_api_key is not None
        return False

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if self.provider == "anthropic":
            return await self._call_anthropic(prompt, max_tokens)
        elif self.provider == "openai":
            return await self._call_openai(prompt, max_tokens)
        elif self.provider == "mistral":  # NEW
            return await self._call_mistral(prompt, max_tokens)
        else:
            raise CloudAPIError(f"Unknown provider: {self.provider}")
    
    async def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Call Anthropic's Messages API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()  # Raises on 4xx/5xx
            except httpx.HTTPStatusError as e:
                raise CloudAPIError(
                    f"Anthropic API returned {e.response.status_code}: {e.response.text}"
                )
            except httpx.RequestError as e:
                raise CloudAPIError(f"Network error calling Anthropic: {e}")
        
        data = response.json()
        # Anthropic returns: {"content": [{"type": "text", "text": "..."}], ...}
        return data["content"][0]["text"]
    
    async def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI's Chat Completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise CloudAPIError(
                    f"OpenAI API returned {e.response.status_code}: {e.response.text}"
                )
            except httpx.RequestError as e:
                raise CloudAPIError(f"Network error calling OpenAI: {e}")
        
        data = response.json()
        # OpenAI returns: {"choices": [{"message": {"content": "..."}}], ...}
        return data["choices"][0]["message"]["content"]

    async def _call_mistral(self, prompt: str, max_tokens: int) -> str:
        """Call Mistral's Chat Completions API (OpenAI-compatible format)."""
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise CloudAPIError(
                    f"Mistral API returned {e.response.status_code}: {e.response.text}"
                )
            except httpx.RequestError as e:
                raise CloudAPIError(f"Network error calling Mistral: {e}")
    
            data = response.json()
            # Mistral returns same format as OpenAI: {"choices": [{"message": {"content": "..."}}]}
            return data["choices"][0]["message"]["content"]

class CloudAPIError(Exception):
    """Raised when the cloud LLM call fails."""
    pass


# Singleton
cloud_client = CloudLLMClient()