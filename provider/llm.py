from typing import Any, Generator
from dify_plugin.entities.model.llm import LLMModelConfig, LLMResult
from dify_plugin.interfaces.model.llm import LLMModel
from openai import OpenAI

class TeamProxyModel(LLMModel):
    def _get_client(self, credentials: dict) -> OpenAI:
        return OpenAI(
            api_key=credentials.get("api_key"),
            # 如果你在 yaml 裡有開 base_url 就用 get，沒有就寫死字串
            base_url=credentials.get("base_url", "https://api.openai.com/v1"), 
        )

    def invoke(self, model: str, credentials: dict, prompt_messages: list, model_parameters: dict, **kwargs) -> LLMResult:
        client = self._get_client(credentials)
        # 簡單轉換 Dify 訊息格式到 OpenAI 格式
        messages = [{"role": m.role, "content": m.content} for m in prompt_messages]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **model_parameters
        )

        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=response.choices[0].message,
            usage=response.usage
        )

    def stream_invoke(self, model: str, credentials: dict, prompt_messages: list, model_parameters: dict, **kwargs) -> Generator[Any, None, None]:
        client = self._get_client(credentials)
        messages = [{"role": m.role, "content": m.content} for m in prompt_messages]
        
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **model_parameters
        )

        for chunk in stream:
            yield chunk
