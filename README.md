還是有問題，我把他們的內容也一起給你好了

langgenius_openrouter_0.0.10\.env.example

```
INSTALL_METHOD=remote
REMOTE_INSTALL_HOST=debug-plugin.dify.dev
REMOTE_INSTALL_PORT=5003
REMOTE_INSTALL_KEY=********-****-****-****-************


langgenius_openrouter_0.0.10\.verification.dify.json

{"authorized_category":"langgenius"}
```

langgenius_openrouter_0.0.10\main.py

```
from dify_plugin import Plugin, DifyPluginEnv

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

if __name__ == '__main__':
    plugin.run()
```

langgenius_openrouter_0.0.10\manifest.yaml

```
meta:
  arch:
    - amd64
    - arm64
  runner:
    entrypoint: main
    language: python
    version: "3.12"
  version: 0.0.1
name: openrouter
author: langgenius
icon: openrouter_square.svg
description:
  en_US: OpenRouter is a powerful platform that provides access to a diverse set of language models, including both commercial and open-source options.
  zh_Hans: OpenRouter 是一个强大的平台，提供对多种语言模型的访问，包括商业和开源选项。
label:
  en_US: OpenRouter
resource:
  memory: 268435456
  permission:
    model:
      enabled: false
type: plugin
plugins:
  models:
    - provider/openrouter.yaml
version: 0.0.10
created_at: 2024-09-20T00:13:50.29298939-04:00
```

langgenius_openrouter_0.0.10\README.md

```
# Overview
OpenRouter is a powerful platform that provides access to a diverse set of language models, including both commercial and open-source options. It enables developers to integrate various AI capabilities such as text generation, image generation, audio processing, and more through a unified API. OpenRouter supports models like GPT-4, Anthropic's Claude, Google's Gemini, and many others, allowing users to leverage advanced AI functionalities in their applications.

# Configuration
After installation, you need to get API keys from [OpenRouter](https://openrouter.ai/keys) and setup in Settings -> Model Provider.

![](_assets/openrouter.PNG)
```

langgenius_openrouter_0.0.10\requirements.txt

```
dify_plugin>=0.2.0,<0.3.0
```

langgenius_openrouter_0.0.10\models\llm\__init__.py

（空白）

langgenius_openrouter_0.0.10\models\llm\_position.yaml

```
- openai/o3-mini
- openai/o3-mini-2025-01-31
- openai/o1-preview
- openai/o1-mini
- openai/gpt-4o
- openai/gpt-4o-mini
- openai/gpt-4
- openai/gpt-4-32k
- openai/gpt-3.5-turbo
- anthropic/claude-3.7-sonnet
- anthropic/claude-3.5-sonnet
- anthropic/claude-3-haiku
- anthropic/claude-3-opus
- anthropic/claude-3-sonnet
- google/gemini-pro-1.5
- google/gemini-flash-1.5
- google/gemini-pro
- cohere/command-r-plus
- cohere/command-r
```

langgenius_openrouter_0.0.10\models\llm\llm.py

```
from collections.abc import Generator
from typing import Optional, Union
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from dify_plugin import OAICompatLargeLanguageModel


class OpenRouterLargeLanguageModel(OAICompatLargeLanguageModel):
    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = "https://openrouter.ai/api/v1"
        credentials["mode"] = self.get_model_mode(model).value
        credentials["function_calling_type"] = "tools"  # change to "tools"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)
        
        # Add parameter conversion logic
        if "functions" in model_parameters:
            model_parameters["tools"] = [{"type": "function", "function": func} for func in model_parameters.pop("functions")]
        if "function_call" in model_parameters:
            model_parameters["tool_choice"] = model_parameters.pop("function_call")
            
        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._update_credential(model, credentials)
        return super().validate_credentials(model, credentials)

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)
        
        # Add parameter conversion logic
        if "functions" in model_parameters:
            model_parameters["tools"] = [{"type": "function", "function": func} for func in model_parameters.pop("functions")]
        if "function_call" in model_parameters:
            model_parameters["tool_choice"] = model_parameters.pop("function_call")
            
        return super()._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def _generate_block_as_stream(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> Generator:
        resp = super()._generate(model, credentials, prompt_messages, model_parameters, tools, stop, False, user)
        yield LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=0,
                message=resp.message,
                usage=self._calc_response_usage(
                    model=model,
                    credentials=credentials,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                ),
                finish_reason="stop",
            ),
        )

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        self._update_credential(model, credentials)
        return super().get_customizable_model_schema(model, credentials)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(model, credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)
```

langgenius_openrouter_0.0.10\models\llm\claude-3.7-sonnet.yaml

```
model: anthropic/claude-3.7-sonnet
label:
  en_US: claude-3.7-sonnet
model_type: llm
features:
  - agent-thought
  - vision
  - tool-call
  - stream-tool-call
  - document
model_properties:
  mode: chat
  context_size: 200000
parameter_rules:
  - name: thinking
    label:
      zh_Hans: 推理模式
      en_US: Thinking Mode
    type: boolean
    default: false
    required: false
    help:
      zh_Hans: 控制模型的推理能力。启用时，temperature、top_p和top_k将被禁用。
      en_US: Controls the model's thinking capability. When enabled, temperature, top_p and top_k will be disabled.
  - name: thinking_budget
    label:
      zh_Hans: 推理预算
      en_US: Thinking Budget
    type: int
    default: 1024
    min: 0
    max: 128000
    required: false
    help:
      zh_Hans: 推理的预算限制（最小1024），必须小于max_tokens。仅在推理模式启用时可用。
      en_US: Budget limit for thinking (minimum 1024), must be less than max_tokens. Only available when thinking mode is enabled.
  - name: temperature
    use_template: temperature
  - name: top_p
    use_template: top_p
  - name: top_k
    label:
      zh_Hans: 取样数量
      en_US: Top k
    type: int
    help:
      zh_Hans: 仅从每个后续标记的前 K 个选项中采样。
      en_US: Only sample from the top K options for each subsequent token.
    required: false
  - name: max_tokens
    use_template: max_tokens
    required: true
    default: 64000
    min: 1
    max: 128000
  - name: response_format
    use_template: response_format
  - name: extended_output
    label:
      zh_Hans: 扩展输出
      en_US: Extended Output
    type: boolean
    default: false
    help:
      zh_Hans: 启用长达128K标记的输出能力。
      en_US: Enable capability for up to 128K output tokens.
pricing:
  input: "3.00"
  output: "15.00"
  unit: "0.000001"
  currency: USD
```

langgenius_openrouter_0.0.10\provider\openrouter.py

```
import logging
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin import ModelProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials(model="openai/gpt-4o-mini", credentials=credentials)
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(f"{self.get_provider_schema().provider} credentials validate failed")
            raise ex

```

langgenius_openrouter_0.0.10\provider\openrouter.yaml

```
background: '#F1EFED'
configurate_methods:
- predefined-model
- customizable-model
extra:
  python:
    model_sources:
    - models/llm/llm.py
    provider_source: provider/openrouter.py
help:
  title:
    en_US: Get your API key from openrouter.ai
    zh_Hans: 从 openrouter.ai 获取 API Key
  url:
    en_US: https://openrouter.ai/keys
icon_large:
  en_US: openrouter.svg
icon_small:
  en_US: openrouter_square.svg
label:
  en_US: OpenRouter
model_credential_schema:
  credential_form_schemas:
  - label:
      en_US: API Key
    placeholder:
      en_US: Enter your API Key
      zh_Hans: 在此输入您的 API Key
    required: true
    type: secret-input
    variable: api_key
  - default: chat
    label:
      en_US: Completion mode
    options:
    - label:
        en_US: Completion
        zh_Hans: 补全
      value: completion
    - label:
        en_US: Chat
        zh_Hans: 对话
      value: chat
    placeholder:
      en_US: Select completion mode
      zh_Hans: 选择对话类型
    required: false
    show_on:
    - value: llm
      variable: __model_type
    type: select
    variable: mode
  - default: '4096'
    label:
      en_US: Model context size
      zh_Hans: 模型上下文长度
    placeholder:
      en_US: Enter your Model context size
      zh_Hans: 在此输入您的模型上下文长度
    required: true
    type: text-input
    variable: context_size
  - default: '4096'
    label:
      en_US: Upper bound for max tokens
      zh_Hans: 最大 token 上限
    show_on:
    - value: llm
      variable: __model_type
    type: text-input
    variable: max_tokens_to_sample
  - default: no_support
    label:
      en_US: Vision Support
      zh_Hans: 是否支持 Vision
    options:
    - label:
        en_US: 'Yes'
        zh_Hans: 是
      value: support
    - label:
        en_US: 'No'
        zh_Hans: 否
      value: no_support
    required: false
    show_on:
    - value: llm
      variable: __model_type
    type: radio
    variable: vision_support
  model:
    label:
      en_US: Model Name
      zh_Hans: 模型名称
    placeholder:
      en_US: Enter full model name
      zh_Hans: 输入模型全称
models:
  llm:
    position: models/llm/_position.yaml
    predefined:
    - models/llm/*.yaml
provider: openrouter
provider_credential_schema:
  credential_form_schemas:
  - label:
      en_US: API Key
    placeholder:
      en_US: Enter your API Key
      zh_Hans: 在此输入您的 API Key
    required: true
    type: secret-input
    variable: api_key
supported_model_types:
- llm
```
