from typing import Any, Generator, List, Optional, Union
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
    AssistantPromptMessage,
    PromptMessageTool
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError
)
import requests
import json
import logging

logger = logging.getLogger(__name__)

class StimaLargeLanguageModel(LargeLanguageModel):
    """
    Stima LLM 模型實作
    """
    
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: List[PromptMessage],
        model_parameters: dict,
        tools: Optional[List[PromptMessageTool]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = True,
        user: Optional[str] = None
    ) -> Union[LLMResult, Generator]:
        """
        調用 LLM 模型
        """
        # 取得 API 設定
        api_key = credentials.get('api_key')
        api_base = credentials.get('api_base', 'https://api.stima.ai/v1')
        
        # 建立請求標頭
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # 轉換訊息格式
        messages = self._convert_messages(prompt_messages)
        
        # 建立請求資料
        data = {
            'model': model,
            'messages': messages,
            'temperature': model_parameters.get('temperature', 0.7),
            'max_tokens': model_parameters.get('max_tokens', 4096),
            'top_p': model_parameters.get('top_p', 1.0),
            'stream': stream
        }
        
        # 加入可選參數
        if stop:
            data['stop'] = stop
        if tools:
            data['tools'] = [self._convert_tool(tool) for tool in tools]
        if user:
            data['user'] = user
        
        try:
            # 發送請求
            response = requests.post(
                f'{api_base}/chat/completions',
                headers=headers,
                json=data,
                stream=stream,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f'API request failed with status {response.status_code}'
                try:
                    error_detail = response.json()
                    error_msg += f': {error_detail}'
                except:
                    error_msg += f': {response.text}'
                raise InvokeError(error_msg)
            
            # 處理回應
            if stream:
                return self._handle_stream_response(response, model, prompt_messages)
            else:
                return self._handle_response(response, model, prompt_messages)
                
        except requests.exceptions.RequestException as e:
            raise InvokeError(f'API request failed: {str(e)}')
    
    def _convert_messages(self, prompt_messages: List[PromptMessage]) -> List[dict]:
        """
        轉換訊息格式為 API 格式
        """
        messages = []
        for msg in prompt_messages:
            if isinstance(msg, SystemPromptMessage):
                messages.append({
                    'role': 'system',
                    'content': msg.content
                })
            elif isinstance(msg, UserPromptMessage):
                messages.append({
                    'role': 'user',
                    'content': msg.content
                })
            elif isinstance(msg, AssistantPromptMessage):
                messages.append({
                    'role': 'assistant',
                    'content': msg.content
                })
        return messages
    
    def _convert_tool(self, tool: PromptMessageTool) -> dict:
        """
        轉換工具定義格式
        """
        return {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters
            }
        }
    
    def _handle_response(self, response: requests.Response, model: str, prompt_messages: List) -> LLMResult:
        """
        處理非串流回應
        """
        data = response.json()
        
        # 提取回應內容
        choice = data['choices'][0]
        message = choice['message']
        
        # 建立結果
        result = LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=AssistantPromptMessage(
                content=message.get('content', ''),
                tool_calls=message.get('tool_calls')
            ),
            usage=LLMUsage(
                prompt_tokens=data['usage']['prompt_tokens'],
                completion_tokens=data['usage']['completion_tokens']
            )
        )
        
        return result
    
    def _handle_stream_response(self, response: requests.Response, model: str, prompt_messages: List) -> Generator:
        """
        處理串流回應
        """
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                
                if data_str == '[DONE]':
                    break
                
                try:
                    chunk_data = json.loads(data_str)
                    choice = chunk_data['choices'][0]
                    delta = choice.get('delta', {})
                    
                    # 建立 chunk
                    chunk = LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            message=AssistantPromptMessage(
                                content=delta.get('content', ''),
                                tool_calls=delta.get('tool_calls')
                            ),
                            finish_reason=choice.get('finish_reason')
                        )
                    )
                    
                    yield chunk
                    
                except json.JSONDecodeError:
                    logger.error(f'Failed to parse streaming response: {data_str}')
                    continue
    
    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: List[PromptMessage],
        tools: Optional[List[PromptMessageTool]] = None
    ) -> int:
        """
        計算 token 數量（簡化實作）
        """
        # 簡單估算：每 4 個字元約為 1 個 token
        total_chars = 0
        for msg in prompt_messages:
            if hasattr(msg, 'content'):
                total_chars += len(str(msg.content))
        
        # 如果有工具，也要計算
        if tools:
            for tool in tools:
                total_chars += len(tool.name) + len(tool.description)
                total_chars += len(json.dumps(tool.parameters))
        
        estimated_tokens = total_chars // 4
        return max(estimated_tokens, 1)
    
    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        驗證模型憑證
        """
        try:
            # 發送一個簡單的測試請求
            self._invoke(
                model=model,
                credentials=credentials,
                prompt_messages=[UserPromptMessage(content="test")],
                model_parameters={'max_tokens': 5},
                stream=False
            )
        except Exception as e:
            raise CredentialsValidateFailedError(str(e))
