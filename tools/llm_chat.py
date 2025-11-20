from collections.abc import Generator
from typing import Any
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class LlmChatTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        調用 LLM 模型
        """
        # 獲取憑證
        api_key = self.runtime.credentials.get("api_key")
        api_base = self.runtime.credentials.get("api_base", "https://api.stima.ai/v1")
        
        # 獲取參數
        model = tool_parameters.get("model", "gpt-4o")
        message = tool_parameters.get("message", "")
        
        if not message:
            yield self.create_text_message("Message cannot be empty")
            return
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 4096
            }
            
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                yield self.create_text_message(content)
            else:
                yield self.create_text_message(f"API Error: {response.status_code}")
                
        except Exception as e:
            yield self.create_text_message(f"Error: {str(e)}")
