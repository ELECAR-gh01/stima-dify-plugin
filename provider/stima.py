import logging
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin import ModelProvider

logger = logging.getLogger(__name__)

class StimaProvider(ModelProvider):
    """
    Stima API Provider
    """
    
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        驗證 API 憑證
        
        :param credentials: 包含 api_key 和 api_base 的字典
        """
        try:
            # 檢查必要的 API Key
            if not credentials.get('api_key'):
                raise CredentialsValidateFailedError('API Key is required')
            
            # 取得 API 基礎網址
            api_base = credentials.get('api_base', 'https://api.stima.ai/v1')
            
            # 可以在這裡加入實際的 API 測試
            # 例如：呼叫一個簡單的 endpoint 來驗證 key 是否有效
            import requests
            
            headers = {
                'Authorization': f'Bearer {credentials.get("api_key")}',
                'Content-Type': 'application/json'
            }
            
            # 測試連線（根據你們的 API 調整）
            test_url = f'{api_base}/models'
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 401:
                raise CredentialsValidateFailedError('Invalid API Key')
            elif response.status_code != 200:
                logger.warning(f'API test returned status {response.status_code}')
                # 某些 API 可能不支援 /models，這裡可以選擇忽略或使用其他測試方式
                
        except requests.exceptions.ConnectionError:
            raise CredentialsValidateFailedError('Cannot connect to API endpoint')
        except requests.exceptions.Timeout:
            raise CredentialsValidateFailedError('API request timeout')
        except CredentialsValidateFailedError:
            raise
        except Exception as e:
            logger.exception(f'Stima credentials validation failed: {str(e)}')
            raise CredentialsValidateFailedError(f'Validation failed: {str(e)}')
