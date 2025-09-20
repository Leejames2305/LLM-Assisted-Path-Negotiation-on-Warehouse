"""
OpenRouter API Client for LLM Communication
"""

import requests
import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # New environment variables for advanced features
        self.provider_order = self._parse_provider_order()
        self.reasoning_enabled = os.getenv('OPENROUTER_REASONING_ENABLED', 'false').lower() == 'true'
        self.reasoning_exclude = os.getenv('OPENROUTER_REASONING_EXCLUDE', 'false').lower() == 'true'
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv('OPENROUTER_REFERER', "https://github.com/Leejames2305/LLM-Assisted-Path-Negotiation-on-Warehouse"),
            "X-Title": os.getenv('OPENROUTER_TITLE', "LLM Multi-Robot Navigation")
        }
    
    def _parse_provider_order(self) -> Optional[List[str]]:
        """Parse provider order from environment variable"""
        provider_order_str = os.getenv('OPENROUTER_PROVIDER_ORDER', '')
        if provider_order_str:
            return [p.strip() for p in provider_order_str.split(',') if p.strip()]
        return None
    
    def send_request(self, model: str, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> Optional[str]:
        """Send a request to OpenRouter API with new advanced parameters"""
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
        
        # Build the request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add provider configuration if specified
        if self.provider_order:
            payload["provider"] = {
                "order": self.provider_order
            }
            print(f"🔧 Using provider order: {self.provider_order}")
        
        # Add reasoning configuration if enabled
        if self.reasoning_enabled or self.reasoning_exclude:
            payload["reasoning"] = {}
            if self.reasoning_enabled:
                payload["reasoning"]["enabled"] = True
                print("🧠 Reasoning enabled")
            if self.reasoning_exclude:
                payload["reasoning"]["exclude"] = True
                print("🚫 Reasoning tokens excluded from response")
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        try:
            print(f"📡 Sending request to {model}...")
            if self.provider_order:
                print(f"   Provider preference: {self.provider_order}")
            
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            finish_reason = data['choices'][0].get('finish_reason', 'unknown')
            
            # Log provider information if available
            if 'provider' in data:
                print(f"✅ Response from provider: {data['provider']}")
            
            # Log finish reason for debugging
            print(f"🔍 DEBUG: OpenRouter finish_reason: {finish_reason}")
            
            if finish_reason == 'length':
                print("⚠️  WARNING: Response was truncated due to token limit!")
                print(f"🔍 Truncated content: {content[:200]}...")
            
            return content
        
        except requests.RequestException as e:
            print(f"❌ Error making request to OpenRouter: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    print(f"🔍 Error details: {error_details}")
                except:
                    print(f"🔍 Response content: {e.response.text}")
            return None
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def create_system_message(self, content: str) -> Dict:
        """Create a system message"""
        return {"role": "system", "content": content}
    
    def create_user_message(self, content: str) -> Dict:
        """Create a user message"""
        return {"role": "user", "content": content}
    
    def create_assistant_message(self, content: str) -> Dict:
        """Create an assistant message"""
        return {"role": "assistant", "content": content}
