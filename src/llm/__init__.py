"""
OpenRouter API Client for LLM Communication
"""

import requests
import json
import os
from typing import Dict, List, Optional, Tuple, Union
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
        
        # Track cumulative token usage across all requests
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
    
    def _parse_provider_order(self) -> Optional[List[str]]:
        """Parse provider order from environment variable"""
        provider_order_str = os.getenv('OPENROUTER_PROVIDER_ORDER', '')
        if provider_order_str:
            return [p.strip() for p in provider_order_str.split(',') if p.strip()]
        return None
    
    def get_token_usage(self) -> Dict:
        """Get cumulative token usage statistics"""
        return {
            'total_tokens': self.total_tokens_used,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'request_count': self.request_count
        }
    
    def reset_token_usage(self):
        """Reset token usage counters"""
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
    
    def send_request(self, model: str, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7, return_full_response: bool = False, **kwargs) -> Union[Optional[str], Tuple[Optional[str], Dict]]:
        """
        Send a request to OpenRouter API with new advanced parameters
        
        Args:
            model: The model to use
            messages: List of message dicts
            max_tokens: Maximum tokens in response
            temperature: Temperature for sampling
            return_full_response: If True, returns (content, usage_dict) tuple
            **kwargs: Additional parameters
            
        Returns:
            If return_full_response=False: content string or None
            If return_full_response=True: (content, usage_dict) tuple
        """
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
        
        # Add reasoning configuration if enabled
        if self.reasoning_enabled or self.reasoning_exclude:
            payload["reasoning"] = {}
            if self.reasoning_enabled:
                payload["reasoning"]["enabled"] = True
            if self.reasoning_exclude:
                payload["reasoning"]["exclude"] = True
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        usage_data = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        try:
            print(f"📡 Sending request to {model}...")
            
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            finish_reason = data['choices'][0].get('finish_reason', 'unknown')
            
            # Extract token usage from response
            if 'usage' in data:
                usage_data = {
                    'prompt_tokens': data['usage'].get('prompt_tokens', 0),
                    'completion_tokens': data['usage'].get('completion_tokens', 0),
                    'total_tokens': data['usage'].get('total_tokens', 0)
                }
                
                # Update cumulative counters
                self.total_prompt_tokens += usage_data['prompt_tokens']
                self.total_completion_tokens += usage_data['completion_tokens']
                self.total_tokens_used += usage_data['total_tokens']
                self.request_count += 1
                
                print(f"📊 Tokens used: {usage_data['total_tokens']} (prompt: {usage_data['prompt_tokens']}, completion: {usage_data['completion_tokens']})")
            
            # Log provider information if available
            if 'provider' in data:
                print(f"✅ Response from provider: {data['provider']}")
            
            if finish_reason == 'length':
                print("⚠️  WARNING: Response truncated due to token limit!")
            
            if return_full_response:
                return content, usage_data
            return content
        
        except requests.RequestException as e:
            print(f"❌ Error making request to OpenRouter: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    print(f"🔍 Error details: {error_details}")
                except:
                    print(f"🔍 Response content: {e.response.text}")
            if return_full_response:
                return None, usage_data
            return None
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            if return_full_response:
                return None, usage_data
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            if return_full_response:
                return None, usage_data
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
