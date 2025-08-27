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
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Leejames2305/LLM-Assisted-Path-Negotiation-on-Warehouse",  # Replace with actual repo
            "X-Title": "LLM Multi-Robot Navigation"
        }
    
    def send_request(self, model: str, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
        """Send a request to OpenRouter API"""
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        except requests.RequestException as e:
            print(f"Error making request to OpenRouter: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    print(f"Error details: {error_details}")
                except:
                    print(f"Response content: {e.response.text}")
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
