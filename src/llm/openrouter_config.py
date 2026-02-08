"""
OpenRouter Configuration Utility
Helps manage and validate OpenRouter API settings
"""

import os
from typing import List, Optional, Dict

class OpenRouterConfig:
    @staticmethod
    # Get provider order from environment variable
    def get_provider_order() -> Optional[List[str]]:
        provider_order_str = os.getenv('OPENROUTER_PROVIDER_ORDER', '')
        if provider_order_str:
            return [p.strip() for p in provider_order_str.split(',') if p.strip()]
        return None
    
    @staticmethod
    # Check if reasoning is enabled
    def is_reasoning_enabled() -> bool:
        return os.getenv('OPENROUTER_REASONING_ENABLED', 'false').lower() == 'true'
    
    @staticmethod
    # Check if reasoning tokens should be excluded
    def should_exclude_reasoning() -> bool:
        return os.getenv('OPENROUTER_REASONING_EXCLUDE', 'false').lower() == 'true'
    
    @staticmethod
    # OpenRouter API key
    def get_api_key() -> Optional[str]:
        return os.getenv('OPENROUTER_API_KEY')
    
    @staticmethod
    # Get X-Referer for API requests
    def get_referer() -> str:
        return os.getenv('OPENROUTER_REFERER', 'https://github.com/Leejames2305/LLM-Assisted-Path-Negotiation-on-Warehouse')
    
    @staticmethod
    # Get X-Title for API requests
    def get_title() -> str:
        return os.getenv('OPENROUTER_TITLE', 'LLM Multi-Robot Navigation')
    
    @staticmethod
    # Get central negotiator model, default to GLM-4.5-Air
    def get_central_model() -> str:
        return os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')
    
    @staticmethod
    # Get agent validator model, default to Gemma-2-9B-IT
    def get_agent_model() -> str:
        return os.getenv('AGENT_LLM_MODEL', 'google/gemma-2-9b-it:free')
    
    @staticmethod
    def print_config():
        print("🔧 OpenRouter Configuration:")
        print(f"   API Key: {'✅ Set' if OpenRouterConfig.get_api_key() else '❌ Not set'}")
        print(f"   Provider Order: {OpenRouterConfig.get_provider_order()}")
        print(f"   Reasoning Enabled: {OpenRouterConfig.is_reasoning_enabled()}")
        print(f"   Reasoning Exclude: {OpenRouterConfig.should_exclude_reasoning()}")
        print(f"   Central Model: {OpenRouterConfig.get_central_model()}")
        print(f"   Agent Model: {OpenRouterConfig.get_agent_model()}")
        print(f"   Referer: {OpenRouterConfig.get_referer()}")
        print(f"   Title: {OpenRouterConfig.get_title()}")
    
    @staticmethod
    # Validate configuration
    def validate_config() -> Dict[str, bool]:
        status = {
            'api_key_set': bool(OpenRouterConfig.get_api_key()),
            'models_configured': bool(OpenRouterConfig.get_central_model() and OpenRouterConfig.get_agent_model()),
            'valid_provider_order': True,
            'reasoning_config_valid': True
        }
        
        # Validate provider order format
        provider_order = OpenRouterConfig.get_provider_order()
        if provider_order:
            status['valid_provider_order'] = all(isinstance(p, str) and p for p in provider_order)
        
        # Validate reasoning configuration
        if OpenRouterConfig.is_reasoning_enabled() and OpenRouterConfig.should_exclude_reasoning():
            # This combination might not make sense for some use cases
            print("⚠️  Warning: Reasoning is enabled but tokens are excluded from response")
        
        return status
    
    @staticmethod
    def get_reasoning_keywords() -> List[str]:
        return [
            'reasoning',   # Models with 'reasoning' in name
            'thinking',    # Kimi Thinking Series
            'think'        # Models that emphasize thinking
        ]
    
    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        model_lower = model.lower()
        
        # Method 1: Check for reasoning keywords
        reasoning_keywords = OpenRouterConfig.get_reasoning_keywords()
        has_reasoning_keyword = any(keyword in model_lower for keyword in reasoning_keywords)
        
        # Method 2: Check for high-end model indicators
        high_end_indicators = [
            'gpt',       # GPT-5 variants tend to have better reasoning
            'claude',    # Claude series
            'gemini',    # Google's premium tier
            'grok',      # Grok models
            'glm',       # GLM models
            'deepseek'   # DeepSeek models
        ]
        is_high_end = any(indicator in model_lower for indicator in high_end_indicators)
        
        # Method 3: Check environment override
        forced_reasoning = os.getenv('FORCE_REASONING_MODE', 'false').lower() == 'true'
        
        # Method 4: Check for explicit model configuration
        reasoning_models_env = os.getenv('REASONING_MODELS', '')
        if reasoning_models_env:
            explicit_models = [m.strip().lower() for m in reasoning_models_env.split(',')]
            is_explicit = model_lower in explicit_models
        else:
            is_explicit = False
        
        # Combine detection methods
        is_reasoning = (
            has_reasoning_keyword or 
            (is_high_end and not OpenRouterConfig._is_excluded_from_reasoning(model_lower)) or
            forced_reasoning or
            is_explicit
        )
        
        # Log detection reasoning for debugging
        if is_reasoning:
            reasons = []
            if has_reasoning_keyword: reasons.append("keyword match")
            if is_high_end: reasons.append("high-end model")
            if forced_reasoning: reasons.append("forced by env")
            if is_explicit: reasons.append("explicit config")
            print(f"🧠 Detected reasoning model '{model}': {', '.join(reasons)}")
        
        return is_reasoning
    
    @staticmethod
    def _is_excluded_from_reasoning(model_lower: str) -> bool:
        # Some high-end models that don't support reasoning features
        exclusions = [
            'verynano'       # Non-existent
        ]
            
        return any(exclusion in model_lower for exclusion in exclusions)
    
    @staticmethod
    def get_recommended_settings_for_model(model: str) -> Dict:
        settings = {
            'max_tokens': 20000,
            'temperature': 0.3,
            'reasoning_enabled': False,
            'reasoning_exclude': False
        }
        
        if OpenRouterConfig.is_reasoning_model(model):
            settings.update({
                'max_tokens': 30000,
                'temperature': 0.1,
                'reasoning_enabled': True,
                'reasoning_exclude': False
            })

        return settings
    
    @staticmethod
    def suggest_provider_order_for_task(task_type: str) -> Optional[List[str]]:
        suggestions = {
            'reasoning': ['OpenAI', 'Anthropic', 'Google'],
            'fast_validation': ['DeepInfra', 'Together AI', 'Fireworks AI'],
            'cost_effective': ['DeepInfra', 'Together AI', 'OpenAI'],
            'high_quality': ['OpenAI', 'Anthropic', 'Google'],
            'experimental': ['Together AI', 'Fireworks AI', 'DeepInfra']
        }
        
        return suggestions.get(task_type)
    
    @staticmethod
    def debug_request_payload(model: str, **kwargs) -> Dict:
        config = OpenRouterConfig.get_recommended_settings_for_model(model)
        provider_order = OpenRouterConfig.get_provider_order()
        
        payload_info = {
            'model': model,
            'is_reasoning_model': OpenRouterConfig.is_reasoning_model(model),
            'recommended_settings': config,
            'provider_order': provider_order,
            'reasoning_enabled': OpenRouterConfig.is_reasoning_enabled(),
            'reasoning_exclude': OpenRouterConfig.should_exclude_reasoning(),
            'additional_params': kwargs
        }
        
        return payload_info