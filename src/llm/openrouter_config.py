"""
OpenRouter Configuration Utility
Helps manage and validate OpenRouter API settings
"""

import os
from typing import List, Optional, Dict

class OpenRouterConfig:
    @staticmethod
    def get_provider_order() -> Optional[List[str]]:
        """Get configured provider order"""
        provider_order_str = os.getenv('OPENROUTER_PROVIDER_ORDER', '')
        if provider_order_str:
            return [p.strip() for p in provider_order_str.split(',') if p.strip()]
        return None
    
    @staticmethod
    def is_reasoning_enabled() -> bool:
        """Check if reasoning is enabled"""
        return os.getenv('OPENROUTER_REASONING_ENABLED', 'false').lower() == 'true'
    
    @staticmethod
    def should_exclude_reasoning() -> bool:
        """Check if reasoning tokens should be excluded"""
        return os.getenv('OPENROUTER_REASONING_EXCLUDE', 'false').lower() == 'true'
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get OpenRouter API key"""
        return os.getenv('OPENROUTER_API_KEY')
    
    @staticmethod
    def get_referer() -> str:
        """Get HTTP referer for API requests"""
        return os.getenv('OPENROUTER_REFERER', 'https://github.com/Leejames2305/LLM-Assisted-Path-Negotiation-on-Warehouse')
    
    @staticmethod
    def get_title() -> str:
        """Get X-Title for API requests"""
        return os.getenv('OPENROUTER_TITLE', 'LLM Multi-Robot Navigation')
    
    @staticmethod
    def get_central_model() -> str:
        """Get central negotiator model"""
        return os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')
    
    @staticmethod
    def get_agent_model() -> str:
        """Get agent validator model"""
        return os.getenv('AGENT_LLM_MODEL', 'google/gemma-2-9b-it:free')
    
    @staticmethod
    def print_config():
        """Print current OpenRouter configuration"""
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
    def validate_config() -> Dict[str, bool]:
        """Validate configuration and return status"""
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
        """Get keywords that typically indicate reasoning-capable models"""
        return [
            'o1',          # OpenAI o1 series
            'reasoning',   # Models with 'reasoning' in name
            'reasoner',    # DeepSeek reasoner variants
            'think',       # Models that emphasize thinking
            'analysis',    # Analysis-focused models
            'logic'        # Logic-focused models
        ]
    
    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        """
        Dynamically check if a model supports reasoning features
        Uses multiple detection methods rather than hard-coded lists
        """
        model_lower = model.lower()
        
        # Method 1: Check for reasoning keywords
        reasoning_keywords = OpenRouterConfig.get_reasoning_keywords()
        has_reasoning_keyword = any(keyword in model_lower for keyword in reasoning_keywords)
        
        # Method 2: Check for high-end model indicators
        high_end_indicators = [
            'gpt-4',       # GPT-4 variants tend to have better reasoning
            'claude-3',    # Claude-3 series
            'opus',        # Anthropic's highest tier
            'ultra',       # Google's premium tier
            'max',         # Often indicates top-tier models
            'pro',         # Professional/premium variants
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
        """Check if a model should be excluded from reasoning detection"""
        # Some high-end models that don't support reasoning features
        exclusions = [
            'turbo',       # Often indicates speed over reasoning
            'fast',        # Speed-optimized variants
            'lite',        # Lightweight versions
            'mini',        # Unless it's o1-mini, which is special
            'free'         # Free tier models usually lack advanced features
        ]
        
        # Special case: o1-mini is actually a reasoning model despite having 'mini'
        if 'o1' in model_lower and 'mini' in model_lower:
            return False
            
        return any(exclusion in model_lower for exclusion in exclusions)
    
    @staticmethod
    def get_recommended_settings_for_model(model: str) -> Dict:
        """Get recommended settings for a specific model"""
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
        
        # Model-specific overrides
        if 'gemma' in model.lower():
            settings['temperature'] = 0.2
        elif 'claude' in model.lower():
            settings['temperature'] = 0.4
        elif 'gpt-4' in model.lower():
            settings['temperature'] = 0.2
        
        return settings
    
    @staticmethod
    def suggest_provider_order_for_task(task_type: str) -> Optional[List[str]]:
        """Suggest provider order based on task type"""
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
        """Generate debug information for a request payload"""
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