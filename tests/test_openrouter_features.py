"""
Test script to demonstrate OpenRouter enhanced features
"""

import sys
import os
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm.openrouter_config import OpenRouterConfig
from src.llm import OpenRouterClient

def test_openrouter_features():
    """Test the new OpenRouter features"""
    load_dotenv()
    
    print("=== OpenRouter Enhanced Features Test ===\n")
    
    # Print current configuration
    OpenRouterConfig.print_config()
    print()
    
    # Validate configuration
    status = OpenRouterConfig.validate_config()
    print("🔍 Configuration Validation:")
    for key, value in status.items():
        print(f"   {key}: {'✅' if value else '❌'}")
    print()
    
    # Test reasoning model detection for configured models
    configured_models = [
        OpenRouterConfig.get_central_model(),
        OpenRouterConfig.get_agent_model()
    ]
    
    # Remove duplicates while preserving order
    unique_models = []
    for model in configured_models:
        if model not in unique_models:
            unique_models.append(model)
    
    print("🧠 Reasoning Detection for Configured Models:")
    for model in unique_models:
        is_reasoning = OpenRouterConfig.is_reasoning_model(model)
        role = "Central" if model == OpenRouterConfig.get_central_model() else "Agent"
        if model == OpenRouterConfig.get_central_model() and model == OpenRouterConfig.get_agent_model():
            role = "Central & Agent"
        print(f"   {model} ({role}): {'🧠 Reasoning' if is_reasoning else '🔧 Standard'}")
    print()
    
    # Test with environment overrides
    print("🔧 Testing Environment Overrides:")
    
    # Test explicit model list
    os.environ['REASONING_MODELS'] = 'custom-model,another-custom'
    print(f"   Set REASONING_MODELS=custom-model,another-custom")
    print(f"   custom-model: {'🧠 Reasoning' if OpenRouterConfig.is_reasoning_model('custom-model') else '🔧 Standard'}")
    print(f"   another-custom: {'🧠 Reasoning' if OpenRouterConfig.is_reasoning_model('another-custom') else '🔧 Standard'}")
    print(f"   unrelated-model: {'🧠 Reasoning' if OpenRouterConfig.is_reasoning_model('unrelated-model') else '🔧 Standard'}")
    
    # Clean up
    if 'REASONING_MODELS' in os.environ:
        del os.environ['REASONING_MODELS']
    print()
    
    # Test recommended settings
    print("⚙️  Recommended Settings for Configured Models:")
    for model in unique_models:
        settings = OpenRouterConfig.get_recommended_settings_for_model(model)
        role = "Central" if model == OpenRouterConfig.get_central_model() else "Agent"
        if model == OpenRouterConfig.get_central_model() and model == OpenRouterConfig.get_agent_model():
            role = "Central & Agent"
        print(f"   {model} ({role}):")
        for key, value in settings.items():
            print(f"     {key}: {value}")
        print()
    
    # Test provider suggestions
    print("🏢 Provider Order Suggestions:")
    task_types = ['reasoning', 'fast_validation', 'cost_effective', 'high_quality']
    for task in task_types:
        providers = OpenRouterConfig.suggest_provider_order_for_task(task)
        print(f"   {task}: {providers}")
    print()
    
    # Test client initialization (without actually making requests)
    if OpenRouterConfig.get_api_key():
        print("🤖 OpenRouter Client Test:")
        client = OpenRouterClient()
        
        # Test debug payload generation with configured central model
        debug_info = OpenRouterConfig.debug_request_payload(
            OpenRouterConfig.get_central_model(),
            custom_param='test_value'
        )
        
        print("📊 Debug Payload Info for Central Model:")
        for key, value in debug_info.items():
            print(f"   {key}: {value}")
        print()
        
        print("✅ Client initialized successfully with enhanced features!")
    else:
        print("⚠️  No API key found - skipping client test")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_openrouter_features()