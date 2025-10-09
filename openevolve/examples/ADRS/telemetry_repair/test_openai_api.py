#!/usr/bin/env python3
"""
Test OpenAI API connectivity for OpenEvolve
"""

import os
import sys
from openai import OpenAI

def test_api_basic():
    """Test basic API connectivity"""
    print("🔍 Testing OpenAI API...")
    
    # Check environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        return False
        
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with simple call
        print("🤖 Testing simple chat completion...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Start with cheaper model
            messages=[
                {"role": "user", "content": "Hello, can you respond with just 'API_TEST_SUCCESS'?"}
            ],
            max_tokens=50,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ Response: {result}")
        
        if "API_TEST_SUCCESS" in result:
            print("✅ Basic API test passed!")
            return True
        else:
            print("⚠️  API responded but unexpected content")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_o3_model():
    """Test o3 model specifically"""
    print("\n🧠 Testing o3 model...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return False
        
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "user", "content": "What is 2+2? Respond with just the number."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ o3 Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ o3 test failed: {e}")
        print("💡 Possible issues:")
        print("   - o3 model might need different API endpoint")
        print("   - Rate limiting")
        print("   - Model not available in your account")
        return False

def test_openevolve_config():
    """Test OpenEvolve's OpenAI configuration"""
    print("\n🔧 Testing OpenEvolve configuration...")
    
    try:
        # Import OpenEvolve components
        from openevolve.llm.openai import OpenAIModel
        from openevolve.config import ModelConfig
        
        # Create config matching your setup
        config = ModelConfig(
            model="o3",
            api_key=os.getenv('OPENAI_API_KEY'),
            api_base="https://api.openai.com/v1",
            temperature=0.7,
            max_tokens=1000,
            timeout=900
        )
        
        print(f"📝 Config: model={config.model}, api_base={config.api_base}")
        
        # Test OpenEvolve's OpenAI wrapper
        llm = OpenAIModel(config)
        
        response = llm.generate(
            messages=[
                {"role": "user", "content": "Hello from OpenEvolve! Just say 'OPENEVOLVE_TEST_SUCCESS'"}
            ]
        )
        
        print(f"✅ OpenEvolve Response: {response}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the OpenEvolve environment")
        return False
    except Exception as e:
        print(f"❌ OpenEvolve test failed: {e}")
        return False

def main():
    print("🚀 OpenAI API Diagnostic Test\n")
    
    print("📋 Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   Working Dir: {os.getcwd()}")
    print(f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    
    # Run tests
    basic_ok = test_api_basic()
    o3_ok = test_o3_model() if basic_ok else False
    openevolve_ok = test_openevolve_config() if basic_ok else False
    
    print(f"\n📊 Results:")
    print(f"   Basic API: {'✅' if basic_ok else '❌'}")
    print(f"   o3 Model: {'✅' if o3_ok else '❌'}")  
    print(f"   OpenEvolve: {'✅' if openevolve_ok else '❌'}")
    
    if all([basic_ok, o3_ok, openevolve_ok]):
        print("\n🎉 All tests passed! OpenEvolve should work.")
    else:
        print("\n🔧 Issues found. Check the errors above.")

if __name__ == "__main__":
    main() 