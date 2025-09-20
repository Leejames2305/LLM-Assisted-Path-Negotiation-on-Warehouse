"""
Quick Structure Test - Minimal dependency test for project structure
"""

import os
import sys

def check_file_structure():
    """Check that all required files and directories exist"""
    print("📁 Checking project structure...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        ".env.example",
        "README.md"
    ]
    
    required_dirs = [
        "src",
        "src/map_generator",
        "src/agents", 
        "src/llm",
        "src/navigation",
        "src/simulation",
        "logs"
    ]
    
    required_py_files = [
        "src/__init__.py",
        "src/map_generator/__init__.py",
        "src/agents/__init__.py", 
        "src/llm/__init__.py",
        "src/llm/central_negotiator.py",
        "src/llm/agent_validator.py",
        "src/navigation/__init__.py",
        "src/simulation/game_engine.py"
    ]
    
    all_good = True
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_good = False
    
    # Check directories  
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✅ {dir}/")
        else:
            print(f"❌ {dir}/ - MISSING")
            all_good = False
    
    # Check Python files
    for pyfile in required_py_files:
        if os.path.exists(pyfile):
            size = os.path.getsize(pyfile)
            print(f"✅ {pyfile} ({size} bytes)")
        else:
            print(f"❌ {pyfile} - MISSING")
            all_good = False
    
    return all_good

def check_basic_syntax():
    """Basic syntax check without imports"""
    print("\n🐍 Checking Python syntax...")
    
    python_files = [
        "main.py",
        "src/__init__.py",
        "src/map_generator/__init__.py",
        "src/agents/__init__.py",
        "src/llm/__init__.py", 
        "src/llm/central_negotiator.py",
        "src/llm/agent_validator.py",
        "src/navigation/__init__.py",
        "src/simulation/game_engine.py"
    ]
    
    all_good = True
    
    for pyfile in python_files:
        if os.path.exists(pyfile):
            try:
                with open(pyfile, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic checks
                if len(content) > 0:
                    compile(content, pyfile, 'exec')
                    print(f"✅ {pyfile} - syntax OK")
                else:
                    print(f"⚠️  {pyfile} - empty file")
                    
            except SyntaxError as e:
                print(f"❌ {pyfile} - syntax error: {e}")
                all_good = False
            except Exception as e:
                print(f"⚠️  {pyfile} - warning: {e}")
        else:
            print(f"❌ {pyfile} - file missing")
            all_good = False
    
    return all_good

def check_requirements():
    """Check requirements.txt content"""
    print("\n📦 Checking requirements...")
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            reqs = f.read().strip().split('\n')
        
        expected_packages = ['requests', 'python-dotenv', 'numpy', 'colorama']
        
        for pkg in expected_packages:
            found = any(pkg in req for req in reqs)
            if found:
                print(f"✅ {pkg}")
            else:
                print(f"❌ {pkg} - not in requirements.txt")
        
        return True
    else:
        print("❌ requirements.txt missing")
        return False

def check_env_example():
    """Check .env.example content"""
    print("\n🔧 Checking environment configuration...")
    
    if os.path.exists(".env.example"):
        with open(".env.example", 'r') as f:
            content = f.read()
        
        required_vars = [
            'OPENROUTER_API_KEY',
            'CENTRAL_LLM_MODEL', 
            'AGENT_LLM_MODEL',
            'MAP_WIDTH',
            'MAP_HEIGHT'
        ]
        
        all_good = True
        for var in required_vars:
            if var in content:
                print(f"✅ {var}")
            else:
                print(f"❌ {var} - missing from .env.example")
                all_good = False
        
        return all_good
    else:
        print("❌ .env.example missing")
        return False

def main():
    """Run quick structure test"""
    print("⚡ QUICK STRUCTURE TEST")
    print("=" * 40)
    print("This test verifies project structure without requiring external dependencies")
    print()
    
    tests = [
        ("File Structure", check_file_structure),
        ("Python Syntax", check_basic_syntax), 
        ("Requirements", check_requirements),
        ("Environment Config", check_env_example)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            print()
    
    print("=" * 40)
    print(f"⚡ STRUCTURE TEST RESULTS")
    print(f"✅ Passed: {passed}/{total}")
    print(f"📊 Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 Structure test passed!")
        print("💡 Next steps:")
        print("   1. Run 'python smoke_test.py' for full functionality test")
        print("   2. Install packages: pip install -r requirements.txt")
        print("   3. Copy .env.example to .env and add your API key")
        print("   4. Run 'python main.py' to start simulation")
    else:
        print(f"\n⚠️  Structure issues found. Please fix the missing files/directories.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
