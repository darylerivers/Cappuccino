#!/usr/bin/env python3
"""
Test GLM-4 VSCode Integration

This script verifies that GLM-4 is properly configured for Continue extension.
"""

import json
import requests
import sys

def test_ollama_api():
    """Test if Ollama API is accessible."""
    print("🔍 Testing Ollama API...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            glm_models = [m for m in models if 'glm' in m.get('name', '').lower()]

            if glm_models:
                print(f"✅ Ollama API is running")
                print(f"✅ GLM-4 found: {glm_models[0]['name']}")
                return True
            else:
                print("❌ GLM-4 not found in Ollama")
                print("   Run: ollama pull glm4")
                return False
        else:
            print(f"❌ Ollama API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama API")
        print("   Run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_continue_config():
    """Test if Continue is configured properly."""
    print("\n🔍 Testing Continue configuration...")
    try:
        import os
        config_path = os.path.expanduser("~/.continue/config.py")

        if not os.path.exists(config_path):
            print(f"❌ Continue config not found at {config_path}")
            return False

        with open(config_path, 'r') as f:
            config_content = f.read()

        if 'glm4' in config_content.lower():
            print("✅ Continue config includes GLM-4")
            return True
        else:
            print("⚠️  Continue config doesn't mention GLM-4")
            print(f"   Edit: {config_path}")
            return False

    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def test_vscode_extension():
    """Check if Continue extension is installed."""
    print("\n🔍 Testing VSCode Continue extension...")
    import subprocess
    try:
        result = subprocess.run(
            ['code', '--list-extensions'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if 'continue.continue' in result.stdout:
            print("✅ Continue extension is installed")
            return True
        else:
            print("❌ Continue extension not found")
            print("   Install: code --install-extension continue.continue")
            return False
    except Exception as e:
        print(f"❌ Error checking extensions: {e}")
        return False

def print_next_steps(all_passed):
    """Print next steps for user."""
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - GLM-4 is ready for VSCode!")
        print("\n📚 Quick Start:")
        print("   1. Open VSCode")
        print("   2. Press Ctrl+Shift+L to open Continue panel")
        print("   3. Click model dropdown → Select 'GLM-4.7-Flash'")
        print("   4. Start coding with AI assistance!")
        print("\n📖 Full Guide: ~/.continue/GLM_VSCODE_GUIDE.md")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\n🔧 Fix the issues above, then run this test again:")
        print(f"   python3 {__file__}")

def main():
    """Run all tests."""
    print("="*60)
    print("GLM-4 VSCode Integration Test")
    print("="*60)

    results = []
    results.append(test_ollama_api())
    results.append(test_continue_config())
    results.append(test_vscode_extension())

    all_passed = all(results)
    print_next_steps(all_passed)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
