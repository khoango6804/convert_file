import os
import json
import sys
import time
from google import genai


def load_api_keys():
    """Load all API keys from api.json file"""
    try:
        # Try to load from api.json file
        with open("api.json", "r") as f:
            config = json.load(f)

            api_keys = []
            if "api_keys" in config and isinstance(config["api_keys"], list):
                api_keys = config["api_keys"]
            elif "api_key" in config:
                # Single key as a list
                api_keys = [config["api_key"]]

            if not api_keys:
                print("No API keys found in api.json")
                return []

            return api_keys

    except FileNotFoundError:
        print("api.json file not found")
        return []
    except json.JSONDecodeError:
        print("api.json is not valid JSON")
        return []
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return []


def test_single_api_key(api_key, index, total):
    """Test a single API key"""
    try:
        print(f"\n--- Testing API Key {index + 1}/{total} ---")
        print(f"Key ID: {api_key[:8]}...")

        # Initialize the client with API key
        client = genai.Client(api_key=api_key)

        # Send a test prompt
        prompt = "Explain quantum computing in one sentence."

        # Generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt]
        )

        if hasattr(response, "text") and response.text:
            print("✅ API Connection Successful!")
            print(f"Response: {response.text.strip()[:100]}...")
            return True
        else:
            print("❌ No valid response received")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_all_api_keys():
    """Test all API keys and report results"""
    api_keys = load_api_keys()

    if not api_keys:
        print("Failed to load any API keys")
        return False

    print(f"Found {len(api_keys)} API keys to test")

    results = []
    for i, key in enumerate(api_keys):
        result = test_single_api_key(key, i, len(api_keys))
        results.append(result)

        # Add a small delay between tests to avoid rate limiting
        if i < len(api_keys) - 1:
            time.sleep(2)

    # Print summary
    print("\n======== TEST RESULTS ========")
    print(f"Total API keys tested: {len(api_keys)}")
    print(f"Successful: {sum(results)}/{len(api_keys)}")

    # Show individual results
    print("\nDetailed Results:")
    for i, success in enumerate(results):
        key_id = api_keys[i][:8] + "..." if api_keys[i] else "N/A"
        status = "✅ Working" if success else "❌ Failed"
        print(f"Key {i + 1}: {key_id} - {status}")

    # Return True only if all keys worked
    return all(results)


if __name__ == "__main__":
    print("===== GEMINI API KEY TEST =====")
    print(f"Testing all API keys at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    success = test_all_api_keys()

    if success:
        print("\n✅ All API keys are working!")
        sys.exit(0)
    else:
        print("\n⚠️ Some API keys failed the test.")
        sys.exit(1)
