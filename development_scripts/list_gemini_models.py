
import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

try:
    print("\nAVAILABLE GEMINI MODELS")
    print("=" * 70)

    found_models = False

    for model in client.models.list():

        supported_actions = model.supported_actions or []

        if "generateContent" in supported_actions:

            found_models = True

            print("\nModel name:", model.name)
            print("Display name:", model.display_name)
            print("Supported actions:", supported_actions)

    if not found_models:
        print("No models supporting generateContent were found.")

except Exception as error:
    print("\nCould not retrieve model list.")
    print("Error:", error)

finally:
    client.close()
