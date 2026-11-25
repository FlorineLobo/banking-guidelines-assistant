
import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

if not api_key:
    raise ValueError("Gemini API key is missing from .env")


# Initialize Gemini client
client = genai.Client(api_key=api_key)


try:
    print("Connecting to Gemini using Interactions API...")
    print("Model:", model_name)

    # Create an interaction
    interaction = client.interactions.create(
        model=model_name,
        input="Reply with exactly: Gemini connection successful!"
    )

    # Read the response
    answer = interaction.output_text

    if answer:
        print("\nGemini response:")
        print(answer)
        print("\nAPI CONNECTION SUCCESSFUL!")

    else:
        print("Gemini returned an empty response.")

except Exception as error:

    print("\nAPI CONNECTION FAILED.")
    print("Error:", error)

finally:
    client.close()
