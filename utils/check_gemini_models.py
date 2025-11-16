import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the .env file")

genai.configure(api_key=api_key)

print("Available models and supported methods:")
for m in genai.list_models():
    # Filter to models that support generate_content and look image-related
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print(f"- {m.name}  methods={m.supported_generation_methods}")
