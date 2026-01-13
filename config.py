import os
from google import genai

GEMINI_CLIENT = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

GEMINI_MODEL = "gemini-3-flash-preview"