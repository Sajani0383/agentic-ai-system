import os
from google import genai
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

models_to_test = ['gemini-3.1-pro-preview', 'gemini-flash-latest', 'gemini-2.0-flash-lite']

for m in models_to_test:
    try:
        res = client.models.generate_content(model=m, contents='Say hello')
        print(f"{m} works:", res.text)
    except Exception as e:
        print(f"{m} failed:", e)
