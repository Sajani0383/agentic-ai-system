import os
from google import genai
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

for m in client.models.list():
    print(m.name)
