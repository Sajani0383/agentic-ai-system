import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def get_llm_decision(state):
    prompt = f"""
    You are an intelligent parking system AI.

    Current parking state:
    {state}

    Tasks:
    1. Which area is most crowded?
    2. Where should vehicles be redirected?
    3. Give a short reasoning.

    Answer clearly.
    """

    response = llm.invoke(prompt)
    return response.content