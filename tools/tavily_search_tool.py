from tavily import TavilyClient
from openai import OpenAI
import os

# os.environ["TAVILY_API_KEY"] = "dummy key"
# os.environ["OPENAI_API_KEY"] = "dummy key"

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TavilySearchTool:
    @staticmethod
    def determine_if_search_needed(question: str):
        """Determine if the question requires an internet search."""
        messages = [
            {"role": "system", "content": "You determine if a question needs real-time information from the internet. Respond with 'yes' or 'no' only."},
            {"role": "user", "content": f"Does this question require searching the internet for up-to-date information? Question: {question}"}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200
        )

        answer = response.choices[0].message.content.strip().lower().rstrip(".")
        return answer == "yes"

    @staticmethod
    def refine_search_term(question: str):
        """Refine the question into a better search term."""
        messages = [
            {"role": "system", "content": "Given a question, generate an effective search query (1-5 words if possible) that will help find the most relevant information."},
            {"role": "user", "content": f"Create a search query for: {question}"}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )

        search_term = response.choices[0].message.content.strip()
        return search_term
    
    @staticmethod
    def search_internet(query: str):
        """Search the internet using Tavily API."""
        search_results = tavily_client.search(query=query, search_depth="advanced", include_images=False)
        return search_results
    