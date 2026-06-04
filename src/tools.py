import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    return response.get("results", [])