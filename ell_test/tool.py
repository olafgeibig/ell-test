import ell
import openai
import os
import json
import re
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

@ell.tool()
def get_html_content(
    url: str = Field(description="The URL to get the HTML content of. Never include the protocol (like http:// or https://)"),
):
    """Get the HTML content of a URL."""
    response = requests.get("https://" + url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()[:1000]

@ell.complex(model="gpt-4o-mini", tools=[get_html_content])
def get_website_content(website: str) -> str:
    """You are an agent that can summarize the contents of a website."""
    return f"Use the get_html_content too to fetch {website} and summarize it"

out = get_website_content("https://docs.crewai.com/getting-started/Installing-CrewAI/")
print(out.call_tools_and_collect_as_message())