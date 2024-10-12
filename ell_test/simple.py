import ell
import openai
import os
import json
import re
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ell.configurator import config, register_provider

load_dotenv()
ell.init(store='./logdir', autocommit=True)
deepseek_client = openai.Client(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
or_client = openai.Client(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
or_model="google/gemini-flash-1.5"
config.register_model(or_model, or_client)
config.default_api_params["response_format"] = {'type': 'json_object'}

def remove_json_backticks(text):
    # Remove ```json from the beginning and ``` from the end
    pattern = r'^```json\s*(.*?)\s*```$'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    return text  # Return original text if pattern not found

class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: int = Field(description="The rating of the movie out of 10")
    summary: str = Field(description="A brief summary of the movie")

# @ell.complex(model="gpt-4o-mini", response_format=MovieReview)
# def generate_movie_review(movie: str) -> MovieReview:
#     """You are a movie review generator. Given the name of a movie, you need to return a structured review."""
#     return f"generate a review for the movie {movie}"

# @ell.complex(model="deepseek-chat", client=deepseek_client) #, response_format={'type': 'json_object'})
# def generate_movie_review_deep(movie: str):
#     """You are a movie review generator. Given the name of a movie, you need to return a structured review."""
#     # return f"generate a review for the movie {movie}"
#     return [
#         ell.system(f"""You are a movie review generator. Given the name of a movie, you need to return a structured review in JSON format.
#         You must absolutely respond in this format with no exceptions.
#         {MovieReview.model_json_schema()}"""),
#         ell.user("Review the movie: {movie}"),
# ]

@ell.complex(model=or_model, client=or_client)#, response_format=MovieReview)
def generate_movie_review_or(movie: str):
    """You are a movie review generator. Given the name of a movie, you need to return a structured review."""
    # return f"generate a review for the movie {movie}"
    return [
        ell.system(f"""You are a movie review generator. Given the name of a movie, you need to return a structured review in JSON format.
The JSON response must fulfill this schema with no exceptions.
{MovieReview.model_json_schema()}"""),
        ell.user(f"Review the movie: {movie}"),
    ]

# @ell.complex(model="openai/gpt-4o-mini", client=or_client, response_format=MovieReview)
# def generate_movie_review_or2(movie: str) -> MovieReview:
#     """You are a movie review generator. Given the name of a movie, you need to return a structured review."""
#     return f"generate a review for the movie {movie}"

print(f"Schema: {MovieReview.model_json_schema()}")
      
response = generate_movie_review_or("The Matrix")
print(f"Response: {response}")
cleaned_response = remove_json_backticks(response)
print(f"Response: {cleaned_response}")
print(MovieReview.model_validate_json(response))
# print(json.loads(response.text))
# print(json.loads(response.choices[0].message.content))

# from bs4 import BeautifulSoup
# import requests

# @ell.tool()
# def get_html_content(
#     url: str = Field(description="The URL to get the HTML content of. Never include the protocol (like http:// or https://)"),
# ):
#     """Get the HTML content of a URL."""
#     response = requests.get("https://" + url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     return soup.get_text()[:100]

# @ell.complex(model="gpt-4o-mini", tools=[get_html_content])
# def get_website_content(website: str) -> str:
#     """You are an agent that can summarize the contents of a website."""
#     return f"Tell me what's on {website}"

# out = get_website_content("https://docs.ell.so/index.html")
# print(out)