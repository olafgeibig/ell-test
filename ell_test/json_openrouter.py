import ell
import openai
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import instructor

from typing import Any, Callable, Dict, Optional, Tuple, cast
from ell.provider import EllCallParams, Metadata, Provider
from ell.providers.openai import OpenAIProvider
from ell.types.message import ContentBlock, Message

load_dotenv()
ell.init(store='./logdir', autocommit=True)

# Initialize OpenAI client with OpenRouter's base URL and API key
openrouter_client = instructor.from_openai(openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
))

class InstructorProvider(OpenAIProvider):
    def translate_to_provider(self, *args, **kwargs):
        """ This translates ell call param,eters to the provider call parameters.  IN this case instructor is jsut an openai client. 
        so we can use the openai provider to do the translation. We just need to modify a few parameters because instructor doesn't support streaming."""
        api_params= super().translate_to_provider(*args, **kwargs)
        # Streaming is not allowed by instructor.
        api_params.pop("stream", None)
        api_params.pop("stream_options", None)
        return api_params
    
    def translate_from_provider(self,provider_response,
            ell_call : EllCallParams,
            provider_call_params : Dict[str, Any],
            origin_id : str, 
            logger : Optional[Callable] = None) -> Tuple[Message, Metadata]:
        """This translates the provider response (the result of calling client.chat.completions.create with the parameters from translate_to_provider)
          to an an ell message. In this case instructor just returns a pydantic type which we can use to create an ell response model. """
        instructor_response = cast(BaseModel, provider_response) # This just means that the type is a pydantic BaseModel. 
        if logger: logger(instructor_response.model_dump_json()) # Don't forget to log for verbose mode!
        return Message(role="assistant", content=ContentBlock(parsed=instructor_response)), {}

# We then register the provider with ell. We will use InstructorProvider any time an instructor.Instructor type client is used.
ell.register_provider(InstructorProvider(), instructor.Instructor)

# # OpenRouter-specific request parameters passed via `extra_body` (optional)
# # For detailed documentation, see "Using OpenAI SDK" at https://openrouter.ai/docs/provider-routing
# extra_body = {
#     "provider": {
#         "allow_fallbacks": True,
#         "data_collection": "deny",
#         "order": ["Hyperbolic", "Together"],
#         "ignore": ["Fireworks"],
#         "quantizations": ["bf16", "fp8"]
#     },
#     # Additional OpenRouter parameters can be added here, e.g.:
#     # "transforms": ["middle-out"]
# }

class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: int = Field(description="The rating of the movie out of 10")
    summary: str = Field(description="A brief summary of the movie")

@ell.complex(model="google/gemini-flash-1.5-exp", client=openrouter_client, response_format=MovieReview)#, extra_body=extra_body)
def generate_movie_review(movie: str) -> MovieReview:
    """You are a movie review generator. Given the name of a movie, you need to return a structured review."""
    return f"generate a review for the movie {movie}"

response = generate_movie_review("The Matrix")
print(response)