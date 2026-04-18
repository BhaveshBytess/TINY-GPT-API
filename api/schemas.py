from pydantic import BaseModel, Field


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    response: str


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,                          # ... means "required, no default"
        min_length=1,                 # Can't be empty string
        max_length=500,               # Don't let users send novels
        examples=["To be or not"]     # Shows up in /docs as example
    )
    max_tokens: int = Field(
        default=50,
        ge=1,                         # Greater than or equal to 1
        le=500,                       # Less than or equal to 500
    )
    temperature: float = Field(
        default=0.8,
        ge=0.01,                      # Can't be zero (division error)
        le=2.0,
    )


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    model_name: str = "tiny-gpt"