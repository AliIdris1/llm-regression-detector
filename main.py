import os
import json
import yaml
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- 1. DEFINE THE INTERFACE CONTRACTS ---

class SupportCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"

class ClassificationOutput(BaseModel):
    """The strict output schema expected from the LLM feature."""
    category: SupportCategory = Field(description="The core category of the support email.")
    summary: str = Field(description="A precise, one-sentence summary of the customer's issue.")

class FewShotExample(BaseModel):
    input: str
    output: dict

class PromptConfig(BaseModel):
    """The strict schema for versioned YAML prompt files."""
    version_id: str
    timestamp: str
    description: str
    system_prompt: str
    user_prompt_template: str
    few_shot_examples: Optional[List[FewShotExample]] = None


# --- 2. YAML LOADER WITH VALIDATION ---

def load_prompt_config(version_filename: str = "v1.yaml") -> PromptConfig:
    """Loads and validates a prompt configuration YAML file against the PromptConfig contract."""
    prompt_path = os.path.join("prompts", version_filename)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)
    
    # Pydantic validates the YAML structure automatically here
    return PromptConfig(**raw_data)


# --- 3. CORE LLM FEATURE FUNCTION ---

def classify_support_email(email_text: str, prompt_version: str = "v1.yaml") -> ClassificationOutput:
    """
    Classifies a customer support email using a versioned prompt and returns 
    a validated Pydantic ClassificationOutput object.
    """
    # Load and validate the prompt configuration
    config: PromptConfig = load_prompt_config(prompt_version)
    
    # Format the template with the email text
    formatted_prompt = config.user_prompt_template.format(email_text=email_text)
    
    client = Groq()
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": formatted_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    # Parse the raw LLM string into a dictionary
    raw_json = json.loads(response.choices[0].message.content)
    
    # Validate the LLM output against our Pydantic output contract
    validated_output = ClassificationOutput(**raw_json)
    return validated_output


# --- 4. EXECUTION ---

if __name__ == "__main__":
    test_email = "Hey, my payment went through twice for this month's subscription. Can I get a refund?"
    
    print(f"Running classification with prompt version: v1.yaml\n")
    result: ClassificationOutput = classify_support_email(test_email, prompt_version="v1.yaml")
    
    # Print fields safely because they are strongly typed objects
    print("Validated Output Contract:")
    print(f"Category: {result.category}")
    print(f"Summary:  {result.summary}")
    print(f"\nRaw JSON dictionary representation:")
    print(json.dumps(result.dict(), indent=2))