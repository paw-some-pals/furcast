import os
from dotenv import load_dotenv
from google import genai
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from typing import Dict

# universal API stuff
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

# toy data - replace with your data!

aac_df = pd.read_csv("datasets/acc_dog_cleaned.csv")
breed_df = pd.read_csv("datasets/dog_breeds.csv")


# define schema
# 1. Define a strict child object instead of a Dict entry
class BreedItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True) # Explicit constraint
    breed_1: str
    breed2: str
    mix: bool

# 2. Main collection wrapper using a List shape
class BreedContainer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    breed_classification: list[BreedItem] # Safe structure (no hidden additionalProperties)

breed_list = breed_df["Name"].tolist()
breed_string =  ", ".join(breed_list)

for index in aac_df.index:
    breed = aac_df.at[index,"breed"]
    base_string = f"Given {breed}, classify it into breed1,breed2 and a boolean value(0 for false/1 for true) for if its a mix or not. If there is no second breed, use None as second breed. Use this {breed_string} to choose the proper breed. If it is not in here, choose the closest match."

    prompt_string = base_string
    print(prompt_string)

    # prompt model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_string,
        config={
            "response_mime_type": "application/json",
            "response_schema": BreedContainer, # Map class directly
        },
    )
    if index == 10:
        print(BreedContainer.model_validate_json(response.text))
        print(response.text)

    # parse the model's response
    
