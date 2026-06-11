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
data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Age": [25, 30, 35, 28],
    "City": ["Edmonton", "Calgary", "Vancouver", "Toronto"],
    "Is_Manager": [False, True, True, False]
}
df = pd.DataFrame(data)


# define schema
# 1. Define a strict child object instead of a Dict entry
class CityClimateItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True) # Explicit constraint
    city_name: str
    climate_description: str

# 2. Main collection wrapper using a List shape
class CityClimatesContainer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    climates: list[CityClimateItem] # Safe structure (no hidden additionalProperties)


# create the prompt
base_string = "Given this list of cities, can you return a brief description of the climate of each city? "
data_string = ", ".join(df['City'].tolist())
print(data_string)

prompt_string = base_string + data_string
print(prompt_string)

# prompt model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt_string,
    config={
        "response_mime_type": "application/json",
        "response_schema": CityClimatesContainer, # Map class directly
    },
)

# parse the model's response
print(response.text)
# print(CityClimatesContainer.model_validate_json(response.text))