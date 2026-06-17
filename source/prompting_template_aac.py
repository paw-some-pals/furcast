import os
from dotenv import load_dotenv
from google import genai
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from typing import Dict

'''
For heer's understanding 
we are getting the user to input all the basic inputs and then putting a prompt into gemini asking it to fill all te other values 
giving it age_intake,sex,spay_neuter,intake_month,intake_day,intake_year,animal_species,animal_size,colour,breed,intake_condition,intake_type
and then asking the api to give us 
is_mixed - boolean 1 or 0 
breed_1 - the main breed 
breed_2 - None if not mixed 
good_with_children - 0 to 5 with 5 being very good with children
good_with_other_dogs - 0 to 5 with 5 being very good with other dogs 
shedding - 0 to 5 with 5 being most shedding
grooming - 0 to 5 with 5 being needs the most grooming
drooling - 0 to 5 with 5 being most drooling
coat_length - 0 to 5 with 5 being most coat length 
good_with_strangers -  0 to 5 with 5 being very good with starngersv
playfulness -  0 to 5 with 5 being very playful 
protectiveness - 0 to 5 with 5 being very protective 
trainability -  0 to 5 with 5 being most trainable 
energy - 0 to 5 with 5 being most energy
barking - 0 to 5 with 5 being most barking 
'''




current_prompt = pd.read_csv("frontend/single_prediction_prompt.csv")


# universal API stuff
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
animal_type = current_prompt["animal_type"][0]
city = current_prompt["city"][0]

# toy data - replace with your data!




# define schema for cat
# 1. Define a strict child object instead of a Dict entry
# class CityClimateItem(BaseModel):
#     model_config = ConfigDict(populate_by_name=True) # Explicit constraint
#     city_name: str
#     climate_description: str

# 2. Main collection wrapper using a List shape
class DogTraitsContainer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    good_with_children: int = Field(ge=0, le=5)
    good_with_other_dogs: int = Field(ge=0, le=5)
    shedding: int = Field(ge=0, le=5)
    grooming: int = Field(ge=0, le=5)
    drooling: int = Field(ge=0, le=5)
    coat_length: int = Field(ge=0, le=2)
    good_with_strangers: int = Field(ge=0, le=5)
    playfulness: int = Field(ge=0, le=5)
    protectiveness: int = Field(ge=0, le=5)
    trainability: int = Field(ge=0, le=5)
    energy: int = Field(ge=0, le=5)
    barking: int = Field(ge=0, le=5)
    population: int
    unemploy_rate: float

    
    # Safe structure (no hidden additionalProperties)

# define schema for dog
# 1. Define a strict child object instead of a Dict entry
# class CityClimateItem(BaseModel):
#     model_config = ConfigDict(populate_by_name=True) # Explicit constraint
#     city_name: str
#     climate_description: str

# 2. Main collection wrapper using a List shape
class CatTraitsContainer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    min_life_expectancy: float = Field(ge=7, le=17)
    max_life_expectancy: float = Field(ge=10, le=20)
    min_weight: float = Field(ge=4,le=15)
    max_weight: float = Field(ge=7, le=30)
    family_friendly: int = Field(ge=3, le=5)
    shedding: int = Field(ge=1, le=5)
    general_health: int = Field(ge=1, le=5)
    playfulness: int = Field(ge=2, le=5)
    children_friendly: int = Field(ge=1, le=5)
    grooming: int = Field(ge=1, le=5)
    intelligence: int = Field(ge=3, le=5)
    other_pets_friendly: int = Field(ge=2, le=5)
    population: int
    unemploy_rate: float
     # Safe structure (no hidden additionalProperties)



if animal_type == "cat":
    #CAT
    # create the prompt for cat
    base_string = f"Given the cat breed {current_prompt["breed_1"][0]}, fill out values for min_life_expectancy,max_life_expectancy,min_weight,max_weight,family_friendly,shedding,general_health,playfulness,children_friendly,grooming,intelligence,other_pets_friendly. Given the city {city} fill out population,unemploy_rate."
    prompt_string = base_string
    print(prompt_string)

    # prompt model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_string,
        config={
            "response_mime_type": "application/json",
            "response_schema": CatTraitsContainer, # Map class directly
        },
    )

    # parse the model's response
    print(response.text)
    cat_dict = response.parsed.model_dump()
    cat_df = pd.DataFrame([cat_dict])
    current_prompt.update(cat_df)
    current_prompt.to_csv("frontend/single_prediction_prompt.csv", index=False)

else:

    #DOG
    # create the prompt for cat
    base_string = f"Given the dog breed {current_prompt["breed_1"][0]} fill out values for good_with_children,good_with_other_dogs,shedding,grooming,drooling,coat_length,good_with_strangers,playfulness,protectiveness,trainability,energy,barking. Given the city {city} fill out population,unemploy_rate."
    prompt_string = base_string
    print(prompt_string)

    # prompt model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_string,
        config={
            "response_mime_type": "application/json",
            "response_schema": DogTraitsContainer, # Map class directly
        },
    )

    # parse the model's response
    print(response.text)
    dog_dict = response.parsed.model_dump()
    dog_df = pd.DataFrame([dog_dict])
    current_prompt.update(dog_df)
    current_prompt.to_csv("frontend/single_prediction_prompt.csv", index=False)




    
