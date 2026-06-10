"""
Helper functions for data processing and feature simplification.
As datasets come from multiple sources, there are often differences in how features are represented. 
These functions help to standardize and simplify the data for analysis and modeling.
"""

def neuter_status(sex):
    '''
    Input: A string representing the sex with neuter/spay status of an animal in the string ("Spayed Male")
    Output: A string categorizing the neuter/spay status of an animal.
    Categorizes animals by neuter/spay status. String is expected to contain "Neutered", "Spayed", "Intact", or be empty/unknown.
    '''
    if "Neutered" in str(sex) or "Spayed" in str(sex):
        return "Neutered/Spayed"
    elif "Intact" in str(sex):
        return "Not Neutered/Spayed"
    else:
        return "Unknown"
    
def simplify_animal_species(df):
    '''
    Input: DataFrame with an "animal_species" column.
    Output: DataFrame filtered to only include rows where "animal_species" is "dog" or "cat".
    Simplifies animal species to drop rows in the df to only include dogs and cats.
    Assumes column named "animal_species" exists in the dataframe, and values are in the format of "dog" or "cat" (case-sensitive).
    '''
    return df[df['animal_species'].isin(['dog', 'cat'])]

# simplifying intake type
def simplifying_intake_type(df):
    '''
    Input: Data frame 
    Output: Returns the Dataframe with simplied intake types according to the data specifications
    '''
    df.rename(columns={'Intake Type': 'intake_type'}, inplace=True)
    for index in df.index:
        type = df.at[index, "intake_type"]
        if type == "STRAY":
            df.at[index, "intake_type"] = "Stray"
        elif type in ["OWNER SURRENDER"]:
            df.at[index, "intake_type"] = "Owner Surrender"
        elif type in ["Euthenasia Required", "EUTHENASIA"]:
            df.at[index, "intake_type"] = "Euthanasia Request"
        else:
            df.at[index, "intake_type"] = "Other"

    return df

def split_date(df, col_name):
   '''
   Input:  Dataframe with datetime column.
           Name of datetime column.
   Output: Dataframe with additional "year", "month", "day" columns and removed original columns
   Splits intake date into 3 columns seperating the year month and day of the given date
   assumes that specified column name will exist in dataframe
   '''
   df['year'] = df[col_name].dt.year
   df['month'] = df[col_name].dt.month
   df['day'] = df[col_name].dt.day
   df.drop(columns= [col_name])
   return df

def categorize_color(color):
    '''
    Input: color - A string containing the original animal color value from the dataset
    Output: A simplified color category as a string.
    How to use it: df["color"] = df["color"].apply(categorize_color)
    '''
    color = str(color).upper()

    if "UNKNOWN" in color or "VARIOUS" in color:
        return "Multi/Unknown"

    if "POINT" in color or "LYNX" in color:
        return "Point/Lynx"

    if (
        "TABBY" in color or 
        "TORTIE" in color or 
        "TORBIE" in color or 
        "CALICO" in color or
        "TRICOLOR" in color or
        "TRICOLOUR" in color
    ):
        return "Tabby/Tortie"

    if (
        "GRAY" in color or 
        "GREY" in color or 
        "BLUE" in color or 
        "SILVER" in color or 
        "SMOKE" in color or 
        "LILAC" in color
    ):
        return "Gray/Blue"

    if (
        "ORANGE" in color or 
        "RED" in color or 
        "FLAME" in color or 
        "RUDDY" in color
    ):
        return "Orange/Red/Flame"

    if (
        "WHITE" in color or 
        "CREAM" in color or 
        "IVORY" in color or 
        "BUFF" in color or 
        "APRICOT" in color
    ):
        return "White"

    if "BLACK" in color or "BLK" in color:
        return "Black"

    return "Multi/Unknown"


def categorize_dog_breed(breed):
    '''
    Input: breed - A string containing the animal breed from the dataset
    Output: A simplified dog breed category as a string.
    '''
    breed = str(breed).upper()

    if "MIX" in breed or "UNKNOWN" in breed:
        return "Mix/Unknown"

    if (
        "GOLDEN RETRIEVER" in breed or
        "LABRADOR RETRIEVER" in breed or
        "COCKER SPANIEL" in breed or
        "FLAT COAT RETRIEVER" in breed or
        "WEIMARANER" in breed or
        "VIZSLA" in breed or
        "POINTER" in breed or
        "SETTER" in breed or
        "SPANIEL" in breed
    ):
        return "Sporting"

    if (
        "BEAGLE" in breed or
        "DACHSHUND" in breed or
        "BASSET" in breed or
        "GREYHOUND" in breed or
        "WHIPPET" in breed or
        "BLOODHOUND" in breed or
        "COONHOUND" in breed or
        "RIDGEBACK" in breed
    ):
        return "Hounds"

    if (
        "ROTTWEILER" in breed or
        "BOXER" in breed or
        "DOBERMAN" in breed or
        "MASTIFF" in breed or
        "ST BERNARD" in breed or
        "SAINT BERNARD" in breed or
        "HUSKY" in breed or
        "MALAMUTE" in breed or
        "BERNESE" in breed or
        "GREAT DANE" in breed or
        "NEWFOUNDLAND" in breed or
        "AKITA" in breed or
        "GREAT PYRENEES" in breed
    ):
        return "Working"

    if (
        "PIT BULL" in breed or
        "AM PIT BULL" in breed or
        "STAFFORDSHIRE" in breed or
        "BULL TERRIER" in breed or
        "FOX TERRIER" in breed or
        "PARSON RUSSELL" in breed or
        "JACK RUSSELL" in breed or
        "CAIRN TERRIER" in breed or
        "SCOTTISH TERRIER" in breed or
        "TERRIER" in breed
    ):
        return "Terriers"

    if (
        "CHIHUAHUA" in breed or
        "SHIH TZU" in breed or
        "MALTESE" in breed or
        "POMERANIAN" in breed or
        "PUG" in breed or
        "MIN PINSCHER" in breed or
        "MINIATURE PINSCHER" in breed or
        "CAVALIER" in breed or
        "YORKSHIRE TERRIER" in breed or
        "TOY POODLE" in breed
    ):
        return "Toys"

    if (
        "BULLDOG" in breed or
        "FRENCH BULLDOG" in breed or
        "STANDARD POODLE" in breed or
        "MINIATURE POODLE" in breed or
        "DALMATIAN" in breed or
        "BOSTON TERRIER" in breed or
        "BICHON" in breed or
        "CHOW CHOW" in breed or
        "SHARPEI" in breed or
        "SHAR-PEI" in breed or
        "SHIBA INU" in breed
    ):
        return "Non-Sporting"

    if (
        "GERMAN SHEPHERD" in breed or
        "BORDER COLLIE" in breed or
        "AUST CATTLE DOG" in breed or
        "AUSTRALIAN CATTLE DOG" in breed or
        "HEELER" in breed or
        "CORGI" in breed or
        "SHELTIE" in breed or
        "SHETLAND SHEEPDOG" in breed or
        "COLLIE" in breed or
        "BELGIAN MALINOIS" in breed or
        "AUSTRALIAN SHEPHERD" in breed
    ):
        return "Herding"

    return "Mix/Unknown"


def categorize_cat_breed(breed):
    '''
    Input: breed - A string containing the animal breed from the dataset
    Output: A simplified cat breed category as a string.
    '''
    breed = str(breed).upper()

    if (
        "DSH" in breed or
        "DOMESTIC SH" in breed or
        "AMERICAN SHORTHAIR" in breed or
        "DOMESTIC SHORT HAIR" in breed
    ):
        return "Domestic Shorthair"
    
    if (
        "DLH" in breed or 
        "DMH" in breed or 
        "DOMESTIC LH" in breed or
        "DOMESTIC MH" in breed or 
        "DOMESTIC LONGHAIR" in breed or 
        "DOMESTIC MEDIUM HAIR" in breed
    ):
        return "Domestic Long/Med Hair"
    
    if (
        "SIAMESE" in breed or 
        "HIMALAYAN" in breed or
        "PERSIAN" in breed or 
        "BRITISH SHORTHAIR" in breed or
        "MAINE COON" in breed or 
        "RAGDOLL" in breed or 
        "BURMESE" in breed or 
        "SPHYNX" in breed or
        "BENGAL" in breed or 
        "ABYSSINIAN" in breed
    ):
        return "Purebred Cat"
    
    return "Mix/Unknown"


def categorize_breed_by_species(row):
    '''
    Input: one row from the dataframe
    Output: simplified breed category based on whether the animal is a dog or cat
    How to use:  df["breed"] = df.apply(categorize_breed_by_species, axis=1)
    '''
    animal_type = str(row["animal_type"]).lower()

    if animal_type == "dog":
        return categorize_dog_breed(row["breed"])
    elif animal_type == "cat":
        return categorize_cat_breed(row["breed"])
    else:
        return "Mix/Unknown"