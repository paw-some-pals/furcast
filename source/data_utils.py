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


# Categorize color into broader color groups based on specs
def categorize_color(color):
    color = str(color).upper()

    # Multi / Unknown
    if "UNKNOWN" in color or "VARIOUS" in color:
        return "Multi/Unknown"

    # Point / Lynx
    if "POINT" in color or "LYNX" in color:
        return "Point/Lynx"

    # Tabby / Tortie
    if (
        "TABBY" in color or 
        "TORTIE" in color or 
        "TORBIE" in color or 
        "CALICO" in color or
        "TRICOLOR" in color or
        "TRICOLOUR" in color
    ):
        return "Tabby/Tortie"

    # Gray / Blue
    if (
        "GRAY" in color or 
        "GREY" in color or 
        "BLUE" in color or 
        "SILVER" in color or 
        "SMOKE" in color or 
        "LILAC" in color
    ):
        return "Gray/Blue"

    # Orange / Red / Flame
    if (
        "ORANGE" in color or 
        "RED" in color or 
        "FLAME" in color or 
        "RUDDY" in color
    ):
        return "Orange/Red/Flame"

    # White
    if (
        "WHITE" in color or 
        "CREAM" in color or 
        "IVORY" in color or 
        "BUFF" in color or 
        "APRICOT" in color
    ):
        return "White"

    # Black
    if "BLACK" in color or "BLK" in color:
        return "Black"

    # Anything else
    return "Multi/Unknown"

    #df["color"] = df["color"].apply(categorize_color)

def categorize_breed(breed):
        breed = str(breed).upper()

        # Mix / Unknown
        if "MIX" in breed or "UNKNOWN" in breed:
            return "Mix/Unknown"

        # Sporting
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

        # Hounds
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

        # Working
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

        # Terriers
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

        # Toys
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

        # Non-Sporting
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

        # Herding
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

        # Anything else
        return "Mix/Unknown"

    #df["breed"] = df["breed"].apply(categorize_breed)
