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
