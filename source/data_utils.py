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

