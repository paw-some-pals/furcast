#add season

from data_utils import get_seasons
import pandas as pd
import numpy as np

df_STcat_final= pd.read_csv('datasets/newcatshorttermABD.csv')
df_STdog_final= pd.read_csv('datasets/dogshorttermABD.csv')

def apply_get_season(df):
    df['season']=df['intake_month'].apply(get_seasons)
    return df

df_STcat_final= apply_get_season(df_STcat_final)

df_STdog_final= apply_get_season(df_STdog_final)

print(df_STcat_final[['intake_month', 'season']])

#add unemployment


#do breed mapping
#merge kaggle 