#intakes with outcome it has date of birth
import pandas as pd
import numpy as np
import matplotlib as plt
import data_utils
import re
df_edmonton=pd.read_csv('datasets/intakes_by_date_with_outcomes2.csv')

#add age_intake
df_edmonton['DATEOFBIRTH']=pd.to_datetime(df_edmonton['DATEOFBIRTH'])
df_edmonton['intake_date']=pd.to_datetime(df_edmonton['DATEBROUGHTIN'])
df_edmonton['age_intake'] = (df_edmonton['intake_date'] - df_edmonton['DATEOFBIRTH']).dt.days / 365.25

#add LOS
df_edmonton['OUTCOMEDATE'] = pd.to_datetime(df_edmonton['OUTCOMEDATE'])

df_edmonton['time_in_shelter'] = (df_edmonton['OUTCOMEDATE'] - df_edmonton['intake_date']).dt.days

#find the problematic values and print the corresponding rows; turn 0 days into one days
    #turn 0 days into 1 
df_edmonton['time_in_shelter']= df_edmonton['time_in_shelter'].replace(0, 1)

nan_days = df_edmonton[df_edmonton['time_in_shelter'].isna()]
#print(nan_days)
    #dropped the rows where time_in_shelter is Nan
df_edmonton = df_edmonton.dropna(subset=['time_in_shelter'])
#print(df_edmonton['time_in_shelter'].min())


#change intake_date to 3 separate year month day columns
df_edmonton["intake_year"] = df_edmonton["intake_date"].dt.year
df_edmonton["intake_month"] = df_edmonton["intake_date"].dt.month
df_edmonton["intake_day"] = df_edmonton["intake_date"].dt.day


#TODO before breed mapping, check the email from gears regarding affenpinscher breed comment







