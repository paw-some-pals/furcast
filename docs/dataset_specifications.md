### Dataset Specifications

#### Austin Dataset
##### Column names
col name - type - range
age_upon_outcome - int - ...
animal_id_outcome - int - ...
date_of_birth	
outcome_type	
sex_upon_outcome	
age_upon_outcome_(days)	
outcome_datetime	
age_upon_intake	
animal_id_intake	
animal_type	breed	
color	
found_location	
intake_condition	
intake_type	
sex_upon_intake	
age_upon_intake_(days)	
intake_datetime	
intake_number	
time_in_shelter

##### Removed columns
names - reason
"outcome_subtype" - Use information "foster", (added to outcome col)
"age_upon_outcome_(years)", "age_upon_outcome_age_group" - Redundant
"outcome_month", "outcome_year", "outcome_monthyear", "outcome_weekday", "outcome_hour", "outcome_number" - redundant
"count" (just a col of ones) - not needed
"age_upon_intake_(years)", "age_upon_intake_age_group" (binned data) - Redundant
"intake_month", "intake_year", "intake_monthyear", "intake_weekday", "intake_hour" - Redundant
"time_in_shelter_days" - redundant (have timedelta obj)
"dob_year", "dob_month", "dob_monthyear"  - Redundant