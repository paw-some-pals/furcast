### Dataset Specifications

#### Austin Dataset
##### Column names
col name - type - range
age_upon_outcome - str - 46 unique values "10 years"
animal_id_outcome - str - 71961 unique values "A006100"
date_of_birth - str - 5923 unique values '2007-07-09 00:00:00'
outcome_subtype - str - ['Partner', 'Foster', 'Suffering', 'Medical', 'Behavior', 'In Kennel', 'Aggressive', 'Rabies Risk', 'In Foster', 'At Vet', 'Offsite', 'Snr', 'Possible Theft', 'SCRP', 'Court/Investigation', 'Enroute', 'In Surgery', 'Barn', 'Underage']
sex_upon_outcome - str - ['Neutered Male', 'Spayed Female', 'Intact Female', 'Intact Male', 'Unknown']	
age_upon_outcome_(days) - int64 - 0 to 9125
outcome_datetime - str - 65686 unique values '2017-12-07 14:07:00'
age_upon_intake - str - 46 unique values "10 years"
animal_id_intake - str - 71961 unique values 'A006100'
animal_type - str - ['Dog', 'Cat', 'Other', 'Bird']	
breed - str - 2155 unique values ['Spinone Italiano Mix', 'Dachshund', ...]
color - str - 529 unique values ['Yellow/White', 'Tricolor', 'Brown/White', ...]
found_location - str - 36576 unique values ['Colony Creek And Hunters Trace in Austin (TX)', '8700 Research Blvd in Austin (TX)', ...]
intake_condition - str - ['Normal', 'Injured', 'Aged', 'Sick', 'Other', 'Feral', 'Pregnant', 'Nursing']	
intake_type - str - ['Stray', 'Public Assist', 'Owner Surrender', 'Euthanasia Request', 'Wildlife']
sex_upon_intake - str - ['Neutered Male', 'Spayed Female', 'Intact Female', 'Intact Male', 'Unknown']
age_upon_intake_(days) - int64 - 0 to 9125
intake_datetime - str - 56747 unique values '2017-12-07 00:00:00'
intake_number - float64 - 1.0 to 13.0	
time_in_shelter - str - 29319 unique values '0 days 14:07:00.000000000'

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



#### Adoptions by breed and date dataset
##### Column names
col name - type - range

index             10290    int64  
id                10290    int64  
intakedate        10290    str    
intakereason      10288    str    
istransfer        10290    int64  
sheltercode       10290    str    
identichipnumber  8324     str    
animalname        10290    str    
breedname         10245    str    
basecolour        10290    str    
speciesname       10290    str    
animalage         10290    str    
sexname           10290    str    
location          10290    str    
movementdate      10290    str    
movementtype      10290    str    
istrial           10289    float64
returndate        3256     str    
returnedreason    10290    str    
deceaseddate      326      str    
deceasedreason    10290    str    
diedoffshelter    10290    int64  
puttosleep        10290    int64  
isdoa             10290    int64 


##### Removed columns
names - reason
'index'- using animal ids instead
'animalage'- changed to float, in new column 'animal_age_float'
'istransfer'- not useful
'sheltercode'- not useful
'identichipnumber'- redundant
'animalname'redunant
'location'-not useful
'istrial'- not useful
'returndate'- animals returned will have another row anyways
'returnedreason'- not useful
'deceaseddate'- not useful
'deceasedreason- not useful
'diedoffshelter'-not useful
'isdoa'- not useful

##### Added columns 
