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
intake_datetime - datetime - 56747 unique values '2017-12-07 00:00:00'
intake_month - int - [1, 12]
intake_number - float64 - 1.0 to 13.0	
time_in_shelter - str - 29319 unique values '0 days 14:07:00.000000000'

##### Removed columns
names - reason
"outcome_subtype" - Use information "foster", (added to outcome col)
"age_upon_outcome_(years)", "age_upon_outcome_age_group" - Redundant
"outcome_month", "outcome_year", "outcome_monthyear", "outcome_weekday", "outcome_hour", "outcome_number" - redundant
"count" (just a col of ones) - not needed
"age_upon_intake_(years)", "age_upon_intake_age_group" (binned data) - Redundant
"intake_year", "intake_monthyear", "intake_weekday", "intake_hour" - Redundant
"time_in_shelter_days" - redundant (have timedelta obj)
"dob_year", "dob_month", "dob_monthyear"  - Redundant



#### Adoptions by breed and date dataset
##### Column names
index	id	intakedate	intakereason	istransfer	sheltercode	identichipnumber	animalname	breedname	basecolour	speciesname	animalage	sexname	location	movementdate	movementtype	istrial	returndate	returnedreason	deceaseddate	deceasedreason	diedoffshelter	puttosleep	isdoa

##### Removed columns
'index','animalage','istransfer','sheltercode','identichipnumber','animalname','location','istrial','returndate','returnedreason','deceaseddate','deceasedreason','diedoffshelter','isdoa'

##### Added columns 



#### Long Beach Dataset 
Animal ID - “AXXXXXX” - 55344 str values - some duplicates 
Animal Type - [CAT, DOG, WILD, BIRD, REPTILE, RABBIT, OTHER, LIVESTOCK, AMPHIBIAN, GUINEA PIG] - 55344 str values  
Primary Colour - 86 unique - 55344 str values  
Sex - [Female, Male, Unknown, Spayed, Neutered] - 55344 str values  
Intake Date - “2017-06-21” - 55344 string values 
Intake Condition - 17 unique - 55344 string values 
Intake Type - 13 unique - 33344 str values 
Outcome Date - “2017-06-21” - 55344 string values
Outcome Type - 19 unique - 55344 string values 

##### Removed columns
Reason for intake -  because most of them i think were just defaulted to Null 
Outcome Subtype -  There are 295 different values so i think they might be too specific and some of them look like random codes 
Intake_is_dead -  all just alive at intake time
DOB - used to calculte the age at inatke but then removed after 


##### Added columns 
Age - calcuted using DOB and Intake Date


