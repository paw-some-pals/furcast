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
days_in_shelter
animal_age_float
age_at_outcome

##### Interesting stats
average days in shelter: 37.91 days
minimum days in shelter: 0
maximum days in shelter: 2723
average time in shelter: 13.87
unique species names and counts:
Cat             5749
Dog             3900
House Rabbit     218
Rat              130
Guinea Pig        93
Bird              38
Livestock         18
Hamster           17
Hedgehog          15
Mouse             13
Gerbil            12
Ferret            11
Pig               11
Chicken           10
Snake              8
Wildlife           7
Lizard             7
Opossum            6
Fish               6
Goat               5
Sugar Glider       5
Turtle             4
Tortoise           2
Chinchilla         2
Squirrel           1
Tarantula          1
Raccoon            1

average time spent in shelter based on species:
Bird             5.078947
Cat             33.867629
Chicken         10.800000
Chinchilla      17.000000
Dog             37.882564
Ferret           3.363636
Fish             3.000000
Gerbil          21.333333
Goat             3.200000
Guinea Pig       9.107527
Hamster          4.058824
Hedgehog        13.466667
House Rabbit    50.344037
Livestock       15.833333
Lizard           4.428571
Mouse           46.000000
Opossum          0.000000
Pig             38.363636
Raccoon          0.000000
Rat              9.669231
Snake            5.500000
Squirrel         0.000000
Sugar Glider     2.200000
Tarantula        0.000000
Tortoise         0.500000
Turtle           1.500000
Wildlife         0.142857

unique cat colours and count:
Black                     917
Orange                    542
Grey                      507
Black and White           501
Black and Brown           480
Tortie                    335
Grey and White            274
Brown and Black           273
Orange and White          218
Calico                    189
Buff                      174
Brown, Black and White    166
Black and grey            161
Torbie                    144
Tabbico                    97
White and Black            84
Dilute tortoiseshell       83
Dilute calico              77
White                      57
Buff and white             51
White and Grey             38
Grey and black             34
Brown                      29
Flame Point                25
White and Orange           25
Grey Black and White       24
Silver                     22
Lynx point                 21
Brown and White            20
Chocolate Point            19
White and Brown            15
Lilac Point                15
White and Tabby            15
Seal Point                 14
Black, Brown and White      9
Cream                       8
Fawn                        7
Blue                        7
Blue Point                  6
Chocolate                   6
Tricolour                   6
Seal                        5
Tabby and White             5
Tan and Black               5
Various                     4
Tabby                       4
White and Tan               4
Siver and Black             4
Black and Tan               4
Black Tortie                3
Tortie and White            3
Tortie Point                3
Tan and Brown               3
Smoke                       2
Golden                      1
Lilac                       1
Brindle and Black           1
Cinnamon                    1
Tan and White               1

average stay based on cat colour:
Black                     35.899673
Black Tortie              26.333333
Black and Brown           30.529167
Black and Tan             56.250000
Black and White           33.471058
Black and grey            38.024845
Black, Brown and White    18.666667
Blue                       7.714286
Blue Point                40.000000
Brindle and Black          7.000000
Brown                     29.827586
Brown and Black           35.838828
Brown and White           28.100000
Brown, Black and White    26.801205
Buff                      31.298851
Buff and white            29.980392
Calico                    35.465608
Chocolate                  9.000000
Chocolate Point           41.789474
Cinnamon                   5.000000
Cream                     18.250000
Dilute calico             33.337662
Dilute tortoiseshell      67.554217
Fawn                      31.571429
Flame Point               23.960000
Golden                    11.000000
Grey                      32.285996
Grey Black and White      24.750000
Grey and White            27.667883
Grey and black            29.941176
Lilac                     77.000000
Lilac Point               22.800000
Lynx point                22.857143
Orange                    35.599631
Orange and White          28.931193
Seal                      44.600000
Seal Point                15.714286
Silver                    53.727273
Siver and Black           18.750000
Smoke                     48.000000
Tabbico                   23.371134
Tabby                      7.000000
Tabby and White            4.000000
Tan and Black             45.600000
Tan and Brown             35.000000
Tan and White              0.000000
Torbie                    37.465278
Tortie                    43.417910
Tortie Point              37.666667
Tortie and White           2.666667
Tricolour                 32.833333
Various                   56.500000
White                     27.561404
White and Black           41.202381
White and Brown           15.066667
White and Grey            22.789474
White and Orange          24.080000
White and Tabby           26.133333
White and Tan              4.000000

outcome types and counts
Adoption            5810
Foster              2509
Reclaimed           1423
Transfer             532
Released To Wild       7
Stolen                 6
Escaped                3
Euthanized           181

average time in shelter based on dog breed:
 TODO need to find a way to list all breeds in terminal

##### Statistics
want to run some kind of t-test but researching indicates i will need to select a proper statistical measure for this dataset bc its not normally distributed and we are comparing many means to eachother



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


#### Dallas Animal Shelter Dataset
##### Column Names 
Animal_Id	
Animal_Type	
Animal_Breed	
Kennel_Status	
Activity_Sequence	
Census_Tract	
Council_District	
Intake_Type	
Intake_Subtype	
Reason	
Intake_Date	
Intake_Time	
Intake_Condition	
Hold_Request	
Outcome_Type	
Outcome_Subtype	
Outcome_Date	
Outcome_Time	
Outcome_Condition	
Chip_Status	
Animal_Origin	
Month	
Year

##### Removed Columns 
Kennel_Status 
Activity_Sequence
Census_Tract
Council_District
Reason
Hold_Request
Outcome_Subtype
Outcome_Time
Chip_Status
Animal_Origin
Month
Year

##### Added Columns 
Shelter Time

##### Stats 
population of dogs - 115194
population of cats - 38961
population of wildlife - 4208
population of birds - 1439
population of livestock -108

maximum shelter time - 310 days
minimum shelter time - 0.0 days  
mean shelter time - 6.1597 days 
mode shelter time - 0.0
median shelter time - 3.0 

number of animals with 0 day stay - 20.713k
number of animals with 1 day stay - 8371
number of animals with 2 day stay - 5298
number of animals with 3 day stay - 3672
number of animals with 4 day stay - 7503
number of animals with 5 day stay - 4221

animal with longest stay - stray, dog, PitBull, 310 days

intakes on stray 
intakes on return 
intakes on 

