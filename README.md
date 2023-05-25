# Fraud Detection Project
Using a sample dataset of transactions and a holdout set, clean and prepare the data, build model(s) to detect and prevent potential fraudulent transactions. The target variable is called EVENT_LABEL and contains a label "legit" or "fraud". 

## Load Library
```
library(tidyverse)
library(tidymodels)
library(janitor)
library(skimr)
library(vip)
```
## Import Data
```
fraud <- read_csv("project_2_training.csv") %>% clean_names()
fraud_kaggle <- read_csv("project_2_holdout.csv") %>% clean_names()

fraud %>% skimr::skim_to_wide()
```
