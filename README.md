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
fraud <- read_csv("training.csv") %>% clean_names()
fraud_kaggle <- read_csv("holdout.csv") %>% clean_names()

fraud %>% skimr::skim_to_wide()
```
## Exploratory Analysis
```
fraud %>%
  count(event_label) %>%
  mutate(pct = n/sum(n)) -> fraud_rate

fraud_rate %>%
  ggplot(aes(x = event_label, y = pct)) +
  geom_col() +
  geom_text(aes(label = pct), color = "red") +
  labs(title = "Fraud Rate")
```
## Data Preparation
```
fraud <- fraud %>%
  mutate(event_label = as.factor(event_label)) %>%
  mutate(card_bin = as.factor(card_bin)) %>%
  mutate(billing_postal = as.factor(billing_postal)) %>%
  mutate_if(is.character, factor)

fraud_kaggle <- fraud_kaggle %>%
  mutate(card_bin = as.factor(card_bin)) %>%
  mutate(billing_postal = as.factor(billing_postal)) %>%
  mutate_if(is.character, factor)
```
## Data Partition
```
set.seed(123)

split <- initial_split(fraud, prop = 0.7)
train <- training(split)
test <- testing(split)

sprintf("Train PCT : %1.2f%%", nrow(train)/nrow(fraud) * 100)
sprintf("Test PCT  : %1.2f%%", nrow(test)/nrow(fraud) * 100)
```
![Picture1](https://github.com/dingy21/dingy21.github.io/assets/134649288/0a4739a2-96fb-4dff-88ac-621ed6ad6393)
