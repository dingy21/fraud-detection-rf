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
![Picture1](https://github.com/dingy21/dingy21.github.io/assets/134649288/0a4739a2-96fb-4dff-88ac-621ed6ad6393)
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
## Create Recipe
```
model_recipe <- recipe(event_label ~ billing_state + currency + cvv + transaction_type + transaction_env + account_age_days + 
                       transaction_amt + transaction_adj_amt, data = train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  themis::step_downsample(event_label, under_ratio = 3) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

bake(model_recipe %>% prep(), train %>% sample_n(1000))
```
## Model Workflow
### Random Forest 1
```
rf_model1 <- rand_forest(trees = 100, min_n = 20) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "permutation")

rf_workflow1 <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(rf_model1) %>%
  fit(train)
```
***random forest 2***
rf_model2 <- rand_forest(trees = 200, min_n = 10) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "permutation")

rf_workflow2 <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(rf_model2) %>%
  fit(train)

# logistic regression
log_model <- logistic_reg() %>%
  set_mode("classification") %>%
  set_engine("glm")

log_workflow <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(log_model) %>%
  fit(train)

log_workflow %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  mutate(across(where(is.numeric), round, 3))
```
