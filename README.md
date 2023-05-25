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
### Random Forest 2
```
rf_model2 <- rand_forest(trees = 200, min_n = 10) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "permutation")

rf_workflow2 <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(rf_model2) %>%
  fit(train)
```
### Logistic Regression
```
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
## Model Evaluation
### Random Forest 1
```
options(yardstick.event_first = FALSE)

rf_scored_train1 <- predict(rf_workflow1, train, type = "prob") %>%
  bind_cols(predict(rf_workflow1, train, type = "class")) %>%
  mutate(part = "train") %>%
  bind_cols(., train)

rf_scored_test1 <- predict(rf_workflow1, test, type = "prob") %>%
  bind_cols(predict(rf_workflow1, test, type = "class")) %>%
  mutate(part = "test") %>%
  bind_cols(., test)

bind_rows(rf_scored_train1, rf_scored_test1) %>%
  group_by(part) %>%
  metrics(event_label, .pred_fraud, estimate = .pred_class) %>%
  filter(.metric %in% c('accuracy', 'roc_auc', "mn_log_loss")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# precision 0.5
bind_rows(rf_scored_train1, rf_scored_test1) %>%
  group_by(part) %>%
  precision(event_label, .pred_class)

# recall 0.5
bind_rows(rf_scored_train1, rf_scored_test1) %>%
  group_by(part) %>%
  recall(event_label, .pred_class)

# spec
bind_rows(rf_scored_train1, rf_scored_test1) %>%
  group_by(part) %>%
  spec(event_label, .pred_class) %>% 
  mutate(fpr = 1 - .estimate)
```
### Random Forest 2
```
rf_scored_train2 <- predict(rf_workflow2, train, type = "prob") %>%
  bind_cols(predict(rf_workflow2, train, type = "class")) %>%
  mutate(part = "train") %>%
  bind_cols(., train)

rf_scored_test2 <- predict(rf_workflow2, test, type = "prob") %>%
  bind_cols(predict(rf_workflow2, test, type = "class")) %>%
  mutate(part = "test") %>%
  bind_cols(., test)

bind_rows(rf_scored_train2, rf_scored_test2) %>%
  group_by(part) %>%
  metrics(event_label, .pred_fraud, estimate = .pred_class) %>%
  filter(.metric %in% c('accuracy', 'roc_auc', "mn_log_loss")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

bind_rows(rf_scored_train2, rf_scored_test2) %>%
  group_by(part) %>%
  precision(event_label, .pred_class)

bind_rows(rf_scored_train2, rf_scored_test2) %>%
  group_by(part) %>%
  recall(event_label, .pred_class)

bind_rows(rf_scored_train2, rf_scored_test2) %>%
  group_by(part) %>%
  spec(event_label, .pred_class) %>% 
  mutate(fpr = 1 - .estimate)
```
### Logistic Regression
```
options(yardstick.event_first = TRUE)

log_scored_train <- predict(log_workflow, train, type = "prob") %>%
  bind_cols(predict(log_workflow, train, type = "class")) %>%
  mutate(part = "train") %>%
  bind_cols(., train)

log_scored_test <- predict(log_workflow, test, type = "prob") %>%
  bind_cols(predict(log_workflow, test, type = "class")) %>%
  mutate(part = "test") %>%
  bind_cols(., test)

bind_rows(log_scored_train, log_scored_test) %>%
  group_by(part) %>%
  metrics(event_label, .pred_fraud, estimate = .pred_class) %>%
  filter(.metric %in% c('accuracy', 'roc_auc', "mn_log_loss")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

bind_rows(log_scored_train, log_scored_test) %>%
  group_by(part) %>%
  precision(event_label, .pred_class)

bind_rows(log_scored_train, log_scored_test) %>%
  group_by(part) %>%
  recall(event_label, .pred_class)
```
### Random Forest ROC
```
bind_rows(rf_scored_train2, rf_scored_test2) %>%
  group_by(part) %>%
  roc_curve(event_label, .pred_fraud) %>%
  autoplot() +
  geom_vline(xintercept = 0.0037, # 5% TPR 
             color = "red",
             linetype = "longdash") +
  geom_vline(xintercept = 0.05,   # 5% FPR 
             color = "red",
             linetype = "longdash") +
  geom_vline(xintercept = 0.25,   # 25% FPR 
             color = "blue",
             linetype = "longdash") +
  geom_vline(xintercept = 0.75,   # 75% FPR 
             color = "green",
             linetype = "longdash") +
  labs(title = "RF ROC Curve", x = "FPR(1 - specificity)", y = "TPR(recall)")
```
### Histogram of Probability of Fraud
```
rf_scored_test2 %>%
  ggplot(aes(.pred_fraud, fill = event_label)) +
  geom_histogram(bins = 50) +
  geom_vline(xintercept = 0.5, color = "red") +
  labs(title = paste("Distribution of the Probabilty of FRAUD:", "RF Model"),
       x = ".pred_fraud",
       y = "count") 
```
### Decide Operational Range
```
operating_range <- rf_scored_test2 %>%
  roc_curve(event_label, .pred_fraud) %>%
  mutate(fpr = round((1 - specificity), 2),
         tpr = round(sensitivity, 3),
         score_threshold = round(.threshold, 3)) %>%
  group_by(fpr) %>%
  summarise(threshold = round(mean(score_threshold), 3),
            tpr = mean(tpr)) %>%
  filter(fpr <= 0.1)

operating_range
```
#### What are the metrics at 5% false positive rate?
```
rf_scored_test2 %>%
  mutate(fpr_5_pct = as.factor(if_else(.pred_fraud >= 0.362, "fraud", "legit"))) %>% 
  precision(event_label, fpr_5_pct)

rf_scored_test2 %>%
  mutate(fpr_5_pct = as.factor(if_else(.pred_fraud >= 0.362, "fraud", "legit"))) %>% 
  recall(event_label, fpr_5_pct)

rf_workflow2 %>%
  extract_fit_parsnip() %>%
  vip()
```
#### Function to find precision at threshold
```
precision_funk <- function(threshold){
  rf_scored_test2 %>%
  mutate(fpr_5_pct = as.factor(if_else(.pred_fraud >= threshold, "fraud", "legit"))) %>% 
  precision(event_label, fpr_5_pct) %>% print()
  
rf_scored_test2 %>%
  mutate(fpr_5_pct = as.factor(if_else(.pred_fraud >= threshold, "fraud", "legit"))) %>% 
  recall(event_label, fpr_5_pct) %>% print()
}
* precision at given threshold *
precision_funk(threshold = 0.247)
```
### Confusion Matrix
```
rf_scored_train2 %>%
  conf_mat(truth = event_label, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "Random Forest 2 Training Confusion Matrix")

rf_scored_test2 %>%
  conf_mat(truth = event_label, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "Random Forest 2 Testing Confusion Matrix")
```
