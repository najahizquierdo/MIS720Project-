# import dataset
source("health_care_analytics.R")
# Loading package
library(dplyr)
library(ROCR)
library(caTools)
library(caret)
library(pROC)
library(e1071)
library(class)
library(tidyverse)
install.packages("gmodels")



# Splitting dataset
train_data <- createDataPartition(y = data_red3$Stay,p = 0.75,list = FALSE)
train_dataset <- data_red3[train_data,]
test_dataset <- data_red3[-train_data,]

#check distribution
prop.table(table(train_dataset$Stay)) * 100
prop.table(table(test_dataset$Stay)) * 100
prop.table(table(data_red3$Stay)) * 100


#Preprocessing for KNN and Logistic Reg
training_dataX <- train_dataset[,names(train_dataset) != "Stay"]
dummy <- dummyVars( ~ ., data=training_dataX)
dummy_train <- data.frame(predict(dummy, newdata = train_dataset))
dummy_test <- data.frame(predict(dummy, newdata = test_dataset))
dummy_test <- na.omit(dummy_test)
dummy_train <- na.omit(dummy_train)
train_dataset <- na.omit(train_dataset)


set.seed(400)

# Logistic Regression

train_dataset$StayNum <- as.factor(as.numeric(train_dataset$Stay))
logistic_model <- glm(train_dataset$StayNum ~ ., 
                      data = dummy_train, 
                      family = "binomial")
summary(logistic_model)
pdata <- predict(logistic_model, newdata = dummy_train, type = "response")
mean(pdata)
#plot(logistic_model, col="steelblue")


#Naive Bayes 
#takes long as well
naive_bayes <- train(Stay ~ ., 
                      data = train_dataset,
                      method = "naive_bayes",
                      usepoisson = TRUE,
                      na.action = na.pass)
naive_bayes
confusionMatrix(
  naive_bayes,
  positive = NULL,
  prevalence = NULL,
  mode = "sens_spec"
)
#.9878


#KNN - this takes AGES but it works, I swear
ctrl_knn <- trainControl(method = "cv",
                                 summaryFunction = defaultSummary,
                                 number = 10)
set.seed(2)
knn <- train(Stay ~ ., 
                data = train_dataset,
                method = "knn",
                trControl=ctrl_knn,
                metric = "Accuracy",
               tuneLength = 2,
               na.action = na.pass
             )
knn
confusionMatrix(
  knn,
  positive = NULL,
  prevalence = NULL,
  mode = "sens_spec"
)

#

#Decision Trees

ctrl_dtree <- trainControl(method="repeatedcv",repeats = 3)
dtree_model <- train(Stay ~., 
                data = train_dataset, 
                method = "rpart",
                parms = list(split = "information"),
                trControl=ctrl_dtree,
                tuneLength = 10,
                na.action = na.pass)
print(dtree_model)
confusionMatrix(
  dtree_model,
  positive = NULL,
  prevalence = NULL,
  mode = "sens_spec"
)
#.7923
plot(dtree_model, col="steelblue")

  
  
  