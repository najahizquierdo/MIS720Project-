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

# splitting the data into 70/30 and 80/20 ratios

sample <- sample.split(data_skew3$Stay, SplitRatio = 0.75)
train_data  <- subset(data_skew3, sample == TRUE)
test_data <- subset(data_skew3, sample == FALSE)


#check distribution of test train split
prop.table(table(train_data$Stay)) * 100
prop.table(table(test_data$Stay)) * 100
prop.table(table(data_skew3$Stay)) * 100

#Preprocessing for Logistic Reg

# splitting the data to X and Y

training_dataX <- train_data[,names(train_data) != "Stay"]

dummy <- dummyVars( ~ ., data=training_dataX)

# Applying one hot encoding to data for logistic regression
dummy_train <- data.frame(predict(dummy, newdata = train_data))

# splitting the data to test and train for logistic regression
dummy_test <- data.frame(predict(dummy, newdata = test_data))
dummy_test <- na.omit(dummy_test)
dummy_train <- na.omit(dummy_train)

# eliminating the null values
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# converting the response to a factor of 0 and 1
train_data$Stay <- as.factor(as.numeric(train_data$Stay))


set.seed(400)

# Logistic Regression


logistic_model <- glm(train_data$Stay ~ ., 
                      data = dummy_train, 
                      family = "binomial")

# observing the best variables in the model
varImp(logistic_model)

par(pty= "s")

# roc curve of training data

p_train_log <- predict(logistic_model, newdata = dummy_train, type = "response")

# roc curve on testind data
roc(train_data$Stay, p_train_log, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive percentage", ylab = "True Positive Percentage",
    main= "ROC of Logistic Regression",
    col="#377eb8", lwd =1, print.auc = TRUE)

# testing the model performance using test data

p_test_log <- predict(logistic_model, newdata = dummy_test, type = "response")

# roc curve on testind data
plot.roc(test_data$Stay, p_test_log, percent = TRUE,lwd = 1,
         col="#4daf4a", print.auc.y = 45, print.auc = TRUE, add = TRUE)
legend("bottomright",c("Training", "Testing"), col = c("#377eb8","#4daf4a"), lwd =3)


#Decision Trees

ctrl_dtree <- trainControl(method="repeatedcv",repeats = 3)
dtree_model <- train(Stay ~., 
                     data = train_data, 
                     method = "rpart",
                     parms = list(split = "information"),
                     trControl=ctrl_dtree,
                     tuneLength = 10,
                     na.action = na.pass)

# observing the bset performing variables in the model
varImp(dtree_model)

print(dtree_model)
confusionMatrix(
  dtree_model,
  positive = NULL,
  prevalence = NULL,
  mode = "sens_spec"
)


# roc curve of training data

p_train_dt <- predict(dtree_model, train_data, type = "prob")

# roc curve on testind data
roc(train_data$Stay, p_train_dt[ ,2], plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive percentage", ylab = "True Positive Percentage",
    main= "ROC of Decision Tree",
    col="#377eb8", lwd =1, print.auc = TRUE)

# testing the model performance using test data

p_test_dt <- predict(dtree_model, test_data, type = "prob")

# roc curve on testind data
plot.roc(test_data$Stay, p_test_dt[ ,2], percent = TRUE,lwd = 1,
         col="#4daf4a", print.auc.y = 45, print.auc = TRUE, add = TRUE)
legend("bottomright",c("Training", "Testing"), col = c("#377eb8","#4daf4a"), lwd =3)


# KNN

ctrl_knn <- trainControl(method = "cv",
                         summaryFunction = defaultSummary,
                         number = 5)
set.seed(2)
knn <- train(Stay ~ ., 
             data = train_data,
             method = "knn",
             trControl=ctrl_knn,
             metric = "Accuracy",
             tuneLength = 2,
             na.action = na.pass
)


print(knn)
confusionMatrix(
  knn,
  positive = NULL,
  prevalence = NULL,
  mode = "sens_spec"
)


# roc curve o training data

p_train_knn <- predict(knn, train_data, type = "prob")

# roc curve on testing data
roc(train_data$Stay, p_train_knn[ ,2], plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive percentage", ylab = "True Positive Percentage",
    main= "ROC of KNN",
    col="#377eb8", lwd =1, print.auc = TRUE)

# model performance on testing data
p_test_knn <- predict(knn, test_data, type = "prob")

# roc curve on testind data
plot.roc(test_data$Stay, p_test_knn[ ,2], percent = TRUE,lwd = 1,
         col="#4daf4a", print.auc.y = 45, print.auc = TRUE, add = TRUE)
legend("bottomright",c("Training", "Testing"), col = c("#377eb8","#4daf4a"), lwd =3)


# naive bayes

naive_bayes <- train(Stay ~ ., 
                     data = train_data,
                     method = "naive_bayes",
                     usepoisson = TRUE,
                     na.action = na.pass)

# Viewing the model
naive_bayes

confusionMatrix(
  naive_bayes,
  positive = NULL,
  prevalence = NULL,
  mode = "sens_spec"
)

# roc curve o training data

p_train_nb <- predict(naive_bayes, train_data, type = "prob")

# roc curve on testind data
roc(train_data$Stay, p_train_nb[ ,2], plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive percentage", ylab = "True Positive Percentage",
    main= "ROC of Naive Bayes",
    col="#377eb8", lwd =1, print.auc = TRUE)

# model performance on testing data
p_test_nb <- predict(naive_bayes, test_data, type = "prob")

# roc curve on testind data
plot.roc(test_data$Stay, p_test_nb[ ,2], percent = TRUE,lwd = 1,
         col="#4daf4a", print.auc.y = 45, print.auc = TRUE, add = TRUE)
legend("bottomright",c("Training", "Testing"), col = c("#377eb8","#4daf4a"), lwd =3)
  