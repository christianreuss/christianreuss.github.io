library(tidyverse)
library(caret)
library(mlbench)
library(doParallel)



training <- read.csv("C:/Users/I0271035/Desktop/coursera/practicalmachinelearning_JohnsHopkins/project/pml-training.csv",na.strings=c("","NA","#DIV/0!"))

testing <- read.csv("C:/Users/I0271035/Desktop/coursera/practicalmachinelearning_JohnsHopkins/project/pml-testing.csv",na.strings=c("","NA", "#DIV/0!"))

summary(training$classe)

training.sum <- training %>% 
  summarise_all(funs(sum(is.na(.))))


real.training <- training %>% 
  select_if(function(x) !any(is.na(x))) %>% 
  select(-contains("time")) %>% 
  select(-X, -user_name, -new_window, -num_window)

# test if uninformative predictors in dataset
t<-nearZeroVar(real.training)
tt <- nearZeroVar(training)

# Split Dataset

real.parts <- createDataPartition(training$classe, p=.75, list = FALSE)

real.training.part <- real.training[real.parts,]
real.testing.part <- real.training[-real.parts,]
dim(real.training.part)

# cross validation
# set.seed(11)
# 
# folds <- createFolds(y=real.training.part$classe, k=4)

# random forest
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

mdl <- train(classe ~., real.training.part, method="rf")

stopCluster(cl)


# gbm
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

mdl.gbm <- train(classe ~., real.training.part, method="gbm")

stopCluster(cl)

predict.rf <- predict(mdl, real.testing.part)
predict.gbm <- predict(mdl.gbm, real.testing.part)

confusionMatrix(predict.rf, real.testing.part$classe)
confusionMatrix(predict.gbm, real.testing.part$classe)

predict.quiz <- predict(mdl, testing)
