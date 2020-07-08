Summary
-------

In this excercise two models are built on fitness measurements. The data
is provided by this project
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
The best model is used to predict results from a small test dataset.

``` r
# load datasets
training <- read.csv("C:/Users/I0271035/Desktop/coursera/practicalmachinelearning_JohnsHopkins/project/pml-training.csv",na.strings=c("","NA","#DIV/0!"))

testing <- read.csv("C:/Users/I0271035/Desktop/coursera/practicalmachinelearning_JohnsHopkins/project/pml-testing.csv",na.strings=c("","NA", "#DIV/0!"))

summary(training$classe)
```

    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

Reducing Predictors
-------------------

Variables with mainly NA values and timestamp information are removed
from the training dataset. Second near zero variance predictors are
searched but the reduced dataset does not contain such values (see empty
summary).

``` r
real.training <- training %>% 
  select_if(function(x) !any(is.na(x))) %>% 
  select(-contains("time")) %>% 
  select(-X, -user_name, -new_window, -num_window)

# test if uninformative predictors in dataset
t<-nearZeroVar(real.training)
summary(t)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## 

Split Dataset
-------------

To estimtate the quality of the model an accuracy on test data should be
provided. So the dataset is split to train the model on training data
and validate it on test data.

``` r
real.parts <- createDataPartition(training$classe, p=.75, list = FALSE)

real.training.part <- real.training[real.parts,]
real.testing.part <- real.training[-real.parts,]
dim(real.training.part)
```

    ## [1] 14718    53

Build Models
------------

The random forest method and the gradient boosting machine (GBM) method
are trained to the data.

``` r
# random forest
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

mdl <- train(classe ~., real.training.part, method="rf")

stopCluster(cl)


# gbm
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

mdl.gbm <- train(classe ~., real.training.part, method="gbm")
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2301
    ##      2        1.4630             nan     0.1000    0.1713
    ##      3        1.3563             nan     0.1000    0.1281
    ##      4        1.2755             nan     0.1000    0.1065
    ##      5        1.2094             nan     0.1000    0.0781
    ##      6        1.1591             nan     0.1000    0.0784
    ##      7        1.1093             nan     0.1000    0.0696
    ##      8        1.0663             nan     0.1000    0.0539
    ##      9        1.0307             nan     0.1000    0.0582
    ##     10        0.9942             nan     0.1000    0.0514
    ##     20        0.7592             nan     0.1000    0.0206
    ##     40        0.5332             nan     0.1000    0.0132
    ##     60        0.4075             nan     0.1000    0.0070
    ##     80        0.3247             nan     0.1000    0.0039
    ##    100        0.2660             nan     0.1000    0.0034
    ##    120        0.2261             nan     0.1000    0.0029
    ##    140        0.1920             nan     0.1000    0.0022
    ##    150        0.1782             nan     0.1000    0.0018

``` r
stopCluster(cl)

predict.rf <- predict(mdl, real.testing.part)
predict.gbm <- predict(mdl.gbm, real.testing.part)

confusionMatrix(predict.rf, real.testing.part$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    4    0    0    0
    ##          B    0  945    8    0    0
    ##          C    0    0  846   19    0
    ##          D    0    0    1  783    3
    ##          E    0    0    0    2  898
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9925          
    ##                  95% CI : (0.9896, 0.9947)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9905          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9958   0.9895   0.9739   0.9967
    ## Specificity            0.9989   0.9980   0.9953   0.9990   0.9995
    ## Pos Pred Value         0.9971   0.9916   0.9780   0.9949   0.9978
    ## Neg Pred Value         1.0000   0.9990   0.9978   0.9949   0.9993
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1927   0.1725   0.1597   0.1831
    ## Detection Prevalence   0.2853   0.1943   0.1764   0.1605   0.1835
    ## Balanced Accuracy      0.9994   0.9969   0.9924   0.9865   0.9981

``` r
confusionMatrix(predict.gbm, real.testing.part$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1369   30    0    0    2
    ##          B   16  901   36    3    8
    ##          C    6   17  805   33    8
    ##          D    2    1   13  759   10
    ##          E    2    0    1    9  873
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9598         
    ##                  95% CI : (0.954, 0.9652)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9492         
    ##                                          
    ##  Mcnemar's Test P-Value : 6.727e-06      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9814   0.9494   0.9415   0.9440   0.9689
    ## Specificity            0.9909   0.9841   0.9842   0.9937   0.9970
    ## Pos Pred Value         0.9772   0.9346   0.9264   0.9669   0.9864
    ## Neg Pred Value         0.9926   0.9878   0.9876   0.9891   0.9930
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2792   0.1837   0.1642   0.1548   0.1780
    ## Detection Prevalence   0.2857   0.1966   0.1772   0.1601   0.1805
    ## Balanced Accuracy      0.9861   0.9667   0.9629   0.9688   0.9830

``` r
predict.quiz <- predict(mdl, testing)
```

As the Accuracy on the test dataset for random forest (0.9924551) is
larger than for gbm (0.9598287), random forest is used to predict on the
small test dataset.

``` r
predict.quiz
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

``` r
write.table(as.tibble(predict.quiz)$value, 'predict_quiz.csv', row.names=FALSE, col.names=FALSE, sep=",")
```

    ## Warning: `as.tibble()` is deprecated, use `as_tibble()` (but mind the new semantics).
    ## This warning is displayed once per session.
