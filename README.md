# Use various machine learning methods to predict london housing price with existing features


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


```{r, load_libraries, include = FALSE}
library(rpart.plot)
library(caret)
library(tidyverse) # the usual stuff: dplyr, readr, and other goodies
library(lubridate)
library(janitor) # clean_names()
library(Hmisc)
library(pROC)
library(dplyr)
```

# Introduction 
This project aims to build an estimation engine to guide investment decisions in London house market. It will first build machine learning algorithms (and tune them) to estimate the house prices given variety of information about each property. 200 houses will then be chosen to invest in out of about 2000 houses on the market at the moment with the best tuned agorithm


# Load data
There are two sets of data, 

i) training data that has the actual prices 

ii) out of sample data that has the asking prices. Load both data sets. 


```{r read-investigate}
#read in the data

london_house_prices_2019_training<- read.csv("train.csv")
london_house_prices_2019_out_of_sample<-read.csv("test.csv")



#fix data types in both data sets

#fix dates
london_house_prices_2019_training <- london_house_prices_2019_training %>% mutate(date=as.Date(date))
london_house_prices_2019_out_of_sample<-london_house_prices_2019_out_of_sample %>% mutate(date=as.Date(date))
#change characters to factors
london_house_prices_2019_training <- london_house_prices_2019_training %>% mutate_if(is.character,as.factor)
london_house_prices_2019_out_of_sample<-london_house_prices_2019_out_of_sample %>% mutate_if(is.character,as.factor)

#str(london_house_prices_2019_training)
#str(london_house_prices_2019_out_of_sample)



```


```{r split the price data to training and testing}
#initial split
library(rsample)
train_test_split <- initial_split(london_house_prices_2019_training, prop = 0.75) #training set contains 75% of the data
# Create the training dataset
train_data <- training(train_test_split)
test_data <- testing(train_test_split)



```


# Visualize data 

Visualize and examine the data. 

```{r visualize}
#scatter plot: housing price and number of tube lines
ggplot(london_house_prices_2019_training, aes(x = num_tube_lines, y = price))+geom_point()


#scatter plot: housing price and total floor area
ggplot(london_house_prices_2019_training, aes(x = total_floor_area, y = price))+geom_point()

#scatter plot: housing price and current energy consumption
ggplot(london_house_prices_2019_training, aes(x = energy_consumption_current, y = price))+geom_point()


```

From the graph, we can observe that 

1.there exists inverse correlation between distance to tube station and housing price.

2. there exists strong linear correlation between total floor area and housing price for housing area under 250 square meter. The correlation appears to be weaker for housing over 250 square meters. 

3. there exists inverse correlation between current energy consumption and housing price.



```{r, correlation table, warning=FALSE, message=FALSE}

# produce a correlation table using GGally::ggcor()
# this takes a while to plot

library("GGally")
london_house_prices_2019_training %>% 
  select(-ID) %>% #keep Y variable last
  ggcorr(method = c("pairwise", "pearson"), layout.exp = 2,label_round=2, label = TRUE,label_size = 2,hjust = 1,nbreaks = 5,size = 2,angle = -20)

```

From the correlation table, we can conclude that average income, CO2 emission potential, CO2 emission potential, CO2 emission current, and total floor area are factors that affect housing price the most.


# Fit a linear regression model


```{r LR model}

#Define control variables
control <- trainControl (
    method="cv",
    number=5,
    verboseIter=TRUE) #by setting this to true the model will report its progress after each estimation

#we are going to train the model and report the results using k-fold cross validation
model1_lm<-train(
    price ~  average_income  * total_floor_area +
    number_habitable_rooms + 
    total_floor_area +
      energy_consumption_current +
      num_tube_lines+
    london_zone * co2_emissions_potential,
    train_data,
   method = "lm",
    trControl = control
   )

# summary of the results
summary(model1_lm) 


```


```{r}
#check variable importance
importance <- varImp(model1_lm, scale=TRUE)
plot(importance)


```

## Predict the values in testing and out of sample data

Use the predict function to test the performance of the model in testing data and summarize the performance of the linear regression model. 

```{r}
#predict the testing values

predictions <- predict(model1_lm,test_data)

lr_results<-data.frame(  RMSE = RMSE(predictions, test_data$price), 
                            Rsquare = R2(predictions, test_data$price))

                            
lr_results                         

#predict prices for out of sample data the same way
predictions_oos <- predict(model1_lm,london_house_prices_2019_out_of_sample)
```


To measure the quality of the prediction, I mainly consider RMSE and Rsquare value. The lower the EMSE value, the more desirable it is on the prediction and the higher the R squared, the more desirable it is. 

# Fit a tree model

Next I fit a tree model using the same subset of features. 


```{r tree model}

model2_tree <- train(
  price ~ average_income  * total_floor_area +
    number_habitable_rooms + 
   #energy_consumption_current +
    total_floor_area +
    num_tube_lines+
    london_zone * co2_emissions_potential,
  train_data,
  method = "rpart",
  trControl = control,
  tuneLength=20
    )

model2_tree$results

rpart.plot(model2_tree$finalModel)

#visualize the variable importance
importance <- varImp(model2_tree, scale=TRUE)
plot(importance)


#prediction
tree_preds_cart <- predict(model2_tree, test_data, type = "raw")

tree_rmse_cart <- RMSE(
   pred = tree_preds_cart,
   obs = test_data$price)

tree_rmse_cart
```
The tree model performs better as it has lower RMSE value, 239496.4, than the linear regression model 273552.8	


# Using KNN to predict prices
```{r}
knn_fit <- train(price~ average_income  * total_floor_area +
    number_habitable_rooms + 
       #energy_consumption_current +
    total_floor_area +
      num_tube_lines+
    london_zone * co2_emissions_potential, data=train_data, 
                 method = "knn",
                 trControl = trainControl("cv", number = 10), 
                 tuneLength = 40, 
                 preProcess = c("center", "scale"))  

knn_fit



plot(knn_fit)

#The final value used for the model was k = 9.
```
Using KNN to predict test data
```{r}
knn_price<-predict(knn_fit, newdata = test_data)


#compute the prediction RMSE
RMSE(knn_price, test_data$price)

#The RMSE for predicting the test data is 235385.9
```

# Use GBM model to predict housing price
```{r}
library(gbm)

my_control <-trainControl(method = "cv",     # Cross-validation
                  number = 5,      # 5 folds
                  )

grid<-expand.grid(interaction.depth = 6,n.trees = 100,shrinkage =0.075, n.minobsinnode = 10)
set.seed(100)

#Train for gbm

gbmFit1 <- train(price~average_income  * total_floor_area +
    number_habitable_rooms + 
    total_floor_area +
      num_tube_lines+
    london_zone * co2_emissions_potential, data=train_data,
                 method = "gbm", 
                 trControl = my_control,
                 tuneGrid =grid,
                 verbose = FALSE
                 )

print(gbmFit1)
price_GBM <-predict(gbmFit1,test_data)


#Compute the RMSE for the test data
RMSE(price_GBM, test_data$price)

#The RMSE is 230750.9
```



# Stacking

Use stacking to ensemble algorithms.

```{r,warning=FALSE,  message=FALSE }
library(caretEnsemble)

model_list <- caretList(
    price~average_income  * total_floor_area +
    number_habitable_rooms + 
    total_floor_area +
      num_tube_lines+
    london_zone * co2_emissions_potential,
    
    data =train_data,
    trControl=my_control,
    #metric = "ROC",
    methodList=c("glm"),
     tuneList=list(
            gbm=caretModelSpec(method="gbm"),
            ranger=caretModelSpec(method="ranger"),
            kknn=caretModelSpec(method="kknn"))
           )

model_list
```


```{r}
summary(model_list)

 modelCor(resamples(model_list))
 
   resamples <- resamples(model_list)
  dotplot(resamples, metric = "RMSE")
  
  
  glm_ensemble <- caretStack(
    model_list, 
    method="glm", 
    metric="RMSE", 
    trControl=my_control
  )
  
    summary(glm_ensemble)    
    
    
#prediction on the test data using ensembled method
    
price_glm_en<-predict(glm_ensemble,test_data)

RMSE(price_glm_en, test_data$price)
#The RMSE is 215269.8
```
The stacking method is the best algorithm as it has the lowest RMSE value. Therefore, I will use the stacking method to predict and identify the 200 properties


# Pick investments

Use the best algorithm to choose 200 properties from the out of sample data.

```{r,warning=FALSE,  message=FALSE }


numchoose=200

oos<-london_house_prices_2019_out_of_sample

#predict the value of houses
oos$predict <- predict(glm_ensemble,oos)
#Choose the ones you want to invest here
#Make sure you choose exactly 200 of them

oos$pred_profit<- (oos$predict - oos$asking_price)/oos$asking_price
  
#To maximize profit, I will select the top 200 properties with the highest predicted profit

top_200<- oos %>% arrange(desc(pred_profit)) %>% slice_max(pred_profit, n =200)

#output your choices. Change the name of the file to your "lastname_firstname.csv"
write.csv(top_200,"my_submission2.csv")

```
