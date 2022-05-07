'Predicting Customer Satisfaction through Machine Learning in Aviation Industry'

'Data Pre-Processing'

'Step 1: Importing Libraries'
library(dplyr)
library(DataExplorer)
library(mice)
library(VIM)
library(missForest)

'reading data'

df <- read.csv ('Full_DS.csv', header=T) 

View(df)
str(df)
summary(df)


#Mutate to convert blank values to na values
budf <-df

df <- mutate_all(df, na_if, "")

plot_missing(df)

'Factorising Character Variables'

df$Gender = factor(df$Gender)
df$Customer.Type = factor(df$Customer.Type)
df$Type.of.Travel = factor(df$Type.of.Travel)
df$Class = factor(df$Class)
df$satisfaction = factor(df$satisfaction)

str(df)

levels(df$Gender)
levels(df$Customer.Type)
levels(df$Type.of.Travel)
levels(df$Class)
levels(df$satisfaction)


# Imputing missing values using mice
imputed_df <- mice(df, m=3)
Final_imputed_df <- complete (imputed_df)

#Check the imputed DF for missing values
plot_missing(Final_imputed_df)

ddf <- Final_imputed_df
ddf <- ddf[-1]


#Dumifying Variables
library(fastDummies)

dummied <- dummy_cols(ddf, remove_first_dummy = TRUE)

data <- dummied[c(-2,-3,-5,-6,-24)]
str(data)

# Normalising the data

#define Min-Max normalization function
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#apply Min-Max normalization to all the columns
norm.data <- as.data.frame(lapply(data, min_max_norm))


##Balancing

satisfied <- which(norm.data$satisfaction_satisfied == 1)
dissatisfied <- which(norm.data$satisfaction == 0)

length(satisfied)
length(dissatisfied)

undersample <- sample(dissatisfied,length(satisfied))

undersampled <- norm.data[c(undersample,satisfied),]

library(ggplot2)

ggplot(undersampled) + 
  geom_bar(aes(x=satisfaction_satisfied,  alpha=0.5, fill='satisfaction'))

write.csv(undersampled,'finaldf.csv')

df <- undersampled


## Exporting Training and Testing sets
set.seed(16)

split = sample.split(df, SplitRatio = 0.1)
ndf = subset(df, split = TRUE)

split = sample.split(ndf, SplitRatio = 0.5)
ndf = subset(df, split = TRUE)

split = sample.split(ndf, SplitRatio = 0.7)
training_set = subset(ndf, split = TRUE)
test_set = subset(ndf, split = FALSE)

write.csv(training_set,'training.csv')
write.csv(test_set,'test.csv')


#Exploratory Data Anlaysis
library(ggplot2)

df <- undersampled

##Explatory plotting

ggplot(df) + 
  geom_bar(aes(x=satisfaction,  alpha=0.5, fill='satisfaction'))

plot_correlation(df,'continuous', cor_args = list("use" = "pairwise.complete.obs"))


# Checking for class balance

satisfied <- which(df$satisfaction == 'satisfied')
dissatisfied <- which(df$satisfaction == 'neutral or dissatisfied')

length(satisfied)
length(dissatisfied)

undersample <- sample(dissatisfied,length(satisfied))

undersampled <- df[c(undersample,satisfied),]

ggplot(undersampled) + 
  geom_bar(aes(x=satisfaction,  alpha=0.5, fill='satisfaction'))



##Loading naivebayes
library(naivebayes)

training_set <- training_set 


##converting Y value to factor
training_set$satisfaction_satisfied = factor(training_set$satisfaction)


##Modelling Naive Bayes with Kfold Cross Validation

install.packages('klaR')
library("klaR")

control <- trainControl(method="repeatedcv", number=10, repeats=5, search="random")

nbgrid <-  expand.grid(fL = c(0,0.5,1), 
                       usekernel = c(TRUE,FALSE),
                       adjust = c(0.5,1,2,2.5,3))

Naive.Flights = train(satisfaction_satisfied ~ ., 
                      data=training_set, 
                      method="nb", 
                      trControl = control,
                      tuneGrid=nbgrid)

Naive.Flights

plot(Naive.Flights)

saveRDS(Naive.Flights, "naivemodel.rds")



## Modelling for Decision Trees

library(caTools)
suppressMessages(library(rattle))
library(caret)

df <- training_set

df <- df[-1]


df$satisfaction_satisfied <- factor(df$satisfaction_satisfied)

control <- trainControl(method="repeatedcv", number=10, repeats=5, search="random")



flight.tree = train(satisfaction_satisfied ~ ., 
                    data=df, 
                    method="rpart", 
                    trControl = control,
                    tuneLength=10)

flight.tree

confusionMatrix(flight.tree)


suppressMessages(library(rattle))

##Visualisation

fancyRpartPlot(flight.tree$finalModel)

plot(flight.tree)

saveRDS(flight.tree, "dtmodel.rds")



## ANN

df <- training_set

library(caTools)
library (nnet)
library(ggplot2)

df <- df[c(-1)]

df$satisfaction_satisfied <- factor(df$satisfaction_satisfied)

str(df)

#####Modeling

control <- trainControl(method="repeatedcv", number=10, repeats=5, search="random")

ann <- train(satisfaction_satisfied ~., 
             data = df, 
             method = "nnet", 
             trControl = control)

ann
plot(ann)


##Further tuning
control <- trainControl(method="repeatedcv", number=10, repeats=5, search="random")

ann <- train(satisfaction_satisfied ~., 
             data = df, 
             method = "nnet", 
             trControl = control,
             tuneLength = 10)

ann
plot(ann)


saveRDS(ann, "annmodel.rds")


install.packages('NeuralNetTools')  
library(NeuralNetTools)  
plotnet(ann, alpha = 0.6)  


importance <- varImp(ann, scale=FALSE)

plot(importance)



## Evaluation

library(pROC)
library(yardstick)
library(caret)

df <- test_set

df <- df[-1]

##Loading the models

naivemodel <- readRDS("naivemodel.rds")
dtmodel <- readRDS("dtmodel.rds")
annmodel <- readRDS("annmodel.rds")


##Prediction Naive Bayes
nmprob = predict(naivemodel,newdata = df[,-24],type='prob')
nmclass = predict(naivemodel,newdata = df[,-24],type='raw')

##Prediction Decision Trees
dtprob = predict(dtmodel,newdata = df[,-24],type='prob')
dtclass = predict(dtmodel,newdata = df[,-24],type='raw')

##Prediction Decision Trees
annprob = predict(annmodel,newdata = df[,-24],type='prob')
annclass = predict(annmodel,newdata = df[,-24],type='raw')

df$satisfaction_satisfied <- factor(df$satisfaction_satisfied)

##Caret Confusion Matrix Evaluation NB
confusionMatrix(df$satisfaction_satisfied,nmclass)

##Caret Confusion Matrix Evaluation DT
confusionMatrix(df$satisfaction_satisfied,dtclass)

##Caret Confusion Matrix Evaluation ANN
confusionMatrix(df$satisfaction_satisfied,annclass)

accmetric <- data.frame(c('Naive Bayes','Decision Trees','ANN'),
                        c(0.8602,0.9117,0.9186),
                        c(0.9121,0.9024,0.9037),
                        c(0.8199,0.9214,0.9346))

names(accmetric) <- c('Model','Accuracy','Sensitivity','Specificity')

accmetric$Model <- factor(accmetric$Model)


##Yardstick F1 score (NBM)
nbeval = data.frame(df$satisfaction_satisfied)
nbeval$cl = nmclass
nbeval$pr = nmprob[,1]


f_meas(data = nbeval, estimate = nbeval$cl, truth = nbeval$df.satisfaction_satisfied)

##Yardstick F1 score (DTM)
dteval = data.frame(df$satisfaction_satisfied)
dteval$cl = dtclass
dteval$pr = dtprob[,1]


f_meas(data = dteval, estimate = dteval$cl, truth = dteval$df.satisfaction_satisfied)

##Yardstick F1 score (ANNM)
anneval = data.frame(df$satisfaction_satisfied)
anneval$cl = annclass
anneval$pr = annprob[,1]


f_meas(data = anneval, estimate = anneval$cl, truth = anneval$df.satisfaction_satisfied)




##ROC NM
plot1<- roc(df$satisfaction_satisfied, nbeval$pr, plot=TRUE, legacy.axes=TRUE,main="Naive Bayes ROC", percent=TRUE, 
            xlab = 'False Positive Percentage', ylab = 'True positive percentage', print.auc = TRUE,col = 'red')

plot2<-roc(df$satisfaction_satisfied, dteval$pr, plot=TRUE, legacy.axes=TRUE, percent=TRUE,main="Decision Tree ROC", 
           xlab = 'False Positive Percentage', ylab = 'True positive percentage', print.auc = TRUE,col='blue')

plot3<-roc(df$satisfaction_satisfied, anneval$pr, plot=TRUE, legacy.axes=TRUE, percent=TRUE, main="ANN ROC", 
           xlab = 'False Positive Percentage', ylab = 'True positive percentage', print.auc = TRUE,col='green')


roc1 <- plot.roc(df$satisfaction_satisfied, nbeval$pr, main="ROC comparison", percent=TRUE, col= "red",)
roc2 <- lines.roc(df$satisfaction_satisfied, dteval$pr, percent=TRUE, col="blue", add = TRUE)
roc3 <- lines.roc(df$satisfaction_satisfied, anneval$pr, percent=TRUE, col="green")

## Feature Importance
importance <- varImp(annmodel, scale=FALSE)
importance
plot(importance)
