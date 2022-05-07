'Predicting Customer Satisfaction through Machine Learning in Aviation Industry'

'Data Pre-Processing'

'Step 1: Importing Libraries'
library(dplyr)
library(DataExplorer)
library(mice)
library(VIM)
library(missForest)

'reading data'

setwd('C:/Users/2607s/OneDrive/Desktop/Masters Data Science/Sem 1/AML/Assignment/Flight Satisfaction')

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

