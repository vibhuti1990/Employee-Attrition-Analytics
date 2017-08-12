#Load required libraries and suppress the generated startup messages.
suppressPackageStartupMessages(library(h2o, quietly = TRUE))
suppressPackageStartupMessages(library(ggplot2, quietly = TRUE))
suppressPackageStartupMessages(library(DMwR, quietly = TRUE))

#--------------
#Load the data
#--------------
setwd('V:/Summer Semester/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing')
ibm_attrition = read.csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
pwd <- getwd()
dir.create(paste0(pwd,"/output"))
View(ibm_attrition)
#-------------------------------------------------------------------------------

#-------------------
#Data Preprocessing
#-------------------
#By looking at the data , we assume that the below two columns will not help
#in prediction.So we'll remove it
ibm_attrition$EmployeeCount <- NULL
ibm_attrition$Over18 <- NULL
ibm_attrition$EmployeeNumber <- NULL
ibm_attrition$StandardHours <- NULL
#1. Checking for missing values
t(apply(is.na(ibm_attrition), 2, sum))
#2. Checking for null values
t(apply(training, 2, is.null))
#check col types
str(ibm_attrition)
#3. Encoding 
factor.col = c("Attrition", "BusinessTravel", "Department", "DistanceFromHome",
               "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
               "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
               "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance")

ibm_attrition[factor.col] = lapply(ibm_attrition[factor.col], factor)
#Run Logistic model ro check significance of variables
classifier_fit <- glm(Attrition ~.-1, family = binomial(link = 'logit'), data = ibm_attrition)
summary(classifier_fit)

#--------------------------------------------------------------------------------
#Improving the Model using selected Cols which are contributing to the prediction
#based on the logistic model ran above
#--------------------------------------------------------------------------------
selected_cols = c('ï..Age','Attrition','DistanceFromHome', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked', 'OverTime', 'RelationshipSatisfaction', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager')
data= ibm_attrition[,selected_cols]
#Run the logistic regression with improved data
classifier_fit <- glm(Attrition ~.-1, family = binomial(link = 'logit'), data = data)
summary(classifier_fit)
#accuracy.meas(test$Attrition, classifier_fit)
#The model gives AIC of 995.98 which is very high but before we finalize our
#model we need to check for the data imbalance first
#--------------------------------------------------------------------------------
#--------------------
#Splitting the Data
#--------------------
install.packages('caTools')
library(caTools)
split = sample.split(data,SplitRatio = 0.75)
training = subset(data,split==TRUE)
test = subset(data,split==FALSE)
#------------------------------------
#---------------------------
#Checking the data Imbalance
#---------------------------
table(training$Attrition)
#The current data is higly imbalanced.So we'll also check class distribution
prop.table(table(training$Attrition))
#By checking the class distribution we get to know that the % of positive 
#values is only 16% of the total data.So,this is a severly imbalanced data
#Conclusion: We need to handle this imbalanced data bucause due to data imbalance 
#the machine learning algorithm becomes bais
#------------------------------------------------------------------------------

-------------------------------------------

#-----------------------
#Handle imbalanced data
#-----------------------
install.packages("ROSE")
library(ROSE)

#Generating Synthetic Data using SMOTE
install.packages("DMwR")
library(DMwR)
# Calculate the number of observations in each class.
class_count = table(training$Attrition)
data.smote<- SMOTE(Attrition ~.,data=training,over=100,under=200)
table(data.smote$Attrition)

#Generate Synthetic Data using ROSE
data.rose <- ROSE(Attrition ~ ., data = training)$data
table(data.rose$Attrition)

#--------------------
#Splitting the Data
#--------------------
install.packages('caTools')
library(caTools)
split = sample.split(data.smote,SplitRatio = 0.80)
training = subset(data,split==TRUE)
test = subset(data,split==FALSE)
#-------------------------------------------------------------------------------


#______________
#Running Model
library(ROSE)
#__________________________
#Logistic regression model
#__________________________
classifier_new<- glm(Attrition ~.-1, family = binomial(link = 'logit'), data = training)
summary(classifier_new)

#Predict
prob_pred = predict(classifier_new, type = 'response', newdata = test)
Log_pred = ifelse(prob_pred > 0.5, 1, 0)
summary(prob_pred)

accuracy.meas(test$Attrition,Log_pred)

# Making the Confusion Matrix
table(test[,2], Log_pred)

#ROC Curve
roc.curve(test$Attrition, Log_pred,col='red')
#_____________________________________________________________________________

#_________________________
#Decision Tree Model
library(rpart)
#_________________________
tree.model <- rpart(Attrition ~ ., data = training)

#make prediction
prob_pred=predict(tree.model,newdata = test)
tree_pred=ifelse(prob_pred > 0.5, 1, 0)

#Check the accuracy
accuracy.meas(test$Attrition,tree_pred[,2])

#ROC Curve
roc.curve(test$Attrition,tree_pred[,2],col='green',add.roc = TRUE)

# Making the Confusion Matrix
table(test[,2], tree_pred[,2])
#____________________________________________________________________________

# Random forest
install.packages("randomForest")
library(randomForest)
fit.forest <- randomForest(Attrition ~., data = training)
rfpreds <- predict(fit.forest, test, type = "class")

roc.curve(test$Attrition,rfpreds,col='magenta',add.roc = TRUE)

#_____________________________
#stepwise Logistic Regression
#_____________________________
install.packages("leaps")
library(leaps)
colnames(data)
model=step(glm(Attrition~+ï..Age+Attrition+DistanceFromHome+EnvironmentSatisfaction+Gender+JobInvolvement+JobRole+JobSatisfaction+MaritalStatus+NumCompaniesWorked+OverTime+RelationshipSatisfaction+TotalWorkingYears+TrainingTimesLastYear+WorkLifeBalance+YearsAtCompany+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager
               ,data=training,family=binomial("logit")),direction="both")

#Predict
prob_pred = predict(classifier_new, type = 'response', newdata = test)
Log_pred = ifelse(prob_pred > 0.5, 1, 0)

roc.curve(test$Attrition,Log_pred)

