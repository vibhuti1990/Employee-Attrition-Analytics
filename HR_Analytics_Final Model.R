#Load required libraries and suppress the generated startup messages.
suppressPackageStartupMessages(library(h2o, quietly = TRUE))
suppressPackageStartupMessages(library(ggplot2, quietly = TRUE))
suppressPackageStartupMessages(library(DMwR, quietly = TRUE))
suppressMessages(library(grid))
suppressMessages(library(gridExtra))
suppressMessages(library(pROC))
suppressMessages(library(caTools))
suppressMessages(library(ROSE))
#--------------
#Load the data
#--------------
setwd('V:/Summer Semester/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing')
ibm_attrition = read.csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
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
factor.col =c("Education", "EnvironmentSatisfaction", "Gender",
              "JobInvolvement", "JobLevel", "JobSatisfaction", "MaritalStatus",
              "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance")
ibm_attrition[factor.col] = lapply(ibm_attrition[factor.col], factor)
  
#_____________
#PCA Analysis
#_____________
num_var=sapply(ibm_attrition,class)
num_var1=names(num_var[num_var=="integer" | num_var=="numeric"])
#Including only continous variables for PCA analyis
PCA_data<- ibm_attrition[num_var1]
colnames(PCA_data)
#save this data with only continous variable
write.csv(PCA_data,"Emp_data.csv")
prin_comp=prcomp(PCA_data,scale = TRUE,center = T)

#plot the resultant principal components
Eigenvector=prin_comp$rotation
write.csv(Eigenvector,"Eigenvector.csv")
biplot(prin_comp,scale=0,arrow.len=0.1,main = "Biplot of PCA")

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev
#compute variance
pr_var <- std_dev^2
#proportion of variance explained
prop_varex <- (pr_var/sum(pr_var))*100

#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")

#Remove unwanted PCA components which explain least variance
Prim_comp=subset(prin_comp$x,select=-c(PC10,PC11,PC12,PC13,PC14))
write.csv(Prim_comp,"Principal_Component.csv")
write.csv(cor(Prim_comp),"CorrelationAmongVar.csv")
#Group together the factor columns
fcol=c("Attrition","BusinessTravel","Department","Education","EducationField",
       "EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel",
       "JobRole","JobSatisfaction","MaritalStatus","OverTime","PerformanceRating",
       "RelationshipSatisfaction","StockOptionLevel","WorkLifeBalance")
factor.data <- ibm_attrition[fcol]

#Final data with PCA 
data<-cbind(Prim_comp,factor.data)

##--------------------
#Splitting the Data catools
#--------------------
split = sample.split(data,SplitRatio = 0.8)
training = subset(data,split==TRUE)
test = subset(data,split==FALSE)
#-------------------------------------------------------------------------------

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

#-----------------------
#Handle imbalanced data
#-----------------------
training<- SMOTE(Attrition ~.,data=training,over=100,under=200)
table(training$Attrition)

#______________
#Running Model
#__________________________
#Logistic regression model
#__________________________
classifier_new<- glm(Attrition ~.-1, family = binomial(link = 'logit'), data = training)
summary(classifier_new)


#Predict on training
prob_pred = predict(classifier_new, type = 'response', newdata = training)
Log_pred = ifelse(prob_pred > 0.5, 1, 0)
summary(prob_pred)

# Making the Confusion Matrix
table(training[,2], prob_pred)

#check the accuracy
accuracy.meas(training$Attrition,Log_pred)
#precision: 0.823
#recall: 0.795
#F: 0.404

#ROC Curve
roc.curve(training$Attrition, Log_pred,plotit = TRUE,n.thresholds = 100)
#Area under curve for trainig is 83.3%
#_____________________________________________________________________________

#_____________________________
#stepwise Logistic Regression
#_____________________________
install.packages("leaps")
library(leaps)
colnames(data)
model=step(glm(Attrition~+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+BusinessTravel
               +Department+Education+EducationField+EnvironmentSatisfaction+Gender
               +JobInvolvement+JobLevel+JobRole+JobSatisfaction+MaritalStatus+RelationshipSatisfaction
               +MaritalStatus+OverTime+PerformanceRating+RelationshipSatisfaction+StockOptionLevel+
                 WorkLifeBalance
               ,data=training,family=binomial("logit")),direction="both")


#Predict
prob_pred = predict(model, type = 'response', newdata = training)
SLog_pred = ifelse(prob_pred > 0.5, 1, 0)

#Check the accuracy
accuracy.meas(training$Attrition,SLog_pred)
#precision: 0.817
#recall: 0.784
#F: 0.400

#ROC Curve
roc.curve(training$Attrition,SLog_pred,add.roc = TRUE ,col="red")
#Area under the curve us 82.6


#_________________________
#Decision Tree Model
library(rpart)
#_________________________
tree.model <- rpart(Attrition ~ ., data = training)
tree.model$variable.importance
#make prediction
prob_pred=predict(tree.model,newdata = training)
tree_pred=ifelse(prob_pred > 0.5, 1, 0)

#Check the accuracy
accuracy.meas(training$Attrition,tree_pred[,2])
#precision: 0.822
#recall: 0.687
#F: 0.374

#ROC Curve
roc.curve(training$Attrition,tree_pred[,2],add.roc = T,col='Green')
#Area under the curve is 78.8

# Making the Confusion Matrix
table(test[,2], tree_pred[,2])
#____________________________________________________________________________

# Random forest
install.packages("randomForest")
library(randomForest)
#Model
fit.forest <- randomForest(Attrition ~., data = training)
#Predict
rfpreds <- predict(fit.forest, training, type = "class")

#Check the accuracy
accuracy.meas(training$Attrition,rfpreds)
#precision: 0.429
#recall: 1.000
#F: 0.300

roc.curve(training$Attrition,rfpreds,col='magenta',add.roc = TRUE)
#Are under the curve is 100%

#____________________________________________________________________________
#Running the models on test data
#____________________________________________________________________________
#Logistic regression
prob_pred = predict(classifier_new, type = 'response', newdata = test)
Log_pred = ifelse(prob_pred > 0.5, 1, 0)

#check the accuracy
accuracy.meas(test$Attrition,Log_pred)

#ROC Curve
roc.curve(test$Attrition, Log_pred,plotit = TRUE,n.thresholds = 100)
#-----------------------------------------------------------------------------
#Stepwise Logistic Regression
#Predict
prob_pred = predict(model, type = 'response', newdata = test)
SLog_pred = ifelse(prob_pred > 0.5, 1, 0)

#Check the accuracy
accuracy.meas(test$Attrition,SLog_pred)


#ROC Curve
roc.curve(test$Attrition,SLog_pred,add.roc = TRUE ,col="red")
#-----------------------------------------------------------------------------
#Decision Tree
#make prediction
prob_pred=predict(tree.model,newdata = test)
tree_pred=ifelse(prob_pred > 0.5, 1, 0)

#Check the accuracy
accuracy.meas(test$Attrition,tree_pred[,2])

#ROC Curve
roc.curve(test$Attrition,tree_pred[,2],add.roc = T,col='Green')
#------------------------------------------------------------------------------
#Random Forest
#Predict
rfpreds <- predict(fit.forest, test, type = "class")

#Check the accuracy
accuracy.meas(test$Attrition,rfpreds)

#ROC
roc.curve(test$Attrition,rfpreds,col='magenta',add.roc = TRUE)


#Spetwise Logistic Regression Performs the Best with Accuracy of 84.40 on test data 








