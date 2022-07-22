# reading data file
df= read.csv(file.choose(), header=T)

# data structure & summary
str(df)
summary(df)
dim(df)

# dropping ineffective columns
df= df[,-c(1:3)]

# checking for missing values
sum(is.na(df))

# checking for data imbalance
library(lessR)
PieChart(Exited, hole = 0, values= "%", data = df,
         fill = c("red","blue"), main = "Proportion of the classes
         in the data")
legend(x="topright", title = "Legend",
       legend = c("Not exited", "Exited"), fill = c("red","blue"),
       cex = 0.7)

df$Geography= as.factor(df$Geography)
df$Gender= as.factor(df$Gender)
df$Exited= as.factor(df$Exited)
df$Geography= as.numeric(df$Geography)
df$Gender= as.numeric(df$Gender)
df$Exited= as.numeric(df$Exited)
df$Exited[df$Exited==1]= 0
df$Exited[df$Exited==2]= 1

# dealing with data imbalance
library(smotefamily)
smote= SMOTE(df[,-11], df$Exited)
str(smote)
str(smote$data)
dim(smote$data)
df1= smote$data
colnames(df1)[11]= "Exited"
PieChart(Exited, hole = 0, values= "%", data = df1,
         fill = c("red","blue"), main = "Proportion of the classes 
         in the data after oversampling")
legend(x="topright", title = "Legend",
       legend = c("Not exited", "Exited"), fill = c("red","blue"),
       cex = 0.7)

df1$Exited= as.factor(df1$Exited)

# train-test split
set.seed(12)
ind= sample(2, nrow(df1), replace = T, prob = c(0.8,0.2))
train= df1[ind==1,]
test= df1[ind==2,]

# Classification Tree

library(tree)
library(rfUtilities)

clf= tree(Exited ~ .-Exited, data= train, split = "gini")
summary(clf)

# confusion matrix of train & test data
table(Predicted= predict(clf,train), Actual= train$Exited)
table(Predicted= predict(clf,test), Actual= test$Exited)

# pruning of classification tree
set.seed(123)
cv= cv.tree(clf, FUN = prune.misclass)
cv$size
cv$dev
cv$method

# plotting cv error wrt tree size & cost-complexity parameter
plot(cv$size, cv$dev, type = "l", xlab = "Terminal nodes",
     ylab = "Cross validation error rate")

# prunned tree for the 2 best cases
par(mfrow= c(1,1), mar= c(2,1,2,1))
prune= prune.misclass(clf, best = 23)
plot(prune, type = "uniform")
text(prune, pretty=0, cex= 0.55)

# performance of pruned tree on test data
p_tree1= predict(prune, test, type = "class")
accuracy(p_tree1, test$Exited)

# confusion matrix of after pruning
table(Predicted= predict(prune,train), Actual= train$Exited)
table(Predicted= p_tree1, Actual= test$Exited)

# Bagging & Random Forest

library(randomForest)
library(rfUtilities)

# Bagging

set.seed(1234)
bag= randomForest(Exited ~ ., data= train, mtry= 10, importance= T, )
bag
par(mfrow= c(1,1), mar= c(5,5,2,1))
plot(bag, main = "Bagging")
legend(x="topright", legend = c("Train data", "Test data"),
       fill = c("red","green"), cex = 0.7)

# prediction on test data
p_bag2= predict(bag, test)
accuracy(p_bag2, test$Exited)

# confusion matrix of train & test data
table(Predicted= predict(bag,train), Actual= train$Exited)
table(Predicted= p_bag2, Actual= test$Exited)

# variable importance
importance(bag)
varImpPlot(bag, main = "Bagging: Variable Importance",
           col= c("red", "blue", "green", "deepskyblue", "black", "skyblue"))

rf.partial.prob(bag, train, xname =  "Age", which.class = "1", main = " ")
rf.partial.prob(bag, train, xname =  "NumOfProducts", which.class = "1",
                main = " ")

# Random Forest for Classification

set.seed(122)
rf= randomForest(Exited ~ ., data= train, importance= T)
rf
plot(rf, main = "Random Forest for Classification")
legend(x="topright", legend = c("Train data", "Test data"),
       fill = c("red","green"), cex = 0.7)

# prediction on test data
p_rf2= predict(rf, test)
accuracy(p_rf2, test$Exited)

# confusion matrix of train & test data
table(Predicted= predict(rf,train), Actual= train$Exited)
table(Predicted= p_rf2, Actual= test$Exited)

# variable importance
importance(rf)
varImpPlot(rf, main = "Random Forest: Variable Importance",
           col= c("red", "blue", "green", "deepskyblue", "black", "skyblue"))

rf.partial.prob(rf, train, xname =  "Age", which.class = "1", main = " ")
rf.partial.prob(rf, train, xname =  "NumOfProducts", which.class = "1",
                main = " ")

# Survival Analysis

df= read.csv(file.choose(), header=T)

df= df[,-c(1:3)]
df$delta= (df$Exited==1)
df$HasCrCard= as.character(df$HasCrCard)
df$IsActiveMember= as.character(df$IsActiveMember)

set.seed(128)
ind= sample(2, nrow(df), replace = T, prob = c(0.8,0.2))
train1= df[ind==1,]
test1= df[ind==2,]

# Cox Model
library(survival)
cox= coxph(Surv(Tenure, delta) ~ .-Exited, data = train1)
summary(cox)
cox1= coxph(Surv(Tenure, delta) ~ CreditScore + Geography + Gender
            + Age + Balance + NumOfProducts + IsActiveMember,
            data = train1, x= T)
summary(cox1)  #considering only the significant predictors

# testing proportional hazards assumption
temp= cox.zph(cox1, transform = "km")
temp
plot(temp, xlab = "Time(years)", col = c("blue", "green"))

# estimating baseline cumulative hazard
L0= basehaz(cox1, centered = F)

train1$GeographyGermany= as.numeric(train1$Geography=="Germany")
train1$GeographySpain= as.numeric(train1$Geography=="Spain")
train1$Gender= as.numeric(as.factor(train1$Gender=="Male"))
train1$IsActiveMember= as.numeric(train1$IsActiveMember)

test1$GeographyGermany= as.numeric(test1$Geography=="Germany")
test1$GeographySpain= as.numeric(test1$Geography=="Spain")
test1$Gender= as.numeric((test1$Gender=="Male"))
test1$IsActiveMember= as.numeric(test1$IsActiveMember)

# estimating & plotting cumulative hazard & survival function of an individual
cols= c("CreditScore", "GeographyGermany", "GeographySpain", "Gender", 
        "Age", "Balance", "NumOfProducts", "IsActiveMember")
p1= sum(test1[cols][1,]*cox1$coefficients)
p0= L0$hazard*exp(p1)
exp(-p0)

par(mfrow= c(1,2), mar= c(5,5,1,1))
plot(p0 ~ L0$time, type= "s", col = "deepskyblue", 
     xlab = "Time [years]", ylim= c(0,1), ylab = "Cumulative hazard")
plot(exp(-p0) ~ L0$time, type = "s", col = "deepskyblue",
     ylim = c(0,1),xlab = "Time [years]",
     ylab = "Survival probability")

# estimating survival probabilities at different time points, for testing and
                                                              # training data
i=1
for(i in 1:nrow(test1))
{
  p1= sum(test1[cols][i,]*cox1$coefficients)
  p0= L0$hazard*exp(p1)
  test1$SurvProb0[i]= exp(-p0)[1]
  test1$SurvProb1[i]= exp(-p0)[2]
  test1$SurvProb2[i]= exp(-p0)[3]
  test1$SurvProb3[i]= exp(-p0)[4]
  test1$SurvProb4[i]= exp(-p0)[5]
  test1$SurvProb5[i]= exp(-p0)[6]
  test1$SurvProb6[i]= exp(-p0)[7]
  test1$SurvProb7[i]= exp(-p0)[8]
  test1$SurvProb8[i]= exp(-p0)[9]
  test1$SurvProb9[i]= exp(-p0)[10]
  test1$SurvProb10[i]= exp(-p0)[11]
}

# calculating brier score at different time points: test data
library(measures)

brier_test= c()
t=0
for(t in 0:10)
  brier_test[t+1]= Brier(test1[,t+13], as.factor(test1$Exited), 0, 1)
brier_test

j=1
for(j in 1:nrow(test1))
  test1$SurvProb[j]= test1[j,test1$Tenure[j] + 13]

i=1
for(i in 1:nrow(train1))
{
  p1= sum(train1[cols][i,]*cox1$coefficients)
  p0= L0$hazard*exp(p1)
  train1$SurvProb0[i]= exp(-p0)[1]
  train1$SurvProb1[i]= exp(-p0)[2]
  train1$SurvProb2[i]= exp(-p0)[3]
  train1$SurvProb3[i]= exp(-p0)[4]
  train1$SurvProb4[i]= exp(-p0)[5]
  train1$SurvProb5[i]= exp(-p0)[6]
  train1$SurvProb6[i]= exp(-p0)[7]
  train1$SurvProb7[i]= exp(-p0)[8]
  train1$SurvProb8[i]= exp(-p0)[9]
  train1$SurvProb9[i]= exp(-p0)[10]
  train1$SurvProb10[i]= exp(-p0)[11]
}

# calculating brier score at different time points: train data
brier_train= c()
t=0
for(t in 0:10)
  brier_train[t+1]= Brier(train1[,t+13], as.factor(train1$Exited), 0, 1)
brier_train

j=1
for(j in 1:nrow(train1))
  train1$SurvProb[j]= train1[j,train1$Tenure[j] + 13]

train2= train1[,-c(15:25)]
train2$SurvProb= as.numeric(train2$SurvProb)
test2= test1[,-c(15:25)]
test2$SurvProb= as.numeric(test2$SurvProb)

# calculating the integrated brier score
library(prodlim)
library(pec)

cols1= c("CreditScore", "Geography", "Gender", "Age", "Balance",
         "NumOfProducts", "IsActiveMember")
p_SurvProb= predictSurvProb(cox1, train1[cols1], c(0,1,2,3,4,5,6,7,8,9,10))
perror= pec(list("Cox"= p_SurvProb),
            Surv(Tenure, delta) ~ CreditScore + Geography + Gender
            + Age + Balance + NumOfProducts + IsActiveMember, 
            data= train1, cens.model = "cox")
perror$n.risk
ibs(perror)
par(mfrow= c(1,1))
plot(perror, xlab = "Time(years)") 

# Regression Tree for Survival Probability

library(tree)
reg= tree(SurvProb ~ .-Exited-delta-Geography, data= train2,
          split = "deviance")
summary(reg)

# plotting the regression tree
par(mfrow= c(1,1))
plot(reg, type = "uniform")
text(reg, pretty = 0, cex= 0.6)

# prediction on test data &  its accuracy
p_reg= predict(reg, test2)
plot(p_reg, test2$SurvProb, xlab = "Predicted Survival Probability",
     ylab = "Survival Probability", col=c("deepskyblue", "red"))
legend(x="topleft", legend = c("Predicted", "Actual"),
       fill = c("deepskyblue", "red"), cex = 0.7)
sqrt(mean(p_reg-test2$SurvProb)^2)

# survival probability of 20 individuals of test data
pred= predict(reg, test2[c(1:20),])
plot(pred, ylab = "Survival Probability", col= "blue", type = "l")

# Random Forest for Survival Probability
set.seed(129)
rfg= randomForest(SurvProb ~ .-Exited-delta-Geography, data= train2,
                  importance= T)
rfg
plot(rfg, main = "Random Forest for Survival Probability")

# prediction on train data
p_rfg1= predict(rfg, train2)
sqrt(mean(p_rfg1-train2$SurvProb)^2)

# prediction on test data
p_rfg2= predict(rfg, test2)
sqrt(mean(p_rfg2-test2$SurvProb)^2)

# variable importance
importance(rfg)
varImpPlot(rfg, main = "Random Forest for Survival Probability: Variable Importance",
           col= c("red", "blue", "green", "deepskyblue", "black", "skyblue"))

