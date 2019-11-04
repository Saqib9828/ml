install.packages("gdata")
install.packages("caret")
install.packages("e1071")
library(gdata) 
library('caret')
library('e1071')

mydata = read.csv("D:/class/M.Tech 1st/mL/lab/4 svm/R/heart.csv")
str(mydata)
head(mydata)

intrain <- createDataPartition(y=mydata$target,p=0.8,list=FALSE)
training <- mydata[intrain,]
testing <- mydata[-intrain,]

anyNA(training)
set.seed(192)
training[["target"]] = factor(training[["target"]])
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
head(training)
svm_linear <-train(target~., data = training, method = "svmLinear", 
                   trainControl = trctrl, preProcess = c("center", "scale"),
                   tuneLength = 10)
svm_linear

test_pred <- predict(svm_linear, newdata = testing)
test_pred

confusionMatrix(table(test_pred,testing$target))

plot()
