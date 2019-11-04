library(ggplot2)
library(dplyr)
#install.packages("gdata") #incase gdata packages not available
library(gdata)  

mydata = read.csv("C:/Users/M. Saqib/Desktop/Courses/R/RR/data.csv")  # read csv file 

names(mydata)

lmTemp = lm(weight2~sleptim1, data = mydata) #Create the linear regression

plot(mydata, pch = 16, col = "blue") #Plot the results
abline(lmTemp, col = "red") #Add a regression line
summary(lmTemp)
