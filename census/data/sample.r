

train <- read.csv("train.csv")
test <- read.csv("test.csv")

names(train)

zeros = data.frame(Mail_Return_Rate_CEN_2010 = rep(0, nrow(test)))
means = data.frame(Mail_Return_Rate_CEN_2010 = rep(mean(train$Mail_Return_Rate_CEN_2010), nrow(test)))
medians = data.frame(Mail_Return_Rate_CEN_2010 = rep(median(train$Mail_Return_Rate_CEN_2010), nrow(test)))


str(train)
summary(train)
library(randomForest)
indices = 1:170
#choose some columns to use:
indices_to_use = aaply(indices, 1, function(i) sum(is.na(train[,i])) == 0 & sum(is.na(test[,i])) == 0 & class(train[,i]) == "integer")
which(indices_to_use)

rf = randomForest(train[,which(indices_to_use)], y=train$Mail_Return_Rate_CEN_2010, ntree=50, do.trace=T, sampsize=5000)
rf_preds = predict(rf, test[,indices_to_use])


write.csv(zeros, file = "./submissions/zeros.csv", row.names = FALSE)
write.csv(means, file = "./submissions/means.csv", row.names = FALSE)
write.csv(medians, file = "./submissions/medians.csv", row.names = FALSE)
write.csv(rf_preds, file = "./submissions/rf.csv", row.names = FALSE)