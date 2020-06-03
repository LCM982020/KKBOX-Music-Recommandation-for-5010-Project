# title: 5010_04264226 Project
# author: ID: 04264226'
# date: "6/3/2020"

# first load up the data
train <- read.csv(file = 'c:/Users/Desktop/MFIT5010/Project/train.csv')
test <- read.csv(file = 'c:/Users/Desktop/MFIT5010/Project/test.csv')
songs <- read.csv(file = 'c:/Users/Desktop/MFIT5010/Project/songs.csv')
members <- read.csv(file = 'c:/Users/Desktop/MFIT5010/Project/members.csv')

# make the response into factor
train$target <- as.factor(train$target)

# label encoding as XGB cannot handle non numeric feactures
songs$genre_ids<- as.numeric(songs$genre_ids)
songs$language<- as.numeric(songs$language)
members$city<- as.numeric(members$city)
members$gender<- as.numeric(members$gender)

# merge the training set with songs and member into one table
train <- merge(train, songs, by="song_id", all.x=TRUE)
train <- merge(train, members, by="msno", all.x=TRUE)
test <- merge(test, songs, by="song_id", all.x=TRUE)
test <- merge(test, members, by="msno", all.x=TRUE)

# merge song, member and source tab of both train and text set to assign lable encoding
merged <- rbind(train[,c("msno", "song_id", "source_system_tab")], test[,c("msno", "song_id", "source_system_tab")])
merged <- cbind(merged, merged)
names(merged) <-c("msno", "song_id", "source_system_tab","msno1", "song_id1", "source_system_tab1")
merged$source_system_tab1<- as.numeric(merged$source_system_tab1)
merged$msno1<- as.numeric(merged$msno1)
merged$song_id1<- as.numeric(merged$song_id1)

# now merged contains the encoded lable, add the encoded lable to test and training set
library(data.table)
merged1<- merged[,c("msno", "msno1")]
merged1 <- unique(merged1, by="msno1")
train <- merge(train, merged1, by="msno", all.x=TRUE)
test <- merge(test, merged1, by="msno", all.x=TRUE)

merged2<- merged[,c("song_id", "song_id1")]
merged2 <- unique(merged2, by="song_id1")
# need to split into 4 times due to lack of memory in my computer
train1 <- train[1:2000000,]
train1 <- merge(train1, merged2, by="song_id", all.x=TRUE)
train2 <- train[2000001:4000000,]
train2 <- merge(train2, merged2, by="song_id", all.x=TRUE)
train3 <- train[4000001:6000000,]
train3 <- merge(train3, merged2, by="song_id", all.x=TRUE)
train4 <- train[6000001:7377418,]
train4 <- merge(train4, merged2, by="song_id", all.x=TRUE)
train <- rbind(train1, train2, train3, train4)
rm(train1)
rm(train2)
rm(train3)
rm(train4)
test <- merge(test, merged2, by="song_id", all.x=TRUE)

merged3<- merged[,c("source_system_tab", "source_system_tab1")]
merged3 <- unique(merged3, by="source_system_tab1")
train <- merge(train, merged3, by="source_system_tab", all.x=TRUE)
test <- merge(test, merged3, by="source_system_tab", all.x=TRUE)


# use XGB to build a model for prediction
library(xgboost)
train_X <- as.matrix(train[,c("msno1", "song_id1","source_system_tab1","genre_ids", "language", "city", "bd", "gender")])
train_y <- as.matrix(train[,"target"])
test_X <- as.matrix(test[,c("msno1", "song_id1", "source_system_tab1", "genre_ids", "language", "city", "bd", "gender", "id")])

# define parameters for XGB
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.7, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

# use XGB_CV to get best number of rounds
xgbcv <- xgb.cv( params = params, data = train_X, label = train_y, nrounds = 150, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F, verbose = 1)

# use he best iteration number from above with eta = 0.2
model.xgb<- xgboost(data = train_X, label = train_y,nrounds = xgbcv$best_iteration, params=params, verbose = 1) 
xgb_pred <- predict(model.xgb, test_X)

# output to CSV file
target <- as.matrix(xgb_pred)
colnames(target) <- "target"
output <- data.frame(cbind(test_X, target))[,c("id", "target")]
# sort by id
output <- output[order(output$id),]
# write csv fie
write.csv(output,'c:/Users/Vincent Lee/Desktop/MFIT5010/Project/submission.csv', row.names = FALSE)