# Quora R script with cosine similarity base model
# installations and library
install.packages("dplyr")
install.packages("stats")
install.packages("SnowballC")
install.packages("tm")
install.packages("caret")
library(stats)
library(dplyr)
library(SnowballC)
library(tm)
library(randomForest)
library(caret)
#library(syuzhet)
library(rpart)

# Importing data
q_train <- read.csv("~/Uconn 2nd Sem/Data Mining and Business Intelligence/Project/train.csv/train.csv",stringsAsFactors = F)
str(q_train)
# Checking proportion of duplicates and not duplicates in response
x=table(q_train$is_duplicate)
prop.table(x)

# taking a sample of 10000 questions
set.seed(299)
train=sample_n(q_train,10000,replace=FALSE)
# str(train)
# we don't wish to remove words like what,why,how so we are
# creating a custom stop word list
# expections- "what" "what's" "when"  "when's"  "where" "where's"  "which" "how"

SE=stopwords("english")
x=as.vector(SE)
x=sort(x)           # sorting for easy identification of index
x=x[-c(63,150:155)] ## 150:155 are indexes where expections are present
x=as.character(x) 

# initializing feature vectors
duplicate=vector()
cosM=vector()
q1length=vector()
q2length=vector()
eucl_dist=vector()
man_dist=vector()
can_dist=vector()
mink_dist=vector()
q1wordcount=vector()
q2wordcount=vector()
# taking the sampled question pairs and making predictions by calculating cosine similarity and other features
for(i in 1:nrow(train))
{
  qpair=Corpus(VectorSource(c(train$question1[i],train$question2[i])))
  # preprocessing starts
  qpair=tm_map(qpair,tolower)
  qpair=tm_map(qpair,removePunctuation)
  qpair=tm_map(qpair,removeWords,x)
  # Creating a term matrix with the question pair
  out=DocumentTermMatrix(x = qpair)
  # converting to a matrix
  out=as.matrix(out)
  # formula for cosine similarity
  cosM[i]=(sum(out[1,]*out[2,]))/length(out[1,])*length(out[2,])
  # we are using a threshold of 0.4 ; we can iterate and find optimal cutoff
  duplicate[i]=ifelse(cosM[i]>0.4,1,0)
  # duplicate matrix has our predictions
  # length of question 1 
  q1length[i]=nchar(q_train$question1[i])
  # length of question 2
  q2length[i]=nchar(q_train$question2[i])
  # word count of question 1
  q1wordcount[i]=sum(out[1,])
  # word count of question 2 
  q2wordcount[i]=sum(out[2,])
  # Euclidean distance
  eucl_dist[i]=(dist(out,method = "euclidean"))[1]
  # Manhattan distance
  man_dist[i]=(dist(out,method = "manhattan"))[1]
  # Canberra distance
  can_dist[i]=(dist(out,method = "canberra"))[1]
  # Minkowski distance
  mink_dist[i]=(dist(out,method = "minkowski"))[1]
}
# Length difference of 2 sentences
lengthdiff=as.vector(abs(q1length-q2length))
# count difference
wordiff=as.vector(abs(q1wordcount-q2wordcount))
# creating a data frame of the distances calculated
train_features=as.data.frame(cbind(q1length,q2length,q1wordcount,q2wordcount,man_dist,eucl_dist,can_dist,mink_dist,cosM,lengthdiff,wordiff))
train_features$response=as.factor(train$is_duplicate)
# there are some NA values in euclidean and other distances ; needs further investigation
train_features=na.omit(train_features)
str(train_features)

# Random Forest Modeling 
rf_model=randomForest(train_features$response~.,data=train_features,method="class",ntree=350,nodesize=5)
importance(rf_model)
plot(rf_model)
varImpPlot(rf_model)
table(rf_model$predicted,train_features$response)

# confusion matrix for our predictions

ConfusionM=table(rf_model$predicted,train_features$response)
ConfusionM

# useful function for preprocessing
prepos = function(qpair)
{
  qpair=tm_map(qpair,tolower)
  qpair=tm_map(qpair,removePunctuation)
  qpair=tm_map(qpair,removeWords,x)
  return(Corpus(VectorSource(qpair)))
}
# function to calculate cosine similarity
cosfun=function(out)
{
  cos=(sum(out[1,]*out[2,]))/length(out[1,])
  return(cos)  
}


# End of script