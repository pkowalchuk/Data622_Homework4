
data<-iris
dim(iris)
tr<-sample(1:nrow(iris),nrow(iris)*0.5)
dim(as.data.frame(tr))
trdata<-iris[tr,]
tstdata<-iris[-tr,]

x<-tstdata[,1:4]
y<-tstdata[[5]]

if(!require(VGAM)) library(VGAM)
if(!require(caret)) library(caret)

vglm_model<-vglm(Species~.,family = "multinomial",data=trdata)

vglm_class_probabilities<-predict(vglm_model,tstdata[,1:4],type="response")

vglm_predicted_class<-apply(vglm_class_probabilities,1,which.max)

vglm_pred<-c()
vglm_pred[which(vglm_predicted_class=="1")]<-levels(y)[1]
vglm_pred[which(vglm_predicted_class=="2")]<-levels(y)[2]
vglm_pred[which(vglm_predicted_class=="3")]<-levels(y)[3]
vglm_pred[which(vglm_predicted_class=="4")]<-levels(y)[4]

vglm_mtab<-table(vglm_pred,tstdata[[5]])
(vglm_cmx<-confusionMatrix(table(vglm_pred,tstdata[[5]])))

(vglm_accuracy<-sum(diag(vglm_mtab))/sum(vglm_mtab))

if(!require(MASS)) library(MASS)

lda_model<-lda(Species~.,trdata)

lda_class_probabilities<-predict(lda_model,tstdata,type="response")

(lda_cmx<-table(lda_class_probabilities$class,tstdata[[5]]))
lda_mtab<-table(lda_class_probabilities$class,tstdata[[5]])
(lda_accuracy<-sum(diag(lda_cmx))/sum(lda_cmx))
 lda_cmx<-confusionMatrix(table(lda_class_probabilities$class,tstdata[[5]]))
 
if(!require(rpart)) library(rpart)
if(!require(rpart.plot)) library(rpart.plot)
if(!require(randomForest)) library(randomForest)

rpart_model<-rpart(Species~.,trdata)

rpart_class_probabilities<-predict(rpart_model,tstdata[,1:4],type="class")

(rpart_mtab<-table(rpart_class_probabilities,tstdata[[5]]))
rpart_cmx<-confusionMatrix(rpart_mtab)
(rpart_accuracy<-sum(diag(rpart_mtab))/sum(rpart_mtab))

# Now let us do ensemble methods
# let us start with bagging bootstrap aggregation

#bagging
if(!require(ipred)) library(ipred)

bag_model<-bagging(Species~.,trdata)

bag_class_probabilities<-predict(bag_model,tstdata[,1:4])#,type="response")

(bag_mtab<-table(bag_class_probabilities,tstdata[[5]]))
(bag_cmx<-confusionMatrix(bag_mtab))

(bag_accuracy<-sum(diag(bag_mtab))/sum(bag_mtab))


nlev<-4 # number of classes
if(!require(gbm)) library(gbm)
gbm_model<-gbm(Species~.,trdata,
               n.trees=5000,
               interaction.depth=nlev,
	             shrinkage=0.001,
               bag.fraction=0.8,
               distribution="multinomial",
               verbose=FALSE,
               n.cores=4)
gbm_class_probabilities<-predict(gbm_model,tstdata[,1:4],n.trees=5000,type="response")
gbm_pred<-apply(gbm_class_probabilities,1,which.max)

gbm_predicted_class<-unlist(lapply(gbm_pred,FUN=function(x)levels(tstdata[[5]])[[x]]))

(gbm_mtab<-table(gbm_predicted_class,tstdata[[5]]))
(gbm_accuracy<-sum(diag(gbm_mtab))/sum(gbm_mtab))



gbm_model2<-gbm(Species~.,trdata,
                n.trees=5000,
                interaction.depth=nlev,
                shrinkage=0.001,
                bag.fraction=0.8,
                distribution="multinomial",
                verbose=FALSE,
                n.cores=4)
gbm_class_probabilities2<-predict(gbm_model2,tstdata[,1:4],n.trees=5000,type="response")
gbm_pred2<-apply(gbm_class_probabilities2,1,which.max)
gbm_pred2[which(gbm_pred2=="1")]<-levels(tstdata[[5]])[1]
gbm_pred2[which(gbm_pred2=="2")]<-levels(tstdata[[5]])[2]
gbm_pred2[which(gbm_pred2=="3")]<-levels(tstdata[[5]])[3]
gbm_pred2[which(gbm_pred2=="4")]<-levels(tstdata[[5]])[4]
gbm_pred2<-as.factor(gbm_pred2)
l<-union(gbm_pred2,tstdata[[5]])
(gbm_mtab2<-table(factor(gbm_pred2,l),factor(tstdata[[5]],l)))
(gbm_accuracy2<-sum(diag(gbm_mtab2))/sum(gbm_mtab2))
(gbm_cmx2<-confusionMatrix(gbm_mtab2))

nlev<-5 # number of classes+1
gbm_model3<-gbm(Species~.,trdata,
                n.trees=5000,
                interaction.depth=nlev,
	              shrinkage=0.001,
                bag.fraction=0.8,
                distribution="multinomial",
                verbose=FALSE,
                n.cores=4)
gbm_class_probabilities3<-predict(gbm_model3,tstdata[,1:4],n.trees=5000,type="response")
gbm_pred3<-apply(gbm_class_probabilities3,1,which.max)

gbm_predicted_class3<-unlist(lapply(gbm_pred3,FUN=function(x)levels(tstdata[[5]])[[x]]))

(gbm_mtab3<-table(gbm_predicted_class3,tstdata[[5]]))
(gbm_accuracy3<-sum(diag(gbm_mtab3))/sum(gbm_mtab3))
(gbm_cmx3<-confusionMatrix(gbm_mtab3))

if(!require(randomForest))require(randomForest)
rf_model<-randomForest(Species~.,
data=trdata)
rf_pred<-predict(rf_model,tstdata[,1:4])
rf_mtab<-table(rf_pred,tstdata[[5]])
rf_cmx<-confusionMatrix(rf_mtab)
rf_cmx$overall
rf_cmx$byClass


#XGBoost only works with numeric vectors. 
#need to convert all other forms of data into numeric vectors.
# we use Matrix sparse.model.matrix for that

if(!require(xgboost)) library(xgboost)
if(!require(Matrix)) library(Matrix)
trdatamx<-sparse.model.matrix(Species~.-1,trdata)
tstdatamx<-sparse.model.matrix(Species~.-1,tstdata)

xgb_model<-xgboost::xgboost(data=trdatamx,label=trdata$Species,max_depth = 2, 
eta = 1, nrounds = 2,nthread = 2, objective = "multi:softmax",num_class=5)

xgb_pred <- predict(xgb_model,tstdatamx)
xgb_tab<-table( xgb_pred)
xgb_mtab<-table(xgb_pred,tstdata[[5]])
#xgb_cmx<-confusionMatrix(xgb_mtab)
#xgb_cmx$overall
#xgb_cmx$byClass


xgb_model4<-xgboost::xgboost(data=trdatamx,label=trdata$Species,max_depth = 4, 
eta = 1, nrounds = 3,nthread = 2, objective = "multi:softmax",num_class=5)
 xgb_pred4 <- predict(xgb_model4,tstdatamx)
xgb_tab4<-table( xgb_pred4)
temp_xgb_tab4<-xgb_tab4

xgb_pred4[which(xgb_pred4=="1")]<-levels(y)[1]
xgb_pred4[which(xgb_pred4=="2")]<-levels(y)[2]
xgb_pred4[which(xgb_pred4=="3")]<-levels(y)[3]
xgb_pred4[which(xgb_pred4=="4")]<-levels(y)[4]
xgb_mtab4<-table(xgb_pred4,tstdata[[5]])
xgb_cmx4<-confusionMatrix(xgb_mtab4)
xgb_cmx4$overall
xgb_cmx4$byClass

xgb_model5<-xgboost::xgboost(data=trdatamx,label=trdata$Species,max_depth = 5, 
eta = 1, nrounds = 4,nthread = 2, objective = "multi:softmax",num_class=5)
 xgb_pred5 <- predict(xgb_model5,tstdatamx)
table( xgb_pred5)

xgb_tab5<-table( xgb_pred5)
temp_xgb_tab5<-xgb_tab5

xgb_pred5[which(xgb_pred5=="1")]<-levels(y)[1]
xgb_pred5[which(xgb_pred5=="2")]<-levels(y)[2]
xgb_pred5[which(xgb_pred5=="3")]<-levels(y)[3]
xgb_pred5[which(xgb_pred5=="4")]<-levels(y)[4]
xgb_mtab5<-table(xgb_pred5,tstdata[[5]])
xgb_cmx5<-confusionMatrix(xgb_mtab5)
xgb_cmx5$overall
xgb_cmx5$byClass

lapply(ls()[grep("mtab",ls())],FUN=function(x)eval(x))

txt<-capture.output({lapply(ls()[grep("cmx",ls())],FUN=function(x)eval(parse(text=x)))})

writeLines(txt,"confusionMxOutput.txt")

mtabtxt<-capture.output({lapply(ls()[grep("mtab",ls())],FUN=function(x)eval(parse(text=x)))})
writeLines(mtabtxt,"mtabOutput.txt") 

