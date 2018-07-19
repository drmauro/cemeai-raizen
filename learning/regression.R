require(e1071)
require(kknn)
require(randomForest)
require(rpart)

REGRESSORS = c("DWNN", "CART", "RF", "SVR", "LM", "DF1")
REGRESSORS = c("XGB", "DF1")

XGB <- function(tran, test) {                                                                                                                 
   print("XGB")                                                                                                                               
   target_id = -2                                                                                                                             
   print('Training')
   tran = binarize(tran)
   test = binarize(test)
   bstDense = xgboost(data = as.matrix(tran[,-target_id]),                                                                                    
                      label = tran$Perc_Falha, max_depth = 2, eta = 1,                                                                        
                      nthread = 3, nrounds = 2, objective = "binary:logistic")                                                                
   print('Testing')                                                                                                                           
   pred = predict(bstDense, as.matrix(test[,target_id]))                                                                                      
   pred                                                                                                                                       
} 

DWNN <- function(tran, test) {
  model = kknn(Perc_Falha ~., tran, test, kernel="gaussian")
  model$fitted.values
}

CART <- function(tran, test) {
  model = rpart(Perc_Falha ~., tran, method="anova")
  as.numeric(predict(model, test))
}

RF <- function(tran, test) {
  model = randomForest(Perc_Falha ~., tran)
  as.numeric(predict(model, test))
}

SVR <- function(tran, test) {
  model = svm(Perc_Falha ~., tran, scale=TRUE, type="eps-regression", kernel="radial")
  as.numeric(predict(model, test))
}

LM <- function(tran, test) {
  model = lm(Perc_Falha ~., binarize(tran))
  as.numeric(predict(model, binarize(test)))
}

DF1 <- function(tran, test) {
  mean(tran[,"Perc_Falha"])
}

DF2 <- function(tran, test) {
  median(tran[,"Perc_Falha"])
}

mse <- function(test, pred) {
  as.numeric((pred - test[,"Perc_Falha"])^2)
}

binarize <- function(x) {
  data.frame(stats::model.matrix(form(x), x))
}

evaluation <- function(tran, test) {
  sapply(REGRESSORS, function(r) {
    pred = eval(call(r, tran, test))
    mse(test, pred)
  })
}

cfold <- function(data) {

  id = rep(1:10, length.out=nrow(data))

  tran = lapply(1:10, function(i) {
    subset(data, id %in% setdiff(1:10, i))
  })

  test = lapply(1:10, function(i) {
    subset(data, id %in% i)
  })

  tmp = list()
  tmp$data = data
  tmp$tran = tran
  tmp$test = test
  return(tmp)
}

normalize <- function(data) {

  for(i in 1:ncol(data))
    if(is.numeric(data[,i]))
      data[,i] = (data[,i] - min(data[,i]))/(max(data[,i]) - min(data[,i]))
  return(data)
}

main <- function(file) {

  data = read.csv(file, sep=";")
  data[is.na(data)] = 0

  data = normalize(data)
  tmp = cfold(data)

  aux = mapply(function(tran, test) {
    evaluation(tran, test)
  }, tran=tmp$tran, test=tmp$test)

  tmp = min(sapply(aux, nrow))
  aux = Reduce("+", lapply(aux, "[", 1:tmp,))/10

  boxplot(aux, outline=FALSE)
}
