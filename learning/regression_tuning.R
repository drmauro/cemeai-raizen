require(e1071)
require(kknn)
require(randomForest)
require(rpart)
require(xgboost)
require(mlr)
library("parallelMap")
parallelStartSocket(3)

REGRESSORS = c("DWNN", "CART", "RF", "SVR", "XGB", "LM", "DF1", "DF2")
# REGRESSORS = c("DWNN", "CART", "DF1")
ITER = 2
stored.models = list(DWNN=list(), CART=list(), RF=list(), 
                     SVR=list(), XGB=list(), LM=list())

XGB <- function(tran, test) {
  tran = binarize(tran)
  test = binarize(test)
 
  trainTask = makeRegrTask(data = tran, target = "Perc_Falha")
  testTask = makeRegrTask(data = test, target = "Perc_Falha")
  
  xgb_learner <- makeLearner(
    "regr.xgboost",
    predict.type = "response",
    par.vals = list(
      objective = "binary:logistic",
      eval_metric = "error",
      nrounds = 200
    )
  )

  xgb_params <- makeParamSet(
    makeIntegerParam("nrounds", lower = 200, upper = 500),
    makeIntegerParam("max_depth", lower = 1, upper = 10),
    makeNumericParam("eta", lower = .01, upper = .3),
    makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
  )
  
  control <- makeTuneControlRandom(maxit = ITER)

  tuned_params <- tuneParams(
    learner = xgb_learner,
    task = trainTask,
    resampling = cv5,
    par.set = xgb_params,
    control = control
  )

  xgb_tuned_learner <- setHyperPars(
    learner = xgb_learner,
    par.vals = tuned_params$x
  )

  xgb_model <- train(xgb_tuned_learner, trainTask)

  result <- predict(xgb_model, testTask)
  stored.models$XGB = append(stored.models$XGB, xgb_tuned_learner$par.vals)
  save(xgb_model,
       file=paste("learning/models/XGB_",
                  as.integer(as.POSIXct( Sys.time() )), ".RData", sep=''))
  result$data$response
}

DWNN <- function(tran, test) {

  tran = binarize(tran)
  test = binarize(test)

  trainTask = makeRegrTask(data = tran, target = "Perc_Falha")
  testTask = makeRegrTask(data = test, target = "Perc_Falha")
  
  dwnn_learner <- makeLearner(
    "regr.kknn",
    predict.type = "response",
    par.vals = list(
      k = 3,
      distance = 1,
      kernel = "gaussian"
    )
  )

  dwnn_params <- makeParamSet(
    makeIntegerParam("k", lower = 3, upper = 15),
    makeIntegerParam("distance", lower = 1, upper = 10),
    makeDiscreteParam("kernel", values=c("rectangular", "triangular",
                                         "epanechnikov", "biweight",
                                         "triweight", "cos", "inv",
                                         "gaussian", "optimal")) 
  )
  
  control <- makeTuneControlRandom(maxit = ITER)

  tuned_params <- tuneParams(
    learner = dwnn_learner,
    task = trainTask,
    resampling = cv5,
    par.set = dwnn_params,
    control = control
  )

  dwnn_tuned_learner <- setHyperPars(
    learner = dwnn_learner,
    par.vals = tuned_params$x
  )

  dwnn_model <- train(dwnn_tuned_learner, trainTask)

  result <- predict(dwnn_model, testTask)
  save(dwnn_model,
       file=paste("learning/models/DWNN_",
                  as.integer(as.POSIXct( Sys.time() )), ".RData", sep=''))
  result$data$response 
}

CART <- function(tran, test) {
  
  tran = binarize(tran)
  test = binarize(test)

  trainTask = makeRegrTask(data = tran, target = "Perc_Falha")
  testTask = makeRegrTask(data = test, target = "Perc_Falha")
  
  rpart_learner <- makeLearner(
    "regr.rpart",
    predict.type = "response"
  )

  rpart_params <- makeParamSet(
    makeIntegerParam("minsplit", lower = 1, upper = 50),
    makeIntegerParam("maxcompete", lower = 1, upper = 10),
    # makeIntegerParam("maxsurrogate", lower = 1, upper = 10),
    makeIntegerParam("maxdepth", lower = 1, upper = 30),
    makeIntegerParam("xval", lower = 1, upper = 30),
    makeNumericParam("cp", lower = 0, upper = 1)
    # makeDiscreteParam("usesurrogate", values=c(0,1,2)), 
    # makeDiscreteParam("surrogatestyle", values=c(0,1)) 
  )
  
  control <- makeTuneControlRandom(maxit = ITER)

  tuned_params <- tuneParams(
    learner = rpart_learner,
    task = trainTask,
    resampling = cv5,
    par.set = rpart_params,
    control = control
  )

  rpart_tuned_learner <- setHyperPars(
    learner = rpart_learner,
    par.vals = tuned_params$x
  )

  rpart_model <- train(rpart_tuned_learner, trainTask)

  result <- predict(rpart_model, testTask)
  save(rpart_model,
       file=paste("learning/models/RPART_",
                  as.integer(as.POSIXct( Sys.time() )), ".RData", sep=''))
  result$data$response 
}

RF <- function(tran, test) {

  tran = binarize(tran)
  test = binarize(test)

  trainTask = makeRegrTask(data = tran, target = "Perc_Falha")
  testTask = makeRegrTask(data = test, target = "Perc_Falha")
  
  rf_learner <- makeLearner(
    "regr.randomForest",
    predict.type = "response"
  )

  rf_params <- makeParamSet(
    makeIntegerParam("ntree", lower = 100, upper = 800)
    # makeIntegerParam("mtry", lower = 1, upper = 10),
  )
  
  control <- makeTuneControlRandom(maxit = ITER)

  tuned_params <- tuneParams(
    learner = rf_learner,
    task = trainTask,
    resampling = cv5,
    par.set = rf_params,
    control = control
  )

  rf_tuned_learner <- setHyperPars(
    learner = rf_learner,
    par.vals = tuned_params$x
  )

  rf_model <- train(rf_tuned_learner, trainTask)

  result <- predict(rf_model, testTask)
  save(rf_model,
       file=paste("learning/models/RF_",
                  as.integer(as.POSIXct( Sys.time() )), ".RData", sep=''))
  result$data$response 
}

SVR <- function(tran, test) {
  tran = binarize(tran)
  test = binarize(test)

  trainTask = makeRegrTask(data = tran, target = "Perc_Falha")
  testTask = makeRegrTask(data = test, target = "Perc_Falha")
  
  svr_learner <- makeLearner(
    "regr.svm",
    predict.type = "response"
  )

  svr_params <- makeParamSet(
    makeDiscreteParam("kernel", values=c('linear','polynomial','radial')), 
    makeIntegerParam("degree", lower = 1, upper = 3),
    makeNumericParam("gamma", lower = 0, upper = 10),
    makeNumericParam("coef0", lower = -10, upper = 10),
    makeNumericParam("cost", lower = .01, upper = 10)
  )
  
  control <- makeTuneControlRandom(maxit = ITER)

  tuned_params <- tuneParams(
    learner = svr_learner,
    task = trainTask,
    resampling = cv5,
    par.set = svr_params,
    control = control
  )

  svr_tuned_learner <- setHyperPars(
    learner = svr_learner,
    par.vals = tuned_params$x
  )

  svr_model <- train(svr_tuned_learner, trainTask)

  result <- predict(svr_model, testTask)
  save(svr_model,
       file=paste("learning/models/SVR_",
                  as.integer(as.POSIXct( Sys.time() )), ".RData", sep=''))
  result$data$response 

}

LM <- function(tran, test) {
  tran = binarize(tran)
  test = binarize(test)

  trainTask = makeRegrTask(data = tran, target = "Perc_Falha")
  testTask = makeRegrTask(data = test, target = "Perc_Falha")
  
  lm_learner <- makeLearner(
    "regr.lm",
    predict.type = "response"
  )

  lm_params <- makeParamSet(
    makeNumericParam("tol", lower = 0, upper = 1)
  )
  
  control <- makeTuneControlRandom(maxit = ITER)

  tuned_params <- tuneParams(
    learner = lm_learner,
    task = trainTask,
    resampling = cv5,
    par.set = lm_params,
    control = control
  )

  lm_tuned_learner <- setHyperPars(
    learner = lm_learner,
    par.vals = tuned_params$x
  )

  lm_model <- train(lm_tuned_learner, trainTask)

  result <- predict(lm_model, testTask)
  save(lm_model,
       file=paste("learning/models/LM_",
                  as.integer(as.POSIXct( Sys.time() )), ".RData", sep=''))
  result$data$response 

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

cfold <- function(data, nfold=10) {

  data = data[sample(1:nrow(data), replace=FALSE),]
  id = rep(1:nfold, length.out=nrow(data))

  tran = lapply(1:nfold, function(i) {
    subset(data, id %in% setdiff(1:nfold, i))
  })

  test = lapply(1:nfold, function(i) {
    subset(data, id %in% i)
  })

  tmp = list()
  tmp$data = data
  tmp$tran = tran
  tmp$test = test
  return(tmp)
}

form <- function(x) {
  att <- paste(colnames(x), collapse="+")
  stats::formula(paste("~ 0 +", att, sep=" "))
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


  save(aux, file='learning/result_aux1.RData')
  
  tmp = min(sapply(aux, nrow))
  aux = Reduce("+", lapply(aux, "[", 1:tmp,))/10

  save(aux, file='learning/result_aux2.RData')
  boxplot(aux, outline=FALSE)
}

set.seed(1234)
main("data/raizen.csv")
parallelStop()
