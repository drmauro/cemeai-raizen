require(e1071)
require(kknn)
require(randomForest)
require(rpart)
require(xgboost)

REGRESSORS = c("DWNN", "CART", "RF", "SVR", "XGB", "LM", "DF1", "DF2")

XGB <- function(tran, test) {
  tran = binarize(tran)
  test = binarize(test)
  bstDense = xgboost(data = as.matrix(tran[,-1]),
    label = tran$Perc_Falha, max_depth = 2, eta = 1,
    nthread = 3, nrounds = 2, objective = "binary:logistic", verbose=0)
  pred = predict(bstDense, as.matrix(test[,-1]))
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

  data = data[sample(1:nrow(data), replace=FALSE),]
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

remove <- function(data) {
  data[data$Perc_Falha >= 0.3,]
}

main <- function(file) {

  data = read.csv(file, sep=";")
  data[is.na(data)] = 0

  df = normalize(data)
  df$Perc_Falha = data$Perc_Falha
  data = df

  tmp = cfold(df)

  aux = mapply(function(tran, test) {
    evaluation(tran, test)
  }, tran=tmp$tran, test=tmp$test)

  aux = do.call("rbind", aux)

  pdf(paste("mse.pdf", sep="."))
    boxplot(aux, xlab="Regressors", ylab="MSE", main=paste("Performance of the regressors", sep=""), outline=FALSE)
  dev.off()
}

set.seed(1234)
