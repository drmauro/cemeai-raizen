require(e1071)
require(kknn)
require(randomForest)
require(rpart)

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

Default <- function(tran, test) {
  mean(tran[,"Perc_Falha"])
}

mse <- function(test, pred) {
  as.numeric((pred - test[,"Perc_Falha"])^2)
}

evaluation <- function(tran, test) {
  sapply(REGRESSORS, function(r) {
    pred = eval(call(r, tran, test))
    mse(test, pred)
  })
}

houdout <- function(data) {

  aux = sample(1:nrow(data), nrow(data)/3, replace=F)

  tmp = list()
  tmp$tran = data[setdiff(1:nrow(data), aux),]
  tmp$test = data[aux,]
  return(tmp)
}

normalize <- function(data) {

  for(i in 1:ncol(data))
    if(is.numeric(data[,i]))
      data[,i] = (data[,i] - min(data[,i]))/(max(data[,i]) - min(data[,i]))
  return(data)
}

REGRESSORS = c("DWNN", "CART", "RF", "SVR", "Default")

main <- function(file) {

  data = read.csv(file, sep=";")
  data[is.na(data)] = 0

  data = normalize(data)

  tmp = houdout(data)

  aux = mapply(function(tran, test) {
    evaluation(tran, test)
  }, tran=tmp$tran, test=tmp$test)

  boxplot(t(aux))
}
