require(e1071)
require(kknn)
require(randomForest)
require(RWeka)

MLP = make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")

ANN <- function(tran, test) {
  model = MLP(Perc_Falha ~ ., tran)
  as.numeric(predict(model, test))
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
  model = lm(Perc_Falha ~., tran)
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

leaveout <- function(data) {

  id = 1:nrow(data)

  tran = lapply(1:nrow(data), function(i) {
    subset(data, id %in% setdiff(1:nrow(data), i))
  })

  test = lapply(1:nrow(data), function(i) {
    subset(data, id %in% i)
  })

  tmp = list()
  tmp$tran = tran
  tmp$test = test
  return(tmp)
}

normalize <- function(data) {

    for(i in 1:ncol(data))
        if(is.numeric(data[,i]))
            data[,i] = (data[,i] - min(data[,i]))/(max(data[,i]) - min(data[,i]))
    data
}

REGRESSORS = c("ANN", "DWNN", "RF", "SVR", "LM", "Default")

main <- function(file) {

  data = read.csv(file, sep=";")
  data[is.na(data)] = 0

  data = normalize(data)

  tmp = leaveout(data)

  aux = mapply(function(tran, test) {
    evaluation(tran, test)
  }, tran=tmp$tran, test=tmp$test)

  boxplot(t(aux))
}
