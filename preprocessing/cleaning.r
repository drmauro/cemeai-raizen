data = read.csv('../data/2017-2016/raizen_meteo.csv', sep=';')

data = data[,-1]
data = data[,-1]
data = data[,-ncol(data)]
data = data[,-ncol(data)]
data = data[,-ncol(data)]
remove = which(data$P > 100, arr.ind=T)
data[remove,] = NA
data$Mg[which(data$Mg == 663.765)] = NA
# data$H_al[which(data$H_al == 181.5)] = NA
data$Na <- NULL
data$Sb[which(data$Sb == 835.115)] = NA
data$Ctc[which(data$Ctc == 845)] = NA
data$Ph[which(data$Ph == 28.8)] = NA
data$Ambiente[data$Ambiente == "mA"] = "A"
data$Ambiente[data$Ambiente == "mB"] = "B"
data$Ambiente[data$Ambiente == "mC"] = "C"
data$Ambiente[data$Ambiente == "mD"] = "D"
data$Ambiente[data$Ambiente == "mE"] = "E"
data$Ambiente[data$Ambiente == "mF"] = "F"
data$Ambiente[data$Ambiente == "mG"] = "G"
data$Ambiente[data$Ambiente == "ADF"] = NA
aux = rep(0, nrow(data))
aux[data$Ambiente == "A"] = 1
aux[data$Ambiente == "B"] = 2
aux[data$Ambiente == "C"] = 3
aux[data$Ambiente == "D"] = 4
aux[data$Ambiente == "E"] = 5
aux[data$Ambiente == "F"] = 6
aux[data$Ambiente == "G"] = 7
data$Ambiente = aux
data$Ambiente[which(data$Ambiente == 0)] = NA
data$RESI_M3[which(data$RESI_M3 > 0)] = 1          
data$RESI_T[which(data$RESI_T > 0)] = 1
data$FERT_KG[data$FERT_KG>2] = NA



