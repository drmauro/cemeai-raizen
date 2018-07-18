


zonas = unique(data$Zona)

tmp = lapply(zonas, function(zona) {

	aux = data[data$Zona == zona, 12:23, drop=F]
	if(!all(is.na(aux))) {
		for(i in 1:ncol(aux)) {
			aux[is.na(aux[,i]),i] <- median(aux[,i], na.rm=T)
		}
		aux
	} else {
		aux
	}


}