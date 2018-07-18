
require(cluster)
require(parallel)

cluster <- function(data, k) {

    aux = mclapply(k, mc.cores=4, function(i) {
        sil = sapply(1:10, function(j) {
            tmp = kmeans(data, i)
            aux = silhouette(tmp$cluster, daisy(data))
            mean(aux[,3])
        })
        mean(sil)
    })

    return(aux)
}

silhouete <- function(data, v) {
    plot(unlist(data), xlab="k value", ylab="Silhouette")
    abline(v=v)
}


pca <- function(data) {
    prcomp(data)
}

scatter <- function(data) {
    tmp = length(which(summary(aux)$importance[3,] < 0.9)) + 1
    plot(data[,1:tmp])
}

form <- function(x) {
  att <- paste(colnames(x), collapse="+")
  stats::formula(paste("~ 0 +", att, sep=" "))
}

binarize <- function(x) {
  data.frame(stats::model.matrix(form(x), x))
}

normalize <- function(data) {

    for(i in 1:ncol(data))
        if(is.numeric(data[,i]))
            data[,i] = (data[,i] - min(data[,i]))/(max(data[,i]) - min(data[,i]))
    data
}

main <- function(file, k=seq(2, 100, by=1)) {

    data = read.csv(file, sep=";")
    data[is.na(data)] = 0

    data = normalize(data)
    data = binarize(data)

    # clustering
    aux = cluster(data, k)
    silhouete(unlist(aux), which.max(aux))

    # pca analysis
    aux = pca(data)
    scatter(aux)
}


#execute
#main(file)
