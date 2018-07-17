
require(cluster)
require(parallel)

cluster <- function(data) {

    k = seq(2, 100, by=1)
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

main <- function(file) {

    data = scale(read.csv(file, sep=";"))

    # clustering
    aux = cluster(data)
    silhouete(unlist(aux), which.max(aux))

    # pca analysis
    aux = pca(data)
    scatter(aux)
}


#execute
main(file)
