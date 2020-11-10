library(rgl)
x <- c(-1, -1, 1)
y <- c(0, -1, -1)
z <- c(0, 0, 0)
M <- cbind(x,y,z)
rgl.bg(color="gray15")
triangles3d(M, col=rainbow(8))
