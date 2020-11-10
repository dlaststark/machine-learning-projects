# Generate and plot Brownian tree. Version #4.
# 7/27/16 aev
# gpBrownianTree4(m, n, clr, fn, ttl, dflg, seed, psz)
# Where: m - defines matrix m x m; n - limit of the number of moves;
#   fn - file name (.ext will be added); ttl - plot title; dflg - 0-no dump,
#   1-dump; seed - 0-center, 1-random: psz - picture size.
gpBrownianTree4 <- function(m, n, clr, fn, ttl, dflg=0, seed=0, psz=600)
{
  cat(" *** START:", date(), "m=",m, "n=",n, "clr=",clr, "psz=",psz, "\n");
  M <- matrix(c(0), ncol=m, nrow=m, byrow=TRUE);
  # Random seed
  if(seed==1)
    {x <- sample(1:m, 1, replace=FALSE);y <- sample(1:m, 1, replace=FALSE)}
  # Seed in center
  else {x <- m%/%2; y <- m%/%2}
  M[x,y]=1;
  pf=paste0(fn,".png");
  cat(" *** Plot file -",pf,"Seed:",x,"/",y,"\n");
  # Main loops
  for (i in 1:n) {
    if(i>1) {
      x <- sample(1:m, 1, replace=FALSE)
      y <- sample(1:m, 1, replace=FALSE)}
    while((x<=m && y<=m && x>0 && y>0)) {
      if(!(x+1<=m && y+1<=m && x-1>0 && y-1>0)) {break;}
      b=M[x+1,y+1]+M[x,y+1]+M[x-1,y+1]+M[x+1,y];
      b=b+M[x-1,y-1]+M[x-1,y]+M[x,y-1]+M[x+1,y-1];
      if(b!=0) {break;}
      x <- x + sample(-1:1, 1, replace=FALSE)
      y <- y + sample(-1:1, 1, replace=FALSE)
      if(!(x<=m && y<=m && x>0 && y>0))
        { x <- sample(1:m, 1, replace=FALSE)
          y <- sample(1:m, 1, replace=FALSE)
        }
    }
    M[x,y]=1;
  }
  plotmat(M, fn, clr, ttl, dflg, psz);
  cat(" *** END:",date(),"\n");
}
gpBrownianTree4(400,15000,"navy", "BT4R", "Brownian Tree v.4", 1);
