# Elastic Net -------------------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")

# Elastic Net with Intercept ----------------------------------------------

# misspecification --------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(5E-2),length.out=50)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
  }
}
set.seed(1234)

# simulation
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
intercept=1
X=matrix(rnorm(n*p,mean=0,sd=sqrt(1/k)),ncol=p)
sigma=rnorm(n,mean=0,sd=0.5)
y=intercept+X%*%beta+sigma
index=which(y>=0)
y[index]=sqrt(y[index])
y[-index]=-sqrt(-y[-index])

# true leave-one-out
y.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
library(foreach)
library(doParallel)
no_cores=detectCores()-1
cl=makeCluster(no_cores)
registerDoParallel(cl)
for(i in 1:n) {
  # do leave one out prediction
  y.temp<-foreach(k=1:length(alpha),.combine=cbind,.packages='glmnet') %dopar%
    Elastic_Net_LOO(X,y,i,alpha[k],lambda,intercept=TRUE)
  # save the prediction value
  y.loo[i,]=y.temp
  # print middle result
  if(i%%10==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo=1/n*colSums((y.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,y.loo,
     file="Elastic_Net_Misspec_LOO.RData")

# approximate leave-one-out
load('Elastic_Net_Misspec_LOO.RData')
# compute the scale parameter for y
sd.y=as.numeric(sqrt(var(y)*length(y)/(length(y)-1)))
y.scaled=y/sd.y
X.scaled=X/sd.y
# find the ALO prediction
y.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X.scaled,y=y.scaled,family="gaussian",
               alpha=alpha[k],lambda=lambda/sd.y^2,thresh=1E-14,
               intercept=TRUE,standardize=FALSE,maxit=1000000)
  # find the prediction for each alpha value
  y.temp<-foreach(j=1:length(lambda),.combine=cbind) %do% {
    ElasticNetALO(as.vector(model$beta[,j]),model$a0[j]*sd.y,X,y,lambda[j],alpha[k])
  }
  y.alo[,((k-1)*length(lambda)+1):(k*length(lambda))]=y.temp
  # print middle result
  print(paste(k," alphas have beed calculated. ",
              "On average, every alpha needs ",
              round((proc.time()-starttime)[3]/k,2)," seconds."))
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((y.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,y.loo,y.alo,
     file="Elastic_Net_Misspec_ALO.RData")

# plot
load("Elastic_Net_Misspec_ALO.RData")
result$alpha=factor(result$alpha)
p=ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
bmp("Elastic_Net_Misspec_with_Intercept.bmp",width=1280,height=720)
p
dev.off()


# heavy-tailed noise ------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(4E-2),length.out=50)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
  }
}
set.seed(1234)

# simulation
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
intercept=1
X=matrix(rnorm(n*p,mean=0,sd=sqrt(1/k)),ncol=p)
sigma=rt(n,df=3)*sqrt(0.25/3)
y=intercept+X%*%beta+sigma

# true leave-one-out
y.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
library(foreach)
library(doParallel)
no_cores=detectCores()-1
cl=makeCluster(no_cores)
registerDoParallel(cl)
for(i in 1:n) {
  # do leave one out prediction
  y.temp<-foreach(k=1:length(alpha),.combine=cbind,.packages='glmnet') %dopar%
    Elastic_Net_LOO(X,y,i,alpha[k],lambda,intercept=TRUE)
  # save the prediction value
  y.loo[i,]=y.temp
  # print middle result
  if(i%%10==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo=1/n*colSums((y.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,y.loo,
     file="Elastic_Net_HTN_LOO.RData")

# approximate leave-one-out
load('Elastic_Net_HTN_LOO.RData')
# compute the scale parameter for y
sd.y=as.numeric(sqrt(var(y)*length(y)/(length(y)-1)))
y.scaled=y/sd.y
X.scaled=X/sd.y
# find the ALO prediction
y.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X.scaled,y=y.scaled,family="gaussian",
               alpha=alpha[k],lambda=lambda/sd.y^2,thresh=1E-14,
               intercept=TRUE,standardize=FALSE,maxit=1000000)
  # find the prediction for each alpha value
  y.temp<-foreach(j=1:length(lambda),.combine=cbind) %do% {
    ElasticNetALO(as.vector(model$beta[,j]),model$a0[j]*sd.y,X,y,lambda[j],alpha[k])
  }
  y.alo[,((k-1)*length(lambda)+1):(k*length(lambda))]=y.temp
  # print middle result
  print(paste(k," alphas have beed calculated. ",
              "On average, every alpha needs ",
              round((proc.time()-starttime)[3]/k,2)," seconds."))
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((y.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,y.loo,y.alo,
     file="Elastic_Net_HTN_ALO.RData")

# plot
load("Elastic_Net_HTN_ALO.RData")
result$alpha=factor(result$alpha)
p=ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
bmp("Elastic_Net_HTN_with_Intercept.bmp",width=1280,height=720)
p
dev.off()


# correlated design -------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(4E-2),length.out=50)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
  }
}
set.seed(1234)

# define Toeplitz matrix
rho=0.8
C=matrix(nrow=p,ncol=p)
for(i in 1:p)
{
  C[i,1:i]=rho^seq(i,1,by=-1)
  if(i==p)
    break
  else
    C[i,(i+1):p]=rho^seq(2,p-i+1,by=1)
}

# simulation
library(MASS)
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
intercept=1
X=mvrnorm(n,mu=rep(0,p),Sigma=C/k)
sigma=rnorm(n,mean=0,sd=0.5)
y=intercept+X%*%beta+sigma

# true leave-one-out
y.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
library(foreach)
library(doParallel)
no_cores=detectCores()-1
cl=makeCluster(no_cores)
registerDoParallel(cl)
for(i in 1:n) {
  # do leave one out prediction
  y.temp<-foreach(k=1:length(alpha),.combine=cbind,.packages='glmnet') %dopar%
    Elastic_Net_LOO(X,y,i,alpha[k],lambda,intercept=TRUE)
  # save the prediction value
  y.loo[i,]=y.temp
  # print middle result
  if(i%%10==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo=1/n*colSums((y.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,y.loo,
     file="Elastic_Net_CorrDesign_LOO.RData")

# approximate leave-one-out
load('Elastic_Net_CorrDesign_LOO.RData')
# compute the scale parameter for y
sd.y=as.numeric(sqrt(var(y)*length(y)/(length(y)-1)))
y.scaled=y/sd.y
X.scaled=X/sd.y
# find the ALO prediction
y.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X.scaled,y=y.scaled,family="gaussian",
               alpha=alpha[k],lambda=lambda/sd.y^2,thresh=1E-14,
               intercept=TRUE,standardize=FALSE,maxit=1000000)
  # find the prediction for each alpha value
  y.temp<-foreach(j=1:length(lambda),.combine=cbind) %do% {
    ElasticNetALO(as.vector(model$beta[,j]),model$a0[j]*sd.y,X,y,lambda[j],alpha[k])
  }
  y.alo[,((k-1)*length(lambda)+1):(k*length(lambda))]=y.temp
  # print middle result
  print(paste(k," alphas have beed calculated. ",
              "On average, every alpha needs ",
              round((proc.time()-starttime)[3]/k,2)," seconds."))
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((y.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,y.loo,y.alo,
     file="Elastic_Net_CorrDesign_ALO.RData")

# plot
load("Elastic_Net_CorrDesign_ALO.RData")
result$alpha=factor(result$alpha)
p=ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
bmp("Elastic_Net_CorrDesign_with_Intercept.bmp",width=1280,height=720)
p
dev.off()


# Elastic Net without Intercept ----------------------------------------------

# misspecification --------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(5E-2),length.out=50)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
  }
}
set.seed(1234)

# simulation
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
intercept=0
X=matrix(rnorm(n*p,mean=0,sd=sqrt(1/k)),ncol=p)
sigma=rnorm(n,mean=0,sd=0.5)
y=intercept+X%*%beta+sigma
index=which(y>=0)
y[index]=sqrt(y[index])
y[-index]=-sqrt(-y[-index])

# true leave-one-out
y.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
library(foreach)
library(doParallel)
no_cores=detectCores()-1
cl=makeCluster(no_cores)
registerDoParallel(cl)
for(i in 1:n) {
  # do leave one out prediction
  y.temp<-foreach(k=1:length(alpha),.combine=cbind,.packages='glmnet') %dopar%
    Elastic_Net_LOO(X,y,i,alpha[k],lambda,intercept=FALSE)
  # save the prediction value
  y.loo[i,]=y.temp
  # print middle result
  if(i%%10==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo=1/n*colSums((y.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,y.loo,
     file="Elastic_Net_Misspec_without_Intercept_LOO.RData")

# approximate leave-one-out
load('Elastic_Net_Misspec_without_Intercept_LOO.RData')
# compute the scale parameter for y
sd.y=as.numeric(sqrt(var(y)*length(y)/(length(y)-1)))
y.scaled=y/sd.y
X.scaled=X/sd.y
# find the ALO prediction
y.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X.scaled,y=y.scaled,family="gaussian",
               alpha=alpha[k],lambda=lambda/sd.y^2,thresh=1E-14,
               intercept=FALSE,standardize=FALSE,maxit=1000000)
  # find the prediction for each alpha value
  y.temp<-foreach(j=1:length(lambda),.combine=cbind) %do% {
    ElasticNetALO(as.vector(model$beta[,j]),model$a0[j]*sd.y,X,y,lambda[j],alpha[k])
  }
  y.alo[,((k-1)*length(lambda)+1):(k*length(lambda))]=y.temp
  # print middle result
  print(paste(k," alphas have beed calculated. ",
              "On average, every alpha needs ",
              round((proc.time()-starttime)[3]/k,2)," seconds."))
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((y.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,y.loo,y.alo,
     file="Elastic_Net_Misspec_without_Intercept_ALO.RData")

# plot
load("Elastic_Net_Misspec_without_Intercept_ALO.RData")
result$alpha=factor(result$alpha)
p=ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
bmp("Elastic_Net_Misspec_without_Intercept.bmp",width=1280,height=720)
p
dev.off()


# heavy-tailed noise ------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(4E-2),length.out=50)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
  }
}
set.seed(1234)

# simulation
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
intercept=0
X=matrix(rnorm(n*p,mean=0,sd=sqrt(1/k)),ncol=p)
sigma=rt(n,df=3)*sqrt(0.25/3)
y=intercept+X%*%beta+sigma

# true leave-one-out
y.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
library(foreach)
library(doParallel)
no_cores=detectCores()-1
cl=makeCluster(no_cores)
registerDoParallel(cl)
for(i in 1:n) {
  # do leave one out prediction
  y.temp<-foreach(k=1:length(alpha),.combine=cbind,.packages='glmnet') %dopar%
    Elastic_Net_LOO(X,y,i,alpha[k],lambda,intercept=FALSE)
  # save the prediction value
  y.loo[i,]=y.temp
  # print middle result
  if(i%%10==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo=1/n*colSums((y.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,y.loo,
     file="Elastic_Net_HTN_without_Intercept_LOO.RData")

# approximate leave-one-out
load('Elastic_Net_HTN_without_Intercept_LOO.RData')
# compute the scale parameter for y
sd.y=as.numeric(sqrt(var(y)*length(y)/(length(y)-1)))
y.scaled=y/sd.y
X.scaled=X/sd.y
# find the ALO prediction
y.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X.scaled,y=y.scaled,family="gaussian",
               alpha=alpha[k],lambda=lambda/sd.y^2,thresh=1E-14,
               intercept=FALSE,standardize=FALSE,maxit=1000000)
  # find the prediction for each alpha value
  y.temp<-foreach(j=1:length(lambda),.combine=cbind) %do% {
    ElasticNetALO(as.vector(model$beta[,j]),model$a0[j]*sd.y,X,y,lambda[j],alpha[k])
  }
  y.alo[,((k-1)*length(lambda)+1):(k*length(lambda))]=y.temp
  # print middle result
  print(paste(k," alphas have beed calculated. ",
              "On average, every alpha needs ",
              round((proc.time()-starttime)[3]/k,2)," seconds."))
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((y.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,y.loo,y.alo,
     file="Elastic_Net_HTN_without_Intercept_ALO.RData")

# plot
load("Elastic_Net_HTN_without_Intercept_ALO.RData")
result$alpha=factor(result$alpha)
p=ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
bmp("Elastic_Net_HTN_without_Intercept.bmp",width=1280,height=720)
p
dev.off()


# correlated design -------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(4E-2),length.out=50)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
  }
}
set.seed(1234)

# define Toeplitz matrix
rho=0.8
C=matrix(nrow=p,ncol=p)
for(i in 1:p)
{
  C[i,1:i]=rho^seq(i,1,by=-1)
  if(i==p)
    break
  else
    C[i,(i+1):p]=rho^seq(2,p-i+1,by=1)
}

# simulation
library(MASS)
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
intercept=0
X=mvrnorm(n,mu=rep(0,p),Sigma=C/k)
sigma=rnorm(n,mean=0,sd=0.5)
y=intercept+X%*%beta+sigma

# true leave-one-out
y.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
library(foreach)
library(doParallel)
no_cores=detectCores()-1
cl=makeCluster(no_cores)
registerDoParallel(cl)
for(i in 1:n) {
  # do leave one out prediction
  y.temp<-foreach(k=1:length(alpha),.combine=cbind,.packages='glmnet') %dopar%
    Elastic_Net_LOO(X,y,i,alpha[k],lambda,intercept=FALSE)
  # save the prediction value
  y.loo[i,]=y.temp
  # print middle result
  if(i%%10==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo=1/n*colSums((y.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,y.loo,
     file="Elastic_Net_CorrDesign_without_Intercept_LOO.RData")

# approximate leave-one-out
load('Elastic_Net_CorrDesign_without_Intercept_LOO.RData')
# compute the scale parameter for y
sd.y=as.numeric(sqrt(var(y)*length(y)/(length(y)-1)))
y.scaled=y/sd.y
X.scaled=X/sd.y
# find the ALO prediction
y.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X.scaled,y=y.scaled,family="gaussian",
               alpha=alpha[k],lambda=lambda/sd.y^2,thresh=1E-14,
               intercept=FALSE,standardize=FALSE,maxit=1000000)
  # find the prediction for each alpha value
  y.temp<-foreach(j=1:length(lambda),.combine=cbind) %do% {
    ElasticNetALO(as.vector(model$beta[,j]),model$a0[j]*sd.y,X,y,lambda[j],alpha[k])
  }
  y.alo[,((k-1)*length(lambda)+1):(k*length(lambda))]=y.temp
  # print middle result
  print(paste(k," alphas have beed calculated. ",
              "On average, every alpha needs ",
              round((proc.time()-starttime)[3]/k,2)," seconds."))
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((y.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,y.loo,y.alo,
     file="Elastic_Net_CorrDesign_without_Intercept_ALO.RData")

# plot
load("Elastic_Net_CorrDesign_ALO.RData")
result$alpha=factor(result$alpha)
p=ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
bmp("Elastic_Net_CorrDesign_without_Intercept.bmp",width=1280,height=720)
p
dev.off()