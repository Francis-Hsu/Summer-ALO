# Elastic Net with Intercept
setwd("E:\\Columbia_University\\Internship\\R_File\\")
library(glmnet)
library(ggplot2)

# misspecification --------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(4E-2),length.out=40)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
    param$lambda.print[j+(i-1)*length(lambda)]=lambda[j]+max(lambda)*1.1*(i-1)
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


# True Leave-One-Out ------------------------------------------------------

prediction.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(i in 1:n) {
  for(k in 1:length(alpha)) {
    # build the model
    model=glmnet(x=X[-i,],y=y[-i],family="gaussian",alpha=alpha[k],thresh=1E-14,
                 intercept=TRUE,standardize=FALSE,lambda=lambda)
    # prediction
    prediction.loo[i,((k-1)*length(lambda)+1):(k*length(lambda))]=
      predict(model,newx=matrix(X[i,],nrow=1),type="response",s=lambda)
    # print result
    cat(i,',',k,'\n')
  }
  # print middle result
  if(i%%100==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
# true leave-one-out risk estimate
risk.loo=1/n*colSums((prediction.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,prediction.loo,
     file="Elastic_Net_LOO.RData")


# ALO - Primal Domain -----------------------------------------------------

load('Elastic_Net_LOO.RData')
prediction.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X,y=y,family="gaussian",alpha=alpha[k],thresh=1E-14,
               intercept=TRUE,standardize=FALSE,lambda=lambda)
  for(j in 1:length(lambda)) {
    # find beta
    beta.hat=predict(model,type='coefficients',s=lambda[j],exact=TRUE,x=X,y=y)
    intercept.hat=beta.hat[1]
    beta.hat=beta.hat[2:length(beta.hat)]
    # find the active set
    if(alpha[k]==0) {
      A=seq(1,p+1)
    } else {
      A=c(1,which(beta.hat!=0)+1)
    }
    # define the full data
    X.full=cbind(1,X)
    # compute matrix H
    if(length(A)>1) {
      H=X.full[,A]%*%
        solve(t(X.full[,A])%*%diag(rep(1,n))%*%X.full[,A]+
                diag(c(0,rep(n*lambda[j]*(1-alpha[k]),length(A)-1))))%*%t(X.full[,A])
    } else if(length(A)==1) {
      H=X.full[,A]%*%
        solve(t(X.full[,A])%*%diag(rep(1,n))%*%X.full[,A])%*%t(X.full[,A])
    }
    
    # ALO prediction
    prediction.alo[,(j+(k-1)*length(lambda))]=
      intercept.hat+X%*%beta.hat+
      diag(H)*(intercept.hat+X%*%beta.hat-y)/(1-diag(H))
  }
  print(k)
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((prediction.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,prediction.loo,prediction.alo,
     file="Elastic_Net_ALO.RData")

# plot
result$alpha=factor(result$alpha)
ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)

# ALO - Proximal Operator -------------------------------------------------

prediction.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X,y=y,family="gaussian",alpha=alpha[k],thresh=1E-14,
               intercept=TRUE,standardize=FALSE,lambda=lambda)
  for(j in 1:length(lambda)) {
    # find beta
    beta.hat=predict(model,type='coefficients',s=lambda[j],exact=TRUE,x=X,y=y)
    intercept.hat=beta.hat[1]
    beta.hat=beta.hat[2:length(beta.hat)]
    # compute theta
    theta.hat=y-X%*%beta.hat-intercept.hat
    # define the full data
    X.full=cbind(1,X)
    # matrix H
    E=c(1,which(beta.hat!=0)+1)
    J=diag(rep(1,p+1)+c(0,rep(lambda[j]*(1-alpha[k]),p)))
    J=solve(J)
    if(length(E)>1) {
      H=X.full[,E]%*%solve(J[E,E]%*%t(X.full[,E])%*%X.full[,E]+
                        diag(rep(1,length(E)))-J[E,E])%*%J[E,E]%*%t(X.full[,E])
    } else {
      H=X.full[,E]%*%solve(t(X.full[,E])%*%X.full[,E])%*%t(X.full[,E])
    }
    # ALO prediction
    prediction.alo[,(j+(k-1)*length(lambda))]=
      intercept.hat+X%*%beta.hat+
      diag(H)*(intercept.hat+X%*%beta.hat-y)/(1-diag(H))
  }
}

# risk
risk.alo=1/n*colSums((prediction.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)





# Without Intercept -------------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\")
library(glmnet)
library(ggplot2)

# misspecification --------------------------------------------------------

# parameters
n=300
p=600
k=60
log10.lambda=seq(log10(1E-3),log10(4E-2),length.out=40)
lambda=10^log10.lambda
lambda=sort(lambda,decreasing=TRUE)
alpha=seq(0,1,0.1)
param=data.frame(alpha=numeric(0),lambda=numeric(0),lambda.print=numeric(0))
for(i in 1:length(alpha)) {
  for(j in 1:length(lambda)) {
    param[j+(i-1)*length(lambda),c('alpha','lambda')]=c(alpha[i],lambda[j])
    param$lambda.print[j+(i-1)*length(lambda)]=lambda[j]+max(lambda)*1.1*(i-1)
  }
}
set.seed(1234)

# simulation
beta=rnorm(p,mean=0,sd=1)
beta[(k+1):p]=0
X=matrix(rnorm(n*p,mean=0,sd=sqrt(1/k)),ncol=p)
sigma=rnorm(n,mean=0,sd=0.5)
y=X%*%beta+sigma
index=which(y>=0)
y[index]=sqrt(y[index])
y[-index]=-sqrt(-y[-index])


# True Leave-One-Out ------------------------------------------------------

prediction.loo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(i in 1:n) {
  for(k in 1:length(alpha)) {
    # build the model
    model=glmnet(x=X[-i,],y=y[-i],family="gaussian",alpha=alpha[k],thresh=1E-14,
                 intercept=FALSE,standardize=FALSE,lambda=lambda)
    # prediction
    prediction.loo[i,((k-1)*length(lambda)+1):(k*length(lambda))]=
      predict(model,newx=matrix(X[i,],nrow=1),type="response",s=lambda)
    # print result
    cat(i,',',k,'\n')
  }
  # print middle result
  if(i%%100==0)
    print(paste(i," samples have beed calculated. ",
                "On average, every sample needs ",
                round((proc.time()-starttime)[3]/i,2)," seconds."))
}
# true leave-one-out risk estimate
risk.loo=1/n*colSums((prediction.loo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(param,risk.loo)
# save the data
save(result,prediction.loo,
     file="Elastic_Net_LOO_WithoutIntercept.RData")


# ALO - Primal Domain -----------------------------------------------------

load('Elastic_Net_LOO_WithoutIntercept.RData')
prediction.alo=matrix(ncol=dim(param)[1],nrow=n)
starttime=proc.time() # count time
for(k in 1:length(alpha)) {
  # build the full data model
  model=glmnet(x=X,y=y,family="gaussian",alpha=alpha[k],thresh=1E-14,
               intercept=FALSE,standardize=FALSE,lambda=lambda)
  for(j in 1:length(lambda)) {
    # find beta
    beta.hat=predict(model,type='coefficients',s=lambda[j],exact=TRUE,x=X,y=y)
    intercept.hat=beta.hat[1]
    beta.hat=beta.hat[2:length(beta.hat)]
    # find the active set
    if(alpha[k]==0) {
      A=seq(1,p)
    } else {
      A=which(beta.hat!=0)
    }
    # compute matrix H
    if(length(A)>=2) {
      H=X[,A]%*%
        solve(t(X[,A])%*%diag(rep(1,n))%*%X[,A]+
                diag(rep(n*lambda[j]*(1-alpha[k]),length(A))))%*%t(X[,A])
    } else if (length(A)==1) {
      H=X[,A]%*%
        solve(t(X[,A])%*%diag(rep(1,n))%*%X[,A]+
                n*lambda[j]*(1-alpha[k]))%*%t(X[,A])
    }
      else if(length(A)==0) {
      H=diag(rep(0,n))
    }
    
    # ALO prediction
    prediction.alo[,(j+(k-1)*length(lambda))]=
      X%*%beta.hat+
      diag(H)*(X%*%beta.hat-y)/(1-diag(H))
  }
  print(k)
}
# true leave-one-out risk estimate
risk.alo=1/n*colSums((prediction.alo-
                        matrix(rep(y,dim(param)[1]),ncol=dim(param)[1]))^2)
# record the result
result=cbind(result,risk.alo)

# save the data
save(result,prediction.loo,prediction.alo,
     file="Elastic_Net_ALO_WithoutIntercept.RData")

# plot
load("Elastic_Net_ALO_WithoutIntercept.RData")
result$alpha=factor(result$alpha)
ggplot(result)+
  geom_line(aes(x=log10(lambda),y=risk.loo),lty=2)+
  geom_line(aes(x=log10(lambda),y=risk.alo),col="red",lty=2)+
  facet_wrap(~alpha,nrow=2)
