#######################################################
###  R code for high-dim inference on Cox models    ###
###  citation: Xia, Nan and Li. (2022+). "Statistical #
###  Inference for Cox Proportional Hazards Models with a Diverging Number of Covariates".
###  Scandinavian Journal of Statistics, in press.  ###

rm(list=ls())
library(quadprog)
library(mvtnorm)
library(Rcpp)
library(RcppArmadillo)
library(glmnet)
library(flare)
library(QUIC)
library(huge)
library(matrixcalc)
library(survival)
library(MASS)
library(lpSolve)

### source the cpp file, which contains code for fast computation of derivatives of log partial likelihood and Sigma matrix
sourceCpp("sim_univLib.cpp")

len_beta <- 11 # total number of beta_1 values from 0 to 2
idx_beta <- 5 # in this setting, beta_1 takes the 5th value, which is 0.8
jobid <- 1 # job ID for simulation replications; also used to determine the seed for data generation

###################
# parameter setup #
###################

n <- 500 # sample size
p <- 100 # covariate dimension
n_multi <- 30 # number of gamma_n values tried in cross-validation
multiplier_seq <- exp(seq(from=log(0.01), to=log(3), length.out=n_multi)) # sequence of gamma_n values
beta_level_seq <- seq(0, 2, length.out=len_beta) # sequence of values for beta_1

set.seed(2019) # set seed for generating true signal positions
n_lambda <- 100 # number of cross-validation for lasso in glmnet
nfold <- 5 # number of cv folds for lasso in glmnet
alpha <- sig_level <- 0.05
alpha_cv <- 0.1
v <- qnorm(sig_level/2, lower.tail=F)
tol <- 1.0e-6
maxiter <- 50000

### setup: four additional signals, covariance is AR(1) with rho=0.5
s0 <- 4
rho <- 0.5
struct <- "ar1"
covmat <- matrix(0, nrow=p, ncol=p) 
if(struct == "indep") {
  covmat <- diag(p)
} else if(struct == "ar1") {
  covmat <- rho^(abs(outer(1:p,1:p,"-")))
} else if(struct == "cs") {
  covmat <- rho*rep(1,p)%*%t(rep(1,p)) + (1-rho)*diag(p)
} else if(struct == "invcs") {
  covmat <- rho*rep(1,p)%*%t(rep(1,p)) + (1-rho)*diag(p)
  covmat <- solve(covmat)
  covmat <- diag(1/sqrt(diag(covmat)))%*%covmat%*%diag(1/sqrt(diag(covmat)))
}
large_signal <- 1 
small_signal <- 0.5
beta_true <- rep(0,p)
beta_true[sample(2:p, size=s0)] <- c(rep(small_signal,s0/2), rep(large_signal,s0/2))
beta_true[1] <- beta_level_seq[idx_beta]
signal_pos <- which(beta_true != 0)
if(beta_true[1]!=0) {
  signal_pos2 <- which(beta_true != 0)
} else {
  signal_pos2 <- c(1,which(beta_true != 0))
}



#########################################
pre_time <- proc.time()
set.seed(123*jobid + 456*idx_beta + 789) # set seed for each replication
#########################################


#########################
#### data generation ####
#########################

baseline <- 1 
X <- rmvnorm(n, mean = rep(0,p), sigma = covmat) # first simulate covariates in mvnorm
X <- ifelse(abs(X)>2.5, sign(X)*2.5, X) # then truncated at +-2.5
latent_t <- rexp(n, rate=baseline*exp(X%*%beta_true)) # latent survival time
latent_c <- runif(n,1,20) # latent censoring time
time <- ifelse(latent_t <= latent_c, latent_t, latent_c) # observed time
delta <- as.numeric(latent_t <= latent_c); rate_censor <- mean(1-delta); rate_censor # event indicator

## have to re-order data as time increases ##
X <- X[order(time),]
delta <- delta[order(time)]
time <- time[order(time)]

# cross-validation for lasso estimator
cvobj_glmnet <- cv.glmnet(x=X, y=cbind(time=time, status=delta), family="cox", 
                          alpha=1, standardize=F, nfolds = nfold,
                          nlambda=n_lambda)
# obtain lasso estimator
beta_glmnet <- as.vector(coef(glmnet(x=X, y=cbind(time=time, status=delta), 
                                     family="cox", alpha=1, lambda=cvobj_glmnet$lambda.min, standardize=F, 
                                     thresh=tol, maxit=maxiter)))

# calculate 
neg_loglik_glmnet <- 0 # log PL
neg_dloglik_glmnet <- rep(0,p) # first-order derivative of log PL
neg_ddloglik_glmnet <- matrix(0, nrow=p, ncol=p) # second-order derivative of log PL
score_sq <- matrix(0, nrow=p, ncol=p) # Sigma matrix
# compute these using the function in .cpp file, evaluated at lasso estimator
neg_loglik_functions_cpp_ext(neg_loglik_glmnet, neg_dloglik_glmnet, neg_ddloglik_glmnet, score_sq,
                             X, time, delta, beta_glmnet)
r <- eigen(score_sq)
r$values[r$values<=1.0e-14] <- 0



################
## new method ##
################

### cross validation ###

n_cv <- 5 # number of cv folds for Theta matrix estimation in QP

# obtain random cv index
all_cv_idx <- rep(1:n_cv, n/n_cv)
all_cv_idx <- sample(all_cv_idx, size=n)
# define variable for cross-validation criterion values
all_cvpl2 <- array(NA, length(multiplier_seq)) # lik evaluation on left-out data

for(jj in 1:length(multiplier_seq)) { # for cvpl
  multiplier <- multiplier_seq[jj]
  
  # cv
  cv_idx <- NULL
  cvpl2 <- 0
  
  for(k in 1:n_cv) { # for cv
    cv_idx <- which(all_cv_idx==k) # test data
    train_x <- X[-c(cv_idx),]
    test_x <- X[cv_idx,]
    train_time <- time[-c(cv_idx)]
    test_time <- time[cv_idx]
    train_delta <- delta[-c(cv_idx)]
    test_delta <- delta[cv_idx]
    
    # use training data to obtain de-biased estimator at a given gamma_n
    cvobj_glmnet_train <- cv.glmnet(x=train_x, y=cbind(time=train_time, status=train_delta), family="cox",
                                    alpha=1, standardize=F, nfolds = nfold,
                                    nlambda=n_lambda)
    beta_glmnet_train <- as.vector(coef(glmnet(x=train_x, y=cbind(time=train_time, status=train_delta),
                                               family="cox", alpha=1, lambda=cvobj_glmnet_train$lambda.min, standardize=F,
                                               thresh=tol, maxit=maxiter)))
    neg_loglik_glmnet_train <- 0
    neg_dloglik_glmnet_train <- rep(0,p)
    neg_ddloglik_glmnet_train <- matrix(0, nrow=p, ncol=p)
    score_sq_train <- matrix(0, nrow=p, ncol=p)
    neg_loglik_functions_cpp_ext(neg_loglik_glmnet_train, neg_dloglik_glmnet_train, neg_ddloglik_glmnet_train,
                                 score_sq_train, train_x, train_time, train_delta, beta_glmnet_train)
    r_train <- eigen(score_sq_train)
    r_train$values[r_train$values<=1.0e-14] <- 0

    
    
    ###############################
    ### CV: QP (using solve.QP) ###
    ###############################
    
    b_hat_new <- rep(NA, p)
    se_new <- rep(NA, p)
    mu <- multiplier*sqrt(log(p)/n)
    my_pos <- which(r_train$values > 0)
    my_rank <- sum(r_train$values > 0)
    Dmat <- diag(r_train$values[my_pos])
    dvec <- rep(0,my_rank)
    Amat <- t(rbind(-r_train$vectors[,my_pos]%*%Dmat, r_train$vectors[,my_pos]%*%Dmat))
    for(j in 1:p) {
      e_j <- rep(0, p); e_j[j] <- 1
      bvec <- (c(-e_j, e_j) - mu*rep(1,2*p))
      res <- solve.QP(Dmat=Dmat, dvec=dvec, Amat=Amat, bvec=bvec)
      m <- as.vector(r_train$vectors[,my_pos]%*%res$solution) +
        as.vector(r_train$vectors[,-my_pos]%*%rep(0, p-my_rank))
      b_hat_new[j] <- beta_glmnet_train[j] - as.numeric(m%*%neg_dloglik_glmnet_train)
      se_new[j] <- sqrt(m[j]/nrow(train_x)) 
    }
    
    # obtain hard thresholded de-biased lasso estimator for de-noising for CV evaluation
    pval_new <- 2*pnorm(abs(b_hat_new/se_new), lower.tail = F)
    tmp_beta <- b_hat_new*as.numeric(pval_new < (alpha_cv/p))
    cvpl2 <- cvpl2 + loglik_cpp_ext(X=test_x, time=test_time, delta=test_delta, beta=tmp_beta)
    
  } # end for cv

  all_cvpl2[jj] <- cvpl2
  
} # end for cvpl


##### results using the chosen tuning parameter from all_cvpl2 #####

multiplier <- multiplier_seq[which.max(all_cvpl2)] # chosen multiplier

b_hat_new <- array(NA, p) # this is the final estimator b_hat
se_new <- array(NA, p) # this is the model-based SE for b_hat
theta_new <- matrix(NA, ncol=p, nrow=p) # estimated Theta matrix from QP approach
mu_new <- multiplier*sqrt(log(p)/n) # chosen tuning parameter gamma_n
my_pos <- which(r$values > 0)
my_rank <- sum(r$values > 0)
Dmat <- diag(r$values[my_pos])
dvec <- rep(0,my_rank)
Amat <- t(rbind(-r$vectors[,my_pos]%*%Dmat, r$vectors[,my_pos]%*%Dmat))
for(j in 1:p) {
  e_j <- rep(0, p); e_j[j] <- 1
  bvec <- (c(-e_j, e_j) - mu_new*rep(1,2*p))
  res <- solve.QP(Dmat=Dmat, dvec=dvec, Amat=Amat, bvec=bvec)
  m <- as.vector(r$vectors[,my_pos]%*%res$solution) + 
    as.vector(r$vectors[,-my_pos]%*%rep(0, p-my_rank))
  b_hat_new[j] <- beta_glmnet[j] - as.numeric(m%*%neg_dloglik_glmnet)
  se_new[j] <- sqrt(m[j]/n) 
  theta_new[j,] <- m
}
# coverage
cov_new <- as.numeric((beta_true <= (b_hat_new+v*se_new)) &
                        (beta_true >= (b_hat_new-v*se_new)))
# p-values
pval_new <- 2*pnorm(abs(b_hat_new/se_new), lower.tail=F)

