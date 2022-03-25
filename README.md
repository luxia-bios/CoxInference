# CoxInference

This repository documents the R code accompanying the article "Statistical Inference for Cox Proportional Hazards Models with a Diverging Number of Covariates" by Lu Xia, Bin Nan and Yi Li (2022+). 

Citation: Xia, L., Nan, B. and Li, Y. (2022+). Statistical Inference for Cox Proportional Hazards Models with a Diverging Number of Covariates. Scandinavian Journal of Statistics, in press.

The R code file `CoxInference.R` provides a simulation example with known true signals and simulated survival data, where the proposed de-biasing lasso approach via solving quadratic programming problems (QP) is applied to draw inference on all regression coefficients in the Cox model. Detailed explanations are available in the comment lines within the code. Please make sure to use `sourceCpp` to include the `Rcpp` file `sim_univLib.cpp`, which contains faster implementation of the computation of the log partial likelihood, its derivatives and the Sigma matrix (defined in the article) for Cox models. 
