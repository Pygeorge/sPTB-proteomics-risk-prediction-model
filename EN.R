#R version 4.2.1 (2022-06-23 ucrt)
library(glmnetUtils) # version 1.1.8
library(ROCR)		 # version 1.0-11
library(nnls)		 # version 1.4
library(glmnet)		 # version 4.1-4
library(broom)		 # version 1.0.0
library(pROC)		 # version 1.18.0
library(MASS)		 # version 7.3-58.1
library(Matrix)		 # version 1.5-1
library(leaps)		 # version 3.1


#####################################
# Data
data_dir = 'data_path_for_training_set'
setwd(data_dir)


proteome_data <- read.csv(file="proteome_data.csv", check.names = FALSE)

sampleid <- proteome_data$SampleId


###########################################
# SAME FOR PROTEOME: MODEL WITH 10 COVARIATES
############################################

number_protein <- seq(2, 50) # chosen number of covariates
number_protein <- seq(2, 5) # chosen number of covariates
roc_select <- numeric(length(number_protein))
roc_select_CI_lower <- numeric(length(number_protein))
roc_select_CI_upper <- numeric(length(number_protein))
model_select <- list()


for (k in 1:length(number_protein)){
  
  random_int <- sample(1:10000, 1)
  set.seed(random_int)
  
  # Initialize the result container
  result <- proteome_data[,c("SampleId", "PTB")]
  result$lambda_min <- 0
  
  
  for (i in 1:length(sampleid)){	
    
    leave_one_out_sample <- sampleid[c(i)]  

    # dataset split
    ######################################
    
    datatrain <- proteome_data[!(proteome_data$SampleId == leave_one_out_sample),]
    ptb_train <- proteome_data$PTB[!(proteome_data$SampleId== leave_one_out_sample)]
    datatest <- proteome_data[proteome_data$SampleId == leave_one_out_sample,]
    

    datatrain$PTB <- NULL
    datatrain$SampleId <- NULL

    datatest$PTB <- NULL
    datatest$SampleId  <- NULL


    
    fit_proteome <- cv.glmnet(as.matrix(datatrain), ptb_train, family = "binomial",  type.measure = "auc", alpha=0.5, nfolds=5)
    plot(fit_proteome)
    model_ptb <- tidy(coef(fit_proteome))
    
    model_ptb <- model_ptb[-1,] #remove intercept
    model_ptb[model_ptb[, 1] != 0, ]
    
    # Choose top k proteins
    model_ptb$absvalue <- abs(model_ptb$value)
    model_ptb <- model_ptb[order(model_ptb$absvalue, decreasing=TRUE),]
    top_k_modelptb <- model_ptb[c(1:number_protein[k]),]
    top_k_modelptb <-top_k_modelptb[!is.na(top_k_modelptb$column),]
    
    
    model_select[length(model_select)+1] <- list(model_ptb$row)

    
    # model construction with selected proteins
    select_proteins_datatrain <- subset(datatrain, select = c(top_k_modelptb$row))
    finalmodel_proteome <-  cv.glmnet(as.matrix(select_proteins_datatrain), ptb_train, family="binomial", type.measure = "auc", alpha=0.5, nfolds=5)

    
    # Calculate scores for sample i
    select_proteins_datatest <- subset(datatest, select = c(top_k_modelptb$row))
    lambda_min_ <- predict(finalmodel_proteome, as.matrix(select_proteins_datatest), s = "lambda.min",  type='response', alpha=0.5)
    
    result$lambda_min[result$SampleId==leave_one_out_sample] <-  lambda_min_
    
  }

  # cal AUC lower/upper 95% CI
  roc_topk_proteome <- roc(result$PTB, result$lambda_min, ci=TRUE, quiet=TRUE, direction="<")
  roc_select[k] <- as.numeric(roc_topk_proteome$auc)
  roc_select_CI_lower[k] <- as.numeric(roc_topk_proteome$ci[1])
  roc_select_CI_upper[k] <- as.numeric(roc_topk_proteome$ci[3])
  print(roc_select)
  print(roc_select_CI_lower)
  print(roc_select_CI_upper)

}


# plot EN selection 
plot(number_protein, roc_select, type="b",col="#F57C00", lwd=2, xlim=c(0,40), ylim=c(0.6, 1))
legend("bottomright", legend=c("Urine metabolome", "Proteome"), col = c("#3F51B5", "#F57C00"), lty = 1.5, cex = 1.5)

# save EN result
auc_ci = c(list(roc_select), list(roc_select_CI_lower), list(roc_select_CI_upper))
auc_ci_pd = as.data.frame(auc_ci)
rownames(auc_ci_pd) <- number_protein
write.table(as.data.frame(auc_ci), "EN_AUC_ci.txt", sep='\t', row.names = TRUE, col.names = TRUE)


selection_proteins <- as.data.frame(table(unlist(model_select)))



