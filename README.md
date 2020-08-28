# cancerprediction
In this data set, I work to improve the prediction of a logistic regression algorithm to predict cancer given 9 different features of a patient. To visualize the data on a 2D scale, I use Principal Component Analysis. I use a Training set, a Validation set and a Test set to find the accuracy, draw learning curves and improve the model. The learning curves are drawn based on the Training set and the Validation set. While accuracy is calculated using the Test set. Each branch is a new approach to tr to improve the model.
Initially, by using a linear hypothesis function, application of the theta gives only an accuracy of 82.6%. We work to improve this. 
What has been done in the branch set;
1) Using Principal component analysis, we have selected 2 principal components to reduce the 9 dimensional data to 2 dimensions and it is presented in the PCA figure. These 2 axes capture 99% of the data. (To be exact 99.355% according to the code and data I have used).
2) Drew a learning curve. Shows that the validation cost at the end is lower than the training cost which according to this thread (https://stats.stackexchange.com/questions/187335/validation-error-less-than-training-error/187404#187404), is a sign of overfitting. 
Hence my next step is to work the regularization parameter lambda to try to reduce the overfit.
