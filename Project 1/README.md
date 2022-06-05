# Breast Cancer Prediction <br>

## Installations <br>
1. Numpy
2. Pandas
3. Seaborn
4. Matplotlib

No additional installations required beyond Anaconda distribution of Python and Jupyter Notebooks 

## Overview & Motivation <br>

Breast Cancer is the most often identified cancer among women and major reason for increasing mortality rate among women. As the diagnosis of this disease manually takes long hours and the lesser availability of systems, there is a need to develop the automatic diagnosis system for early detection of cancer.
Data mining and machine learning techniques contribute a lot in the development of such system. <br>

With this motivation in mind, we perform data analysis of Breast Cancer Wisconsin (Diagnostic) Data Set available on Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download). In this dataset, features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. <br>

More information about how the data was collected and its creators can be found here : [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
<br>

For my analysis before application of any machine learning techniques, I am interested to find answers to the following questions:<br>
1. a. Is the radius of the cells different for Benign(non- cancerous) and Malignant (cancerous) cells? <br>
   b. Is the symmetry of the cells different for Benign and Malignant cells?<br>
2. How are the features distributed in the dataset?<br>
3. Which features are highly correlated to each other? <br>
4. Which are the most important features that can predict if the tumour cell is Benign or Malignant in nature?

For the classification of benign and malignant tumour cells I have used classification technique (Logistic Regression) of machine learning.

## Result Summary <br>
My analysis shows that some attributes like mean_radius of the cells are important for prediction between Benign and Malignant cells. The dataset is a mix of some features which are highly correlated with each other while others have with 0 correlation. Most of the features are measured in different scales and are (mostly) not evenly distributed. These issues have been fixed and addressed before the implementation of Logistic Regression, which gives a precision score of 95.03% .

