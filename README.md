# Overview of this Project
In this project, we built various machine learning models using Pyspark to identify clients, who will subscribe (yes/no) for a term deposit in the bank and finally selected the best model based on Accuracy. To recognize trends and relationships in the data we begin the project by conducting Exploratory Data Analysis (EDA) and after performing data preprocessing, we used machine learning models for classification such as Logistic Regression, Decision Tree Classifier, Random Forest Classifier and GBT Classifier to identify clients that’ll subscribe for a term deposit

## Steps followed in the Project
### Sample Data
We begin by reading the data into the pyspark environment and identifying the data types of columns
![image](https://user-images.githubusercontent.com/89102349/202362679-c2092488-fd88-4cd6-8fe4-bdb35cb42bde.png)
### Data Type of Variables
![image](https://user-images.githubusercontent.com/89102349/202362771-6b6d426b-6323-4275-84db-df7b2436a7cd.png)
### EDA ( Exploratory Data Analysis )
* Statistics

![image](https://user-images.githubusercontent.com/89102349/202362795-e34a6407-760c-4b4e-8de4-a1471d28d1d3.png)

* Distribution of Features

![image](https://user-images.githubusercontent.com/89102349/202362908-fab246c8-c0e0-427a-b9b2-e565ab949ec7.png)

* Check for Null Values
There are no null values for any given variable.

![image](https://user-images.githubusercontent.com/89102349/202363090-5b63cbac-5daf-4407-90fc-ff38cabfffbf.png)

* Correlation
We performed Pearson correlation to find the correlation between given variables, and the below table mentions the correlation values between any two given variables.

![image](https://user-images.githubusercontent.com/89102349/202364176-8b18dedf-9567-4919-b857-55a1673af6c3.png)

* Target Variable Distribution

![image](https://user-images.githubusercontent.com/89102349/202364241-8e69bec7-8bac-4d7f-96a2-570f735cd790.png)

Looking at the screenshot presented above, we could see the target variable has been imbalanced. To tackle the issue of class imbalance, we used Synthetic Minority Over-sampling Technique (SMOTE). Below is the target variable distribution before and after applying the technique

![image](https://user-images.githubusercontent.com/89102349/202364340-4904c38e-ef0f-495a-b896-4a218892ccf5.png)


### Data preprocessing
* String Indexer
Below is a list of indexers for all categorical columns

![image](https://user-images.githubusercontent.com/89102349/202363273-84d1e668-8839-4e1f-a36e-d9267ffa02f4.png)

* One Hot Encoder Estimator
We then created a list of encoders based on above indexers and tested pipeline with indexers + encoders

![image](https://user-images.githubusercontent.com/89102349/202363336-04b11bf8-700b-4510-b00d-6e953a4d95e3.png)

* Vector Assembler

![image](https://user-images.githubusercontent.com/89102349/202363377-1c20558d-8e2d-4f13-8b7c-b4bbb7835ba5.png)

* Label Indexer for output 'y'

![image](https://user-images.githubusercontent.com/89102349/202363427-219ee621-312f-4c77-be53-abc8a80026f9.png)

* Standard Scaler

![image](https://user-images.githubusercontent.com/89102349/202363470-e13bcf27-938f-4b4e-9eab-beecb62b38cf.png)

### Modeling
we split the data into training and testing datasets with training containing 80% and testing containing 20% of the data
* Model-1: Logistic Regression

![image](https://user-images.githubusercontent.com/89102349/202363613-c9f13223-7fdd-4c16-b045-56cdc8dc6ba9.png)

* Confusion Matrix
![image](https://user-images.githubusercontent.com/89102349/202363673-00ce8e2c-de5c-417e-addc-eb8069804704.png)


Sensitivity- 56.2%
Specificity- 95.5%
* Accuracy

![image](https://user-images.githubusercontent.com/89102349/202363746-656a70af-f265-47f6-950d-f767b9445be2.png)

* ROC Curve
![image](https://user-images.githubusercontent.com/89102349/202363768-78dbc09e-8164-457d-9044-956d22d9560c.png)

* Model 2: Decision Tree Classifier
![image](https://user-images.githubusercontent.com/89102349/202364910-202343a2-d40d-483f-9a12-8c8e18284d97.png)

* Confusion Matrix
![image](https://user-images.githubusercontent.com/89102349/202364841-a906c190-da89-4a9f-8e5b-4d0b5b19d7b1.png)


Sensitivity- 69.3%
Specificity- 93%

* Accuracy

![image](https://user-images.githubusercontent.com/89102349/202364805-a6ed5e31-ea00-4e28-b533-6f05885a1b10.png)

* ROC Curve
![image](https://user-images.githubusercontent.com/89102349/202364786-24a1d7ec-df82-4a94-9a13-769f50c86af5.png)

* Model 3: Random Forest Classifier

![image](https://user-images.githubusercontent.com/89102349/202364735-f6c61584-8a0a-4c7a-a6c9-c20181525276.png)

* Confusion Matrix
![image](https://user-images.githubusercontent.com/89102349/202364712-38cac9b2-d6a5-4d12-a566-3740ae6033bb.png)


Sensitivity- 44.1%
Specificity- 94%

* Accuracy

![image](https://user-images.githubusercontent.com/89102349/202364682-10a0bf40-92dd-48bc-bcf1-f291dfe4964f.png))

* ROC Curve
![image](https://user-images.githubusercontent.com/89102349/202364666-7470764d-fec1-4d9c-818d-f640a38e066d.png)

* Model-4: Gradient-boosted Tree Classifier

![image](https://user-images.githubusercontent.com/89102349/202364609-f2c14776-77f0-4926-8da2-f013f461faba.png)

* Confusion Matrix
![image](https://user-images.githubusercontent.com/89102349/202364587-6107a332-3e0a-4081-872d-b1eec4f0886d.png)


Sensitivity- 69%
Specificity- 93.6%

* Accuracy

![image](https://user-images.githubusercontent.com/89102349/202364550-11d831b2-f015-4cfd-8709-b4b5fbf215be.png)

* ROC Curve
![image](https://user-images.githubusercontent.com/89102349/202364537-aa22af71-7cf1-4161-91c7-a640312fc089.png))

### Model Comparison

![image](https://user-images.githubusercontent.com/89102349/202365311-441b6268-60fa-4f1d-95e9-7a002f7749d5.png)

Looking at the table displayed above, we could see that logistic regression is performing well in classifying the prediction of the clients who will subscribe (yes/no) for a term deposit. The next best model would be the GBT Classifier with almost the exact same accuracy as the logistic regression, along with a better AUC.

From this we can understand that logistic regression though simple, is a powerful algorithm to solve the binary classification problems which matches our mentioned dataset.

### K-Means Clustering Algorithm

As part of the K-Means clustering process, we have used the silhouette analysis to find the optimal number of clusters. The silhouette plot shows that the n_clusters value based upon the evaluating the silhouette scores which is calculated by the mean of the coefficient for each sample belonging to different clusters.
The graph below represents the values of the silhouette Coefficient

![image](https://user-images.githubusercontent.com/89102349/202365395-2ef6d132-9904-4f62-9c5c-d46ff340592c.png)

Looking at the graph, we could say that 7 is the ideal number of clusters for this model as the local maximum occurs there in the graph

![image](https://user-images.githubusercontent.com/89102349/202365441-025f72b0-eabf-41a6-9ba5-e53b8270492b.png)

Based on the number of cluster groups, we have assigned the cluster group number to every record.









