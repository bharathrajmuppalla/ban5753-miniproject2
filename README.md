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
![image](https://user-images.githubusercontent.com/89102349/202362941-a922a999-14b2-4388-bfe8-722a2d9306ac.png)
* ![image](https://user-images.githubusercontent.com/89102349/202363143-1db359db-5ad2-4f36-b9b1-ac1674362e2d.png)
Looking at the screenshot presented above, we could see the target variable has been imbalanced. To tackle the issue of class imbalance, we used Synthetic Minority Over-sampling Technique (SMOTE). Below is the target variable distribution before and after applying the technique
![image](https://user-images.githubusercontent.com/89102349/202363179-53770c2e-4830-454b-ac61-65d7012d51b7.png)
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





