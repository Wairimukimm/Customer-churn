# Customer-churn
![image](https://github.com/Wairimukimm/Customer-churn/assets/122815372/919e394d-e96a-41e6-badb-e5739e32aa7d)




## Overview
This project consist of supervised machine learning. It makes use of classification algorithms to predict the likelihood of a customer to churn in the banking industry




## Business and Data Understanding
The stakeholders of this project are banks. Retaining customers is crucial as it builds trust and prevents losses. Customer churn is crucial in the banking industry. Detecting a Customer who is about to churn can help banks to apply strategies to retain the Customer. It can also help them know where and how to improve their servives.
The dataset used here is approriate as it contains binary outcomes, that is 0 and 1.
## Problem Statement
Banks have a hard time to retain customers due to challenges such as different pricing strategies. Retaining customers is very crucial because it is cheaper than trying to acquire new ones. Their concern is trying to identify customers who are about to churn. The current lack of a model that can classify customers can result to losses. To address the challenges , I aim to develop a model that can predict the likelihood of user churn in the near future. This will help the bank come up with specific targeted strategies to minimize customer churn and hence enhance their relationships with a customer hence continue making profits
## Components
* The [Jupyter Notebook](https://github.com/Wairimukimm/Customer-churn/blob/main/Untitled.ipynb) is the main deliverable. It contains the details of the approach taken and the methodology. It contains data cleaning, exploratory data analysis, data preparation for modelling and building the classification models.

* The [Presentation](https://github.com/Wairimukimm/Customer-churn/blob/main/presentation.pdf) is the non technical Presentation of the project. It contains the objectives, problem statment, model evaluation, findings and recommendations.

* The dataset used for this project can be found in [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn?source=post_page-----fa6e2324c245--------------------------------)




## Technologies
* Python version: 3.6.9
* Matplotlib version: 3.1.3
* Seaborn version: 0.9.0
* Pandas version: 0.25.1
* Numpy version: 1.16.5
* Statsmodels version: 0.10.1
* Scikit-learn version: 0.21.2
    
## To begin

* Clone this [repository](https://github.com/Wairimukimm/Customer-churn)
* Download the [Dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn?source=post_page-----fa6e2324c245--------------------------------)
* Check the technologies used and install if necessary


## Data Wrangling
In this section, I did data preparation which invloved:
* Checking for missing values, removing duplicates, renaming columns and dropping unnecessary columns to ensure the data is clean and suitable for analysis and modelling.



## Explaratory Data Analysis(EDA)
I perfomed both univariate and bivariate analysis to uncover patterns in the dataset. 
* Customers who complain have a very high chance of churning.
* Females are more likely to churn than men
* Customers with few products are more likely to churn


## Decision Tree(Baseline Model)

This was my baseline model. I used Decision Trees because of its simplicity and the fact that it can capture complex relationships.

* Recall Score : 0.67
* Accuracy of the model: 0.74

Recall: Recall calculates the proportion of correctly identified churned customers among all actual churned customers. This is particularly critical for a bank, as failing to identify actual churned customers could result in significant revenue loss.

The recall for the decision tree model is 67%. A recall score of 0.67 indicates that the model correctly identified approximately 67% of the total positive instances (churned customers) in the datase


## Random Forest Classifier
I settled for Random Forest because they provide a higher accuracy than decision trees due to their ensemble nature.

* Recall Score : 0.70
* Accuracy of the model: 0.81

Recall: Recall calculates the proportion of correctly identified churned customers among all actual churned customers. This is particularly critical for a bank, as failing to identify actual churned customers could result in significant revenue loss.

 Accuracy: Overall model accuracy is 81%, indicating that the model correctly predicted the churn or no-churn status for approximately 81% of all customers.
## Gradient Boosting Classifier

My final model was gradient boosting because it allows for Predictive Accuracy: Gradient Boosting tends to provide better predictive accuracy compared to Random Forests in many cases, especially when the data is not too noisy and when there are complex relationships between features and target variables.

* Recall Score : 0.72
* Accuracy of the model: 0.82

Recall: Recall calculates the proportion of correctly identified churned customers among all actual churned customers. This is particularly critical for a bank, as failing to identify actual churned customers could result in significant revenue loss.

 Accuracy: Overall model accuracy is 82%, indicating that the model correctly predicted the churn or no-churn status for approximately 81% of all customers.

## Conclusions
In conclusion, the customer churn prediction model aims to provide valuable insights into customer behavior in the context of the banking industry. Given the financial implications of customer churn, it's crucial to refine the model to accurately identify customers at risk of leaving. Throughout the iterative process, various changes were applied to enhance model performance. These include adjusting hyperparameters, building multiple models, and addressing imbalanced data to improve the model's predictive ability. The rationale behind these alterations was to iteratively fine-tune the model for better accuracy and recall. These results indicate that the gradient boost model performed well in terms of accuracy and recall. 
 


