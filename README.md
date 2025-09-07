<div align="right">
  
https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/edit/main/README.md

</div>


# <div align="center">Telecom Customer Churn Prediction</div>



## What is Customer Churn?
Customer churn is defined as when customers or subscribers discontinue doing business with a firm or service.

Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next. The telecommunications business has an annual churn rate of 15-25 percent in this highly competitive market.

Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these "high risk" clients. The ultimate goal is to expand its coverage area and retrieve more customers loyalty. The core to succeed in this market lies in the customer itself.

Customer churn is a critical metric because it is much less expensive to retain existing customers than it is to acquire new customers.

To detect early signs of potential churn, one must first develop a holistic view of the customers and their interactions across numerous channels.As a result, by addressing churn, these businesses may not only preserve their market position, but also grow and thrive. More customers they have in their network, the lower the cost of initiation and the larger the profit. As a result, the company's key focus for success is reducing client attrition and implementing effective retention strategy.
## Objectives:
- Finding the % of Churn Customers and customers that keep in with the active services.
- Analysing the data in terms of various features responsible for customer Churn
- Finding a most suited machine learning model for correct classification of Churn and non churn customers.

## Dataset:
 [Telco Customer Churn](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data)

### The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents
## Implementation:

**Libraries:** sklearn, Matplotlib, pandas, seaborn, and NumPy



## Few glimpses of EDA:
### 1. Churn distribution:

> ![Churn distribution]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Churn%20Distribution.png
> 26.6 % of customers switched to another firm.

### 2. Churn distribution with respect to gender:
> ![Churn distribution wrt Gender]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/distributionWRTGender.PNG
> There is negligible difference in customer percentage/count who chnaged the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.`

### 3. Customer Contract distribution:
> ![Customer contract distribution]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Contract%20distribution.png
> About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customrs with One Year Contract and 3% with Two Year Contract

### 4. Payment Methods:
> ![Distribution of Payments methods]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/payment%20ethods%20with%20respectto%20churn.PNG
> Major customers who moved out were having Electronic Check as Payment Method.
> Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.

### 5. Internet services:

> Several customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate, this might suggest a dissatisfaction with this type of internet service.
> Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service.
![Churn distribution w.r.t Internet services and Gender]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/internet%20services.PNG
### 6. Dependent distribution:

> Customers without dependents are more likely to churn.
![Churn distribution w.r.t dependents]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/dependents.PNG
### 7. Online Security:

> As shown in following graph, most customers churn due to lack of online security
![Churn distribution w.r.t online security]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/onlineSecurity.PNG

### 8. Senior Citizen:

> Most of the senior citizens churn; the number of senior citizens are very less in over all customer base.
![Churn distribution w.r.t Senior Citizen]
### 9. Paperless Billing:

> Customers with Paperless Billing are most likely to churn.
![Churn distribution w.r.t mode of billing]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/billing.PNG
### 10. Tech support:

> As shown in following chart, customers with no TechSupport are most likely to migrate to another service provider.
![Churn distribution w.r.t Tech support]
### 11. Distribution w.r.t Charges and Tenure:
> ![Monthly Charges]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/carges%20distribution.PNG
> ![Total Charges]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/total%20charges.PNG
> ![Tenure]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/tenure%20and%20churn.PNG
> Customers with higher Monthly Charges are also more likely to churn.<br>
> New customers are more likely to churn.

## Machine Learning Model Evaluations and Predictions:
![ML Algorithms
#### Results after K fold cross validation:
[logistic Regression]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/LR.PNG
[KNN](https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/KNN.PNG
[NAVIE BAYES]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Naive%20Bayes.PNG
[DECISION TREE]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Decision%20trees.PNG
[RANDOM FOREST]https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Random%20Forest.PNG
[ADABOOST](https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Adaboost.PNG

[GRADIENT BOOST](https://github.com/Shubhanshu007iit/Customer-Churn-Analyzer/blob/main/Gradient%20boost.PNG

[VOTING CLASS](


[CONFUSION MATRIX](


#### Final Model: Voting Classifier
* We have selected Gradient boosting, Logistic Regression, and Adaboost for our Voting Classifier.
```
    from sklearn.ensemble import VotingClassifier
    clf1 = GradientBoostingClassifier()
    clf2 = LogisticRegression()
    clf3 = AdaBoostClassifier()
    eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
    eclf1.fit(X_train, y_train)
    predictions = eclf1.predict(X_test)
    print("Final Accuracy Score ")
    print(accuracy_score(y_test, predictions))
```
```
Final Score 
{'LogisticRegression': [0.841331397558646, 0.010495252078550477],
 'KNeighborsClassifier': [0.7913242024807321, 0.008198993337848612],
 'GaussianNB': [0.8232386881685605, 0.00741678015498337],
 'DecisionTreeClassifier': [0.6470213137060805, 0.02196953973039052],
 'RandomForestClassifier': [0.8197874155380965, 0.011556155864106703],
 'AdaBoostClassifier': [0.8445838813774079, 0.01125665302188384],
 'GradientBoostingClassifier': [0.844630629931458, 0.010723107447558198],
 'VotingClassifier': [0.8468096379573085, 0.010887508320460332]}

```
* Final confusion matrix we got:
<img src=  width = "425" />

>From the confusion matrix we can see that: There are total 1383+166=1549 actual non-churn values and the algorithm predicts 1400 of them as non churn and 149 of them as churn. While there are 280+280=561 actual churn values and the algorithm predicts 280 of them as non churn values and 281 of them as churn values.
## Optimizations

We could use Hyperparamete Tuning or Feature enginnering methods to improve the accuracy further.


### Feedback

If you have any feedback, please reach out at shubhanshu292@gmail.com



### 🚀 About Me
#### Hi, I'm Shubhanshu! 👋
I am an AI Enthusiast and  Data science & ML practitioner




