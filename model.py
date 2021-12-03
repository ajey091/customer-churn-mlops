import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'Churn_Modelling.csv')
X = df.iloc[:,3:13]
y = df.iloc[:,13]


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])

X['Geography']=label.fit_transform(X['Geography'])
X['Geography'].value_counts()

ct = ColumnTransformer([("CustomerId", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.35,random_state=42)
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
#Train Model  
neigh = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)
#Prediction
prediction = neigh.predict(X_test)
prediction1=pd.DataFrame(prediction)
prediction1.head()


#Accuracy
from sklearn import metrics
percent1 = metrics.accuracy_score(y_test, prediction)
percent1


# Now Using SVM Algorithm 
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=1)
classifier.fit(X_train,y_train)


#Predict
y_pred=classifier.predict(X_test)
prediction2=pd.DataFrame(y_pred)
prediction2.head()

#Accuracy
percent2 =metrics.accuracy_score(y_test, prediction2)
percent2

from sklearn.ensemble import RandomForestClassifier
classifier_4 = RandomForestClassifier(n_estimators=100) #warning 10 to 100
classifier_4.fit(X_train,y_train)

#Predict
y_randomfor=classifier_4.predict(X_test)
prediction3=pd.DataFrame(y_randomfor)
prediction3.head()

#Accuracy
percent3 = metrics.accuracy_score(y_test, prediction3)
percent3

#Using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
TeleTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)


TeleTree.fit(X_train,y_train)
y_predtree = TeleTree.predict(X_test)
prediction4=pd.DataFrame(y_pred)
prediction4.head()

#Accuracy
percent4 = metrics.accuracy_score(y_test,prediction4)
percent4

models = pd.DataFrame({'name_model':["KNN","SVM","Random Forest","Decision Trees"],\
                                'accuracy':[percent1,percent2,percent3,percent4]})


models.to_csv('results.txt', sep='\t', index=False)

models.plot.bar(x='name_model', y='accuracy', rot=0)
plt.ylim((0.6,1))
plt.xlabel('Model')
plt.savefig('accuracies.png',dpi=120)

