#!/usr/bin/python

from sys import path, exit
import pickle
path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'restricted_stock', 'from_poi', 'to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

all_features_list= data_dict[list(data_dict.keys())[0]]
# all_features_list.pop('poi')

all_features_list= list(all_features_list.keys())
# print(all_features_list)
# exit()

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

def outlierCleaner(predictions, features, labels):

    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = (labels-predictions)**2
    cleaned_data =zip(features,labels,errors)
    # for i in cleaned_data:
        # print(i[2])
    # exit()
    cleaned_data = sorted( cleaned_data ,key=lambda x:x[2], reverse=True)
    limit = int(len(labels)*0.1)
    
    return cleaned_data[limit:]

def checkCalssifierScore (clf, features_train,features_test,labels_train,labels_test, classifier_name):
    clf.fit(features_train, labels_train)
    # print(" Calssifier:  ", classifier_name)
    predNB= clf.predict(features_test)
    # print(predNB)

    a_score= accuracy_score(labels_test, predNB)
    r_score= recall_score(labels_test, predNB, zero_division= 0)
    p_score= precision_score(labels_test, predNB, zero_division= 0)

    # print(" accourcy score before clean outliers ", a_score )
    # print(" recall score before clean outliers ", r_score )
    # print(" precision score before clean outliers ", p_score )
    if( classifier_name == 'SVM' ):
        return a_score, r_score, p_score
        # exit()

    # return a_score, r_score, p_score
    cleaned_data = outlierCleaner( predNB, features_test, labels_test )
    # print(cleaned_data)
    new_features, new_labels, errors = zip(*cleaned_data)
    # print(new_features)
    new_features       = np.reshape( np.array(new_features), (len(new_features), (len(features_list) - 1)))
    # print(len(features_test))
    # print(len(new_features))
    # exit()
    new_labels = np.reshape( np.array(new_labels), (len(new_labels), 1))

    predNB= clf.predict(new_features)
    a_score= accuracy_score(new_labels, predNB)
    r_score= recall_score(new_labels, predNB, zero_division= 0)
    p_score= precision_score(new_labels, predNB, zero_division= 0)
    # print(" accourcy score after clean ", accuracy_score(new_labels, predNB) )
    # print(" recall score after clean ", r_score )
    # print(" precision score after clean ", p_score )
    return a_score, r_score, p_score



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
# print(my_dataset)
for person in my_dataset:
    if ('NaN' != my_dataset[person]['from_messages'] and 'NaN' != my_dataset[person]['from_this_person_to_poi'] ):
        my_dataset[person]['to_poi'] = my_dataset[person]['from_this_person_to_poi'] / my_dataset[person]['from_messages']
    else:
        my_dataset[person]['to_poi'] = 0

    if ('NaN' != my_dataset[person]['to_messages'] and 'NaN' != my_dataset[person]['from_poi_to_this_person'] ):
        my_dataset[person]['from_poi'] = my_dataset[person]['from_poi_to_this_person'] / my_dataset[person]['to_messages']
    else:
        my_dataset[person]['from_poi'] = 0

# print(my_dataset)
# exit()
### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

select_k = SelectKBest(f_classif, k= 'all' )

features = select_k.fit_transform(features, labels)

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.30,random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# clf = GaussianNB()
# nb_a_score, nb_r_score, nb_p_score = checkCalssifierScore(clf, features_train,
                                                        # features_test,labels_train,labels_test "Naiv Bais ")
# # Gives 0.9393939393939394 0.6666666666666666 0.66666666666666
# print("Naiv Bais ", nb_a_score, nb_r_score, nb_p_score)

# So SLow
# clf = SVC(kernel='linear',  C=10000)
# nb_a_score, nb_r_score, nb_p_score = checkCalssifierScore(clf, features_train,
                                                        # features_test,labels_train,labels_test "SVM")
# print("SVM ", nb_a_score, nb_r_score, nb_p_score)

clf = DecisionTreeClassifier()
dt_a_score, dt_r_score, dt_p_score = checkCalssifierScore(clf, features_train,
                                                        features_test,labels_train,labels_test, "Descion Tree ")
# Gives only 0.9090909090909091 0.6666666666666666 0.5
print("Descion Tree ", dt_a_score, dt_r_score, dt_p_score)

# clf = KNeighborsClassifier(n_neighbors=16, weights='uniform')
# kn_a_score, kn_r_score, kn_p_score = checkCalssifierScore(clf, features_train,
                                                        # features_test,labels_train,labels_test "KNeighborsClassifier")
# # Gives only  1.0 0.0 0.0
# print("KNeighborsClassifier ", kn_a_score, kn_r_score, kn_p_score)

# Walla, That is the heighest score 
clf = RandomForestClassifier()
rf_a_score, rf_r_score, rf_p_score = checkCalssifierScore(clf, 
                                features_train, features_test,labels_train,labels_test ,"RandomForestClassifier")
# Gives Only 0.9696969696969697 0.6666666666666666 1
print("RandomForestClassifier ", rf_a_score, rf_r_score, rf_p_score)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Walla, That is the heighest score 
clf = RandomForestClassifier(criterion= 'entropy', max_features= 3)
rf_a_score, rf_r_score, rf_p_score = checkCalssifierScore(clf, features_train,features_test,labels_train,labels_test, "RandomForestClassifier")
print("RandomForestClassifier ", rf_a_score, rf_r_score, rf_p_score)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)