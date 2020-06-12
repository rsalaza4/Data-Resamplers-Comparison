# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Set random seed
np.random.seed(42)

# Load csv file
df = pd.read_csv('framingham.csv')

# View top 5 rows
df.head()

# EXPLORATORY DATA ANALYSIS

# Visualize the classes distributions
sns.countplot(x=df["TenYearCHD"]).set_title("Outcome Count")

# DATA CLEANING

# Check if there are any null values
df.isnull().values.any()

# Remove null values
df = df.dropna()

# Check if there are any null values
df.isnull().values.any()

# Specify features columns
X = df.drop(columns="TenYearCHD", axis=0)

# Specify target column
y = df["TenYearCHD"]

# MODELS BUILDING AND PERFORMANCE EVALUATION

# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}
           
# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression

# Instantiate the machine learning classifier
log_model = LogisticRegression(max_iter=10000)

# Perform cross-validation to the logistic regression model
log = cross_validate(log_model, X, y, cv=5, scoring=scoring)
    
# Create a data frame with the model perforamnce measures scores
unbalanced_scores = pd.DataFrame({'Unbalanced Data':[log['test_accuracy'].mean(),
                                                           log['test_precision'].mean(),
                                                           log['test_recall'].mean(),
                                                           log['test_f1_score'].mean()]},
                                  index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

unbalanced_scores

# DATA BALANCING

# Import under sampling functions
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import CondensedNearestNeighbour

# Impot over sampling functions
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler

# Instantiate the under sampling techniques
rus = RandomUnderSampler(random_state=42)
nm1 = NearMiss(version=1)
nm2 = NearMiss(version=2)
nm3 = NearMiss(version=3)
tl = TomekLinks()
cnn = CondensedNearestNeighbour(random_state=42)

# Instantiate the over sampling techniques
sm = SMOTE(random_state=42)
blSMOTE = BorderlineSMOTE(random_state=42)
smotenc = SMOTENC(random_state=42, categorical_features=[0,1])
adasyn = ADASYN(random_state=42)
ros = RandomOverSampler(random_state=42)

# Create a list with the resampling techniques
techniques = [rus, nm1, nm2, nm3, tl, cnn, sm, blSMOTE, smotenc, adasyn, ros]

# Define the resamplers evaluation function
def resamplers_evaluation(X, y, folds):

    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds

    '''

    # Create empty lists to append performance metrics scores
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    # Get performance metrics for the model with each resampling technqiue
    for technique in techniques:
 
        # Perform resampling
        X, y = technique.fit_resample(X, y)
        
        # Perform cross-validation to the logistic regression model
        log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
        
        # Append performance metrics scores
        accuracy.append(log['test_accuracy'].mean())
        precision.append(log['test_precision'].mean())
        recall.append(log['test_recall'].mean())
        f1.append(log['test_f1_score'].mean())
        
    # Create data frame with performance metrics scores for each resampling technique    
    results = pd.DataFrame({'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1-Score':f1},
                          index=['RUS', 'NM1', 'NM2', 'NM3', 'Tomek Links', 'CNN', 'SMOTE', 'Borderline SMOTE', 'SMOTE-NC', 'ADASYN', 'ROS']).T
    
    # Add 'Best Technique' column
    results['Best Technique'] = results.idxmax(axis=1)

    # Return performance metrics scores data frame
    return(results)
    
# Run resamplers_evaluation function
resamplers_evaluation(X, y, 5)
