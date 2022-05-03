import mlflow
import pandas as pd
import numpy as np
import warnings
import uuid
from random import randint
from random import choice
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

def sample_data(n):
    '''
    This function will generate random data
    '''
    d = pd.DataFrame(
        {
            'uuid' : [uuid.uuid4() for _ in range(n)],
            'watch_time' : [randint(0, 60) for _ in range(n)],
            'engagement' : [randint(1, 6) for _ in range(n)],
            'age_range' : [randint(1,5) for _ in range(n)],
            'ads_seen' : [randint(5,100) for _ in range(n)],
            'ads_clicked' : [randint(0,15) for _ in range(n)],
            'revenue_generated' : [randint(0, 1000) for _ in range(n)]
        }
    ).drop_duplicates()
    return d

d = sample_data(n = 1000)
d.shape

# train test split
X = d[['watch_time', 'engagement', 'ads_seen', 'ads_clicked']].values
y = d['age_range'].values
x_train, x_test, y_train, y_test = train_test_split(X ,y ,test_size = 0.3, stratify = y)

# build model
lr = 1.0
n_est = 100
clf = GradientBoostingClassifier(
    n_estimators=n_est, learning_rate=lr,max_depth=1, random_state=0
).fit(x_train, y_train)

# evaluate model
test_cv = cross_val_score(clf, x_test, y_test, cv=8) 
y_pred = clf.predict(x_test)
test_acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred, output_dict = True)

print("Testing Cross Validation : ", np.mean(test_cv))
print("Testing Accuracy : ", test_acc)
print(classification_report(y_test, y_pred))

