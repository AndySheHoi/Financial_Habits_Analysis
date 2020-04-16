import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE

from EDA import EDA

# Users who were 60 days enrolled, churn in the next 30 days
df = pd.read_csv('churn_data.csv')

df = EDA.eda(df)

# =============================================================================
# Data Preprocessing
# =============================================================================

user_identifier = df['user']
df = df.drop(columns = ['user'])

# One-Hot Encoding
df.housing.value_counts()
df.groupby('housing')['churn'].nunique().reset_index()
df = pd.get_dummies(df)
df.columns
df = df.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])

# Splitting the df into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'churn'), df['churn'],
                                                    test_size = 0.2, random_state = 0)

# Balancing the Training Set
y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]

# Feature Scaling
sc_X = StandardScaler()

X_train_scaled = pd.DataFrame(sc_X.fit_transform(X_train))
X_train_scaled.columns = X_train.columns.values
X_train_scaled.index = X_train.index.values
X_train = X_train_scaled

X_test_scaled = pd.DataFrame(sc_X.transform(X_test))
X_test_scaled.columns = X_test.columns.values
X_test_scaled.index = X_test.index.values
X_test = X_test_scaled


# Feature Selection by using Recursive Feature Elimination

# Model to Test
model = LogisticRegression(random_state = 0)

# Select Best X Features
rfe = RFE(model, 20)
rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes
# selected features are assigned True value
rfe.support_
# selected features are assigned rank 1
rfe.ranking_

X_train.columns[rfe.support_]

# Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = X_train[X_train.columns[rfe.support_]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix After RFE", fontsize = 40)
# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})    


# Fitting Model to the Training Set
model = LogisticRegression()
model.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = model.predict(X_test[X_train.columns[rfe.support_]])

# Evaluating Results
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (1, 0), columns = (1, 0))
plt.figure(figsize = (10,7))

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Set Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = model,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("K-Fold Accuracy: %0.4f (+/- %0.4f)" % (accuracies.mean(), accuracies.std()))

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),
           pd.DataFrame(np.transpose(model.coef_), columns = ["coef"])
           ],axis = 1)


# Formatting Final Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)
final_results.to_csv('final_result.csv', index = False)


