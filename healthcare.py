# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Υποερώτημα Α
#
# Α. Να πραγματοποιηθεί ανάλυση του dataset και γραφική αναπαράσταση αυτής.

# %%
missing_values = ['Unknown', 'N/A']
healthcare = pd.read_csv(
    'data\healthcare-dataset-stroke-data.csv', na_values=missing_values)

pp.ProfileReport(healthcare)


# %% [markdown]
# # Υποερώτημα Β
# #### Από τα παραπάνω φαίνεται ότι λείπουν τιμές από δύο στήλες, την αριθμητική στήλη bmi και την κατηγορηματική στήλη smoking_status.
#
# ## Τρόπος 1 - drop columns
#

# %%
# drop column that has any null items applies on both columns
healthcare_b1 = healthcare.dropna(axis=1, how='any')

healthcare_b1.columns


# %% [markdown]
# ## Τρόπος 2 - mean value

# %%
# fill missing values with the mean value
# applies only to bmi column.
# smoking_status column will be removed

healthcare_b2 = healthcare.fillna(healthcare.mean())
healthcare_b2 = healthcare_b2.dropna(axis=1, how='any')

print(healthcare_b2.isnull().sum())
# print(healthcare_b2.head())

# %% [markdown]
# ## Τρόπος 3 - Linear Regression

# %%
# fill missing values using Linear Regression
# applies only to numerical columns--bmi
# smoking_status column will be removed

healthcare_b3 = healthcare.drop(['smoking_status'], axis=1)
# age and bmi have the highest correlation so I well use a simple linear model regression bmi on age to fill the missing values in bmi
# new dataset with no missing values in the age and bmi columns
df_bmi_age = healthcare_b3.dropna(axis=0, subset=['age', 'bmi'])
df_bmi_age = df_bmi_age.loc[:, ['age', 'bmi']]
# print(df_bmi_age)
# find where
missing_bmi = healthcare_b3['bmi'].isnull()
age_missing_bmi = pd.DataFrame(healthcare_b3['age'][missing_bmi])
missing_bmi_index = age_missing_bmi.index

X = df_bmi_age[['age']]
y = df_bmi_age['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit linear moder
lr = LinearRegression().fit(X_train, y_train)
bmi_pred = lr.predict(age_missing_bmi)
# print(missing_bmi_index)

for fill_index, dataframe_index in enumerate(missing_bmi_index):
    healthcare_b3.loc[dataframe_index, 'bmi'] = bmi_pred[fill_index]


print(healthcare_b3.isnull().sum())
# visualise data
plt.scatter(age_missing_bmi,
            healthcare_b3['bmi'][missing_bmi], marker='o', edgecolor='none')
plt.plot(age_missing_bmi, bmi_pred, color="royalblue", linewidth=2)
plt.xlabel('age')
plt.ylabel('bmi')
plt.show()

# %% [markdown]
# ## Τρόπος 4 - K-Nearest neighbors

# %%
# fill missing values using k-nearest neighbors
# applies to categorical columns -- smoking_status
# bmi column will be removed

# disable chained assignments
pd.options.mode.chained_assignment = None

# encode categorical data of smoking_status column
healthcare_b4 = healthcare.drop(['bmi'], axis=1)

encoder = OrdinalEncoder()
cat_cols = healthcare_b4.select_dtypes('object').columns


def encode(data):
    nonulls = np.array(data.dropna())
    # reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1, 1)
    # encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    # Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data


for columns in cat_cols:
    encode(healthcare_b4[columns])


# #knn imputer
imputer = KNNImputer(n_neighbors=5, weights="uniform")
healthcare_b4 = pd.DataFrame(np.round(imputer.fit_transform(
    healthcare_b4)), columns=healthcare_b4.columns)


# healthcare_b4[['smoking_status']] = encoder.inverse_transform(healthcare_b4[['smoking_status']])
print(healthcare_b4.head(5))
print(healthcare_b4.isnull().sum())


# %% [markdown]
# # Υποερώτημα Γ

# %%
healthcare_c = healthcare.copy()
healthcare_c[['bmi']] = healthcare_b3[['bmi']]
healthcare_c[['smoking_status']] = healthcare_b4[['smoking_status']]
healthcare_c[['gender']] = healthcare_b4[['gender']]
healthcare_c[['ever_married']] = healthcare_b4[['ever_married']]
healthcare_c[['work_type']] = healthcare_b4[['work_type']]
healthcare_c[['Residence_type']] = healthcare_b4[['Residence_type']]

# healthcare_c.head(10)

# %% [markdown]
# ## TASK 1 - RANDOM FOREST
# Για τα νέα μητρώα που προκύπτουν στο υποερώτημα Β, να προβλέψετε αν ένας ασθενής είναι επιρρεπής ή όχι να πάθει εγκεφαλικό χρησιμοποιώντας Random Forest χωρίζοντας το dataset σε trainingtest με αναλογία 75%-25%
#

# %%
X = healthcare_c.drop(['stroke'], axis='columns')
y = healthcare_c['stroke']
# print(X.head(), y.head())

# split training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model1 = RandomForestClassifier(n_estimators=1000, min_samples_leaf=75)
model1.fit(X_train, y_train)

# prediction
y_test_array = y_test.values
y_predicted = model1.predict(X_test)
print(metrics.accuracy_score(y_test_array, y_predicted))


# df=pd.DataFrame({'Actual':y_test, 'Predicted':y_predicted})
# df
print(metrics.confusion_matrix(y_test_array, y_predicted))
print('Classification Report  --> \n',
      metrics.classification_report(y_test_array, y_predicted, labels=[0, 1]))

# %% [markdown]
# ## TASK 3 - Comment results of the categorisation and experiment with different input parameters

# %%


# %%
