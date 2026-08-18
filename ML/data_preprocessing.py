import pandas as pd

df = pd.read_csv('./dataset/train.csv')

print(df.head())
print(df.shape)
print(df.columns)
print("Info:\n")
print(df.info())

print('Total Null:\n')
print(df.isnull().sum())
print("Datatypes")
print(df.dtypes)

# Numerical Columns
num_cols = df.select_dtypes(
include=["int64","float64"]
).columns
print("Numeric Columns:")
print(num_cols)

categorical_cols = df.select_dtypes(include=['object']).columns
print(categorical_cols)

# Handle Age
df["Age"] = df['Age'].fillna(df['Age'].median())
print(df['Embarked'].unique())
# Handle Embarked
df["Embarked"] = df["Embarked"].fillna(df['Embarked'].mode()[0])
print("MODE:\n")
print(df['Embarked'].mode())
df["Cabin"] = df['Cabin'].fillna('Unknown')
df['Sex']= df["Sex"].map({
"male":0,
"female":1
})
"""
Embarked
--------
S
C
Q
S
C
"""
# One Hot Encoding
df= pd.get_dummies(df,columns=["Embarked"],dtype=int)
"""
Embarked_C    Embarked_Q    Embarked_S
-----------   -----------   -----------
0             0             1
1             0             0
0             1             0
0             0             1
1             0             0
"""

# print(df['Embarked'].unique())

# Remove Unnecessary Columns

df = df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)





