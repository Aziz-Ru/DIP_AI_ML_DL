import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder

def load_titanic(file_path = './dataset/train.csv', scaler=False):
    df = pd.read_csv(file_path)
    #print("Missing Values")
    # print(df.isnull().sum())
    print(df.head(10))
    # Set default mid age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # set most frequecy 
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Map Gender 0,1
    df['Sex']= df['Sex'].map({"male":0,"female":1})
    # One Hot Encoding
    df = pd.get_dummies(df,columns=['Embarked'],dtype=int)
    df = df.drop(['PassengerId','Name',"Ticket","Cabin"],axis=1)
    # df = df.dropna()
    # df.drop_duplicates
    print(df.head(10))
    X = df.drop('Survived',axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return df, X_train, X_test, y_train, y_test

def load_blobs(n_samples=1000, centers=4, random_state=42): 
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state, cluster_std=2.5)
    return X, y

def load_wine():
    df = pd.read_csv('./dataset/wine.csv')
    print(df.head())
    # print(df.isnull().sum())
    # print(df.duplicated())
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtypes=='Object':
            df[col]=le.fit_transform(df[col])

    X = df.drop('Wine', axis=1)
    y = df['Wine']
    return X,y, df