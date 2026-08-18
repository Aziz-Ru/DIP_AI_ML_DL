from data_loader import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

X,y,df = load_wine()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def show_metrics(y_pred):
  accuracy = accuracy_score(y_test,y_pred)
  cr = classification_report(y_test,y_pred)
  cm = confusion_matrix(y_test,y_pred)

  print(f'Accuracy : {accuracy:.3f}')
  print(f"Classification Report : {cr}")
  print(f"Confusion Matrix : {cm}")

  disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
  )
  disp.plot()
  plt.title("Confusion Matrix")
  plt.show()

def logit():

  model = LogisticRegression()
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  show_metrics(y_pred)
  featureimportance = model.coef_[0]
  feature_names = df.drop(columns=["Wine"]).columns
  plt.bar(feature_names,abs(featureimportance))
  plt.show()


def knn():

  x_range = range(1,11)
  result = []
  for k in x_range:
    model = KNeighborsClassifier(n_neighbors= k)
    model.fit(x_train,y_train)
    score = cross_val_score(model,x_train,y_train,cv=5,scoring='accuracy')
    print(f'k = {k} mean={score.mean()}')
    result.append(score.mean())

  plt.plot(x_range,result)
  plt.xlabel("Value of K for KNN")
  plt.ylabel("Cross-Validated Accuracy")
  plt.title("KNN: Varying Number of Neighbors")
  plt.grid(True)
  plt.xticks(x_range)
  plt.show()

  model = KNeighborsClassifier(n_neighbors=7)
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  show_metrics(y_pred)

  


def naive_bayes():
  model = GaussianNB()
  model.fit(x_train,y_train)
  y_pre = model.predict(x_test)
  show_metrics(y_pre)

def dtree():
  model = DecisionTreeClassifier(max_depth=5,random_state=42)
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  show_metrics(y_pred)


def rf():
  n_clusters = [20,30,40,50,100,150,200]
  result  =[]
  for k in n_clusters:
    rf = RandomForestClassifier(n_estimators=k,random_state=42)
    score = cross_val_score(rf,x_train,y_train,cv=5,scoring='accuracy')
    result.append(score.mean())
  
  plt.plot(n_clusters, result)
  plt.xlabel('Number of Estimators')
  plt.ylabel('Cross-Validation Accuracy')
  plt.title('Random Forest Classifier Performance')
  plt.xticks(n_clusters)
  plt.show()

  model = RandomForestClassifier(n_estimators=30)
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  show_metrics(y_pred)
  featureimportance = model.feature_importances_
  feature_names = df.drop(columns=["Wine"]).columns
  plt.bar(feature_names,abs(featureimportance))
  plt.show()



if __name__=='__main__':
  rf()




