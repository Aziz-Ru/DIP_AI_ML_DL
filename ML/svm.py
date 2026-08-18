from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import accuracy_score, confusion_matrix, classification_report
from data_loader import load_titanic
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import pandas as pd

df,X_train, X_test, y_train, y_test = load_titanic()

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel= 'rbf', random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)
classification_re = classification_report(y_test,pred)
confusion_matrix = confusion_matrix(y_test,pred)
print("Accuracy: ",accuracy)
print("Classification Report: \n",classification_re)
print("Confusion Matrix: \n",confusion_matrix)


result = permutation_importance(
  model,X_test,y_test, n_repeats=10, random_state=42, scoring='accuracy'
)

importance_df = pd.DataFrame({
  "Feature": df.drop(columns=['Survived']).columns,
  "Importance": result.importances_mean
})

print("Importance of Features: \n",importance_df.sort_values(by='Importance', ascending=False))

print("Permutation Importance: \n",result.importances_mean)