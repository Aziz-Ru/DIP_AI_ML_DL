from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_mutual_info_score,adjusted_rand_score, silhouette_score
from data_loader import load_titanic
from sklearn.linear_model import LogisticRegression
import pandas as pd
scaler = StandardScaler()

df,X_train, X_test, y_train, y_test = load_titanic()

X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train,y_train)
print("Model Train Completed")
y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)
classification_re = classification_report(y_test,y_pred)
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Accuracy: ",accuracy)
print("Classification Report: \n",classification_re)
print("Confusion Matrix: \n",confusion_matrix)

feature_importance = model.coef_[0]
importance_df = pd.DataFrame({
  "Feature": df.drop(columns=['Survived']).columns,
  "Coefficient": feature_importance,
  "Importance": abs(feature_importance)
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Importance of Features: \n",importance_df)




