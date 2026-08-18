from data_loader import load_titanic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

df,X_train, X_test, y_train, y_test = load_titanic()

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = []
prev,best_n = 0,0
val = [10,20,30,40,50,100,150,200,250,300]
for n in val:
  rf = RandomForestClassifier(n_estimators=n, random_state=42)
  score =cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'

  )
  if score.mean()>prev:
    best_n = n
    prev = score.mean()
  results.append(score.mean())

print(f"Best number of estimators: {best_n}")
print(f"Best cross-validation score: {prev}")

# plt.plot(val, results)
# plt.xlabel('Number of Estimators')
# plt.ylabel('Cross-Validation Accuracy')
# plt.title('Random Forest Classifier Performance')
# plt.show()

model = RandomForestClassifier(n_estimators=best_n, random_state=42)
model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)
classification_re = classification_report(y_test,pred)
confusion_matrix = confusion_matrix(y_test,pred)
print("Accuracy: ",accuracy)
print("Classification Report: \n",classification_re)
print("Confusion Matrix: \n",confusion_matrix)

feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
  "Feature": df.drop(columns=['Survived']).columns,
  "Importance": feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Importance of Features: \n",importance_df)

plt.bar(df.drop(columns=['Survived']).columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest Classifier')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()