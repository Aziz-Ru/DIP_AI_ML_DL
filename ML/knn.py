from data_loader import load_titanic
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df,X_train, X_test, y_train, y_test = load_titanic()
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

values = range(1,20)
result = []
best_k , prev_score = 0, 0
for k in values:
    model = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=5,
        scoring='accuracy'
      )
    if score.mean() > prev_score:
        best_k = k
    prev_score = score.mean()
    result.append(score.mean())

plt.plot(values,result)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.title("KNN: Varying Number of Neighbors")
plt.grid(True)
plt.show()

print(f"Best value of k: {best_k}")

#  before calculating classifier we need get n value of k
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
classification_re = classification_report(y_test,y_pred)
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Accuracy: ",accuracy)
print("Classification Report: \n",classification_re)
print("Confusion Matrix: \n",confusion_matrix)

