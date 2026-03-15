import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/vishn/Downloads/iris.csv")
df.drop_duplicates()
df.dropna()
print(f"After cleaning no of rows in data:{len(df)}")

x=df.drop("species", axis=1)
y=df['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

model=SVC()
model.fit(x_train,y_train)

print(x_test)
y_pred =model.predict(x_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
testing = pd.DataFrame({
    'sepal_length': [6.6],
    'sepal_width': [2.9],
    'petal_length': [4.5],
    'petal_width': [1.3]
})
print(f"Accuracy of SVM model: {accuracy:.2f}")
print(model.predict(testing)[0])
