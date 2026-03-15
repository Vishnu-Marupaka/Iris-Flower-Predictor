import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

df = pd.read_csv("iris.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

x = df.drop("species", axis=1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = SVC()
model.fit(x_train, y_train)

joblib.dump(model, 'model.joblib')