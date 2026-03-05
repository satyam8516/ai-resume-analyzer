
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("resume_dataset.csv")

X = data.drop("selected", axis=1)
y = data["selected"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

joblib.dump(model, "resume_model.pkl")

print("Model trained and saved as resume_model.pkl")
