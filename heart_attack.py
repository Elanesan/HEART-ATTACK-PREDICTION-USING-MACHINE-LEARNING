import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv("/content/Cardiovascular_Disease_Dataset.csv")

X = data[["age", "gender", "chestpain", "restingBP", "serumcholestrol", "fastingbloodsugar", "restingrelectro", "maxheartrate", "exerciseangia"]]
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = GaussianNB()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


new_age = float(input("AGE : "))
new_gender = float(input("GENDER (Male : 1 / Female : 0): "))
new_cp = float(input("chest pain type ( typical angina : 1 / atypical angina: 2 / non-anginal pain: 3 / asymptomatic: 4 ): Enter the value  "))
new_trtbps = float(input("Resting blood pressure (in mm Hg) : "))
new_chol = float(input("Cholestoral in mg/dl fetch via BMI sensor : "))
new_fbs = float(input("Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) : "))
new_rest_ecg = float(input("Resting electrocardiographic results [normal: 0/ having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV): 1/ showing probable or definite left ventricular hypertrophy by Estes' criteria: 2 ]: "))
new_heart_rate = float(input("Maximum heart rate achieved : "))
new_exercise = float(input("Exercise ( yes : 1/ NO : 0) : "))

new_data = [[new_age,new_gender,new_cp,new_trtbps,new_chol,new_fbs,new_rest_ecg,new_heart_rate,new_exercise]]
prediction = model.predict(new_data)

print("Predicted:", prediction[0])
if prediction[0] == 0:
    print("Less Chance of Heart Attack")
elif prediction[0]==1:
    print("More Chance of Heart Attack")
else:
  print("Invalid Data")
