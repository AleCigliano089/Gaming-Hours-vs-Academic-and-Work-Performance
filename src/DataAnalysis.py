import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')

print("Info Dataset:")
df = df.drop(columns=['User_ID'])
df = df.drop(columns=['Weekly_Gaming_Hours'])
print(df.info())

missing_values = df.isnull().sum()

print("\nMissing Values per colonna:\n", missing_values[missing_values > 0])

duplicati = df.duplicated().sum()
print(f"Numero di righe duplicate: {duplicati}")


if duplicati > 0:
    print("\nEcco le righe duplicate:")
    print(df[df.duplicated(keep=False)])
else:
    print("\nOttimo! Il dataset non contiene duplicati.")

plt.figure(figsize=(8, 5))

sns.countplot(x='Performance_Impact', data=df, order=['Negative', 'Neutral', 'Positive'])

plt.title('Distribuzione della Variabile Target (Performance_Impact)')

plt.show()

X = df.drop('Performance_Impact', axis=1)
y = df['Performance_Impact']

le = LabelEncoder()
y = le.fit_transform(y)

target_names = le.classes_


X = pd.get_dummies(X, columns=['Gender', 'Occupation', 'Game_Type', 'Primary_Gaming_Time'], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(f"Distribuzione classi nel Training Set PRIMA di SMOTE: {Counter(y_train)}")


smote = SMOTE(random_state=42)


X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Distribuzione classi nel Training Set DOPO SMOTE: {Counter(y_train_resampled)}")

y_resampled_labels = le.inverse_transform(y_train_resampled)

plt.figure(figsize=(8, 6))

sns.countplot(x=y_resampled_labels, order=['Negative', 'Neutral', 'Positive'])
plt.title('Distribuzione Target nel Training Set DOPO SMOTE')
plt.xlabel('Classe')
plt.ylabel('Numero di campioni')
plt.show()
