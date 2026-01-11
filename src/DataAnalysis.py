import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')

print("Info Dataset:")
if 'User_ID' in df.columns:
    df = df.drop(columns=['User_ID'])
if 'Weekly_Gaming_Hours' in df.columns:
    df = df.drop(columns=['Weekly_Gaming_Hours'])
    df = df.drop(columns=['Productivity_Level'])
print(df.info())

print("\n--- Controllo Valori Mancanti (Null) ---")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "Nessun valore mancante trovato.")

duplicati = df.duplicated().sum()
print(f"Numero di righe duplicate: {duplicati}")
if duplicati > 0:
    print(df[df.duplicated(keep=False)])

plt.figure(figsize=(8, 5))
sns.countplot(x='Performance_Impact', data=df, order=['Negative', 'Neutral', 'Positive'])
plt.title('Distribuzione della Variabile Target (Performance_Impact)')
plt.show()


print("\n" + "="*60)
print("PARTE 1: CLASSIFICAZIONE (Target: Performance_Impact)")
print("="*60)

X = df.drop('Performance_Impact', axis=1)
y = df['Performance_Impact']


le_target = LabelEncoder()
y = le_target.fit_transform(y)
target_names = le_target.classes_


cat_cols = ['Gender', 'Occupation', 'Game_Type', 'Primary_Gaming_Time']


for col in cat_cols:
    le_cat = LabelEncoder()
    X[col] = le_cat.fit_transform(X[col])


cat_indices = [X.columns.get_loc(col) for col in cat_cols]
print(f"Indici colonne categoriche per SMOTENC: {cat_indices}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


target_outlier_cols = ['Daily_Gaming_Hours', 'Age', 'Sleep_Hours']
print(f"\n--- Gestione Outliers (Post-Split) su: {target_outlier_cols} ---")

X_train = X_train.copy()
X_test = X_test.copy()

for col in target_outlier_cols:
    if col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_train_count = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).sum()
        print(f"Colonna '{col}': trovati {outliers_train_count} ")



scaler_base = MinMaxScaler()
X_train_scaled = scaler_base.fit_transform(X_train)
X_test_scaled = scaler_base.transform(X_test)


print(f"Distribuzione classi nel Training Set PRIMA di SMOTENC: {Counter(y_train)}")


smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

print(f"Distribuzione classi nel Training Set DOPO SMOTENC: {Counter(y_train_resampled)}")


scaler_smote = MinMaxScaler()
X_train_resampled_scaled = scaler_smote.fit_transform(X_train_resampled)

X_test_scaled_smote = scaler_smote.transform(X_test)


y_resampled_labels = le_target.inverse_transform(y_train_resampled)
plt.figure(figsize=(8, 6))
sns.countplot(x=y_resampled_labels, order=['Negative', 'Neutral', 'Positive'])
plt.title('Distribuzione Target nel Training Set DOPO SMOTENC')
plt.show()



print("\n" + "-"*40)
print("A. Random Forest su dati NON BILANCIATI (Scalati)")
print("-"*40)

param_grid_no_smote = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

rf_base_no_smote = RandomForestClassifier(random_state=42)
grid_search_no_smote = GridSearchCV(estimator=rf_base_no_smote, param_grid=param_grid_no_smote, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)


grid_search_no_smote.fit(X_train_scaled, y_train)
rf_no_smote = grid_search_no_smote.best_estimator_

print(f"Migliori parametri (No SMOTENC): {grid_search_no_smote.best_params_}")
y_pred_no_smote = rf_no_smote.predict(X_test_scaled)

print("Report (No SMOTENC):")
print(classification_report(y_test, y_pred_no_smote, target_names=target_names))

acc_no = accuracy_score(y_test, y_pred_no_smote)
prec_no = precision_score(y_test, y_pred_no_smote, average='weighted')
rec_no = recall_score(y_test, y_pred_no_smote, average='weighted')
f1_no = f1_score(y_test, y_pred_no_smote, average='weighted')

metrics_df_no = pd.DataFrame({'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'], 'Valore': [acc_no, prec_no, rec_no, f1_no]})
plt.figure(figsize=(10, 6))
ax1 = sns.barplot(x='Metrica', y='Valore', data=metrics_df_no)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.title('Performance Modello - SENZA SMOTENC')
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_no_smote)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice di Confusione (Random Forest Senza SMOTENC)')
plt.ylabel('Reale')
plt.xlabel('Predetto')
plt.show()

print("\n" + "-"*40)
print("B. Random Forest (Con SMOTENC + Tuning + Scaling)")
print("-"*40)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)


grid_search.fit(X_train_resampled_scaled, y_train_resampled)
rf_model = grid_search.best_estimator_

print(f"Migliori parametri (SMOTE): {grid_search.best_params_}")

y_pred = rf_model.predict(X_test_scaled_smote)

acc_smote = accuracy_score(y_test, y_pred)
prec_smote = precision_score(y_test, y_pred, average='weighted')
rec_smote = recall_score(y_test, y_pred, average='weighted')
f1_smote = f1_score(y_test, y_pred, average='weighted')

print("Report (Con SMOTE):")
print(classification_report(y_test, y_pred, target_names=target_names))

metrics_df_smote = pd.DataFrame({'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'], 'Valore': [acc_smote, prec_smote, rec_smote, f1_smote]})
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x='Metrica', y='Valore', data=metrics_df_smote)
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.title('Performance Modello - CON SMOTENC (Ottimizzato)')
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice di Confusione (Random Forest SMOTENC)')
plt.ylabel('Reale')
plt.xlabel('Predetto')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x='Daily_Gaming_Hours', y='Sleep_Hours', hue='Performance_Impact',
    palette={'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'},
    style='Performance_Impact', s=100, alpha=0.7
)
plt.axhline(y=6, color='black', linestyle='--', label='Soglia Sonno (6h)')
plt.axvline(x=4, color='black', linestyle='--', label='Soglia Gioco (4h)')
plt.title('Separazione delle Classi (Dati Originali)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n--- Analisi Feature Importance (Random Forest) ---")
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importanza di TUTTE le Feature (Random Forest Classifier)')
plt.tight_layout()
plt.show()



print("\n" + "="*60)
print("PARTE 2: REGRESSIONE (Target: Academic_or_Work_Score)")
print("="*60)
print("Nota: Rimuoviamo 'Productivity_Level' dalle feature per evitare Data Leakage (correlazione ~96%)")

X_reg = df.drop(['Performance_Impact', 'Academic_or_Work_Score'], axis=1)
y_reg = df['Academic_or_Work_Score']


X_reg = pd.get_dummies(X_reg, columns=['Gender', 'Occupation', 'Game_Type', 'Primary_Gaming_Time'], drop_first=True, dtype=int)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = MinMaxScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
print("Addestramento RandomForestRegressor in corso...")
rf_regressor.fit(X_train_r_scaled, y_train_r)

y_pred_r = rf_regressor.predict(X_test_r_scaled)

mae = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print(f"\n--- Risultati Regressione ---")
print(f"MAE (Errore Medio Assoluto): {mae:.2f}")
print(f"RMSE (Radice Errore Quadratico Medio): {rmse:.2f}")
print(f"R2 Score (Coefficiente di Determinazione): {r2:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.6)
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--', lw=2, label='Previsione Perfetta')
plt.xlabel('Voto Reale (Academic/Work Score)')
plt.ylabel('Voto Predetto')
plt.title('Regressione: Reale vs Predetto')
plt.legend()
plt.show()

metrics_reg_df = pd.DataFrame({
    'Metrica': ['MAE', 'RMSE', 'R2 Score'],
    'Valore': [mae, rmse, r2]
})

plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x='Metrica', y='Valore', data=metrics_reg_df, palette='viridis')
for p in ax3.patches:
    ax3.annotate(f'{p.get_height():.4f}',
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9),
                 textcoords='offset points', fontsize=12, fontweight='bold')
plt.title('Performance Modello di Regressione (Random Forest)', fontsize=16)
plt.ylabel('Valore')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("\n--- Analisi Feature Importance (Regressione) ---")


importances_reg = rf_regressor.feature_importances_


feature_names_reg = X_reg.columns


feature_importance_reg_df = pd.DataFrame({
    'Feature': feature_names_reg,
    'Importance': importances_reg
})


feature_importance_reg_df = feature_importance_reg_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_reg_df)
plt.title('Importanza delle Feature nel Modello di Regressione\n(Target: Academic_or_Work_Score)', fontsize=14)
plt.xlabel('Importanza (Gini Importance)')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


print("Top 5 Feature più influenti sul punteggio:")
print(feature_importance_reg_df.head(5).to_string(index=False))