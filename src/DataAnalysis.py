import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')

print("Info Dataset:")
if 'User_ID' in df.columns:
    df = df.drop(columns=['User_ID'])
if 'Weekly_Gaming_Hours' in df.columns:
    df = df.drop(columns=['Weekly_Gaming_Hours'])
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


X = df.drop('Performance_Impact', axis=1)
y = df['Performance_Impact']

le = LabelEncoder()
y = le.fit_transform(y)
target_names = le.classes_

X = pd.get_dummies(X, columns=['Gender', 'Occupation', 'Game_Type', 'Primary_Gaming_Time'], drop_first=True, dtype=int)

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

        # 2. Applico Clipping a X_train
        outliers_train_count = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).sum()
        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)

        # 3. Applico gli STESSI bound del train a X_test
        X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)

        print(f"Colonna '{col}': trovati {outliers_train_count} outliers nel Train -> Applicato Clipping su Train e Test.")


scaler = MinMaxScaler()
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
plt.show()



print("\n" + "="*60)
print("Random Forest su dati NON BILANCIATI (No SMOTE) - Con Tuning + CV")
print("="*60)

param_grid_no_smote = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

rf_base_no_smote = RandomForestClassifier(random_state=42)

grid_search_no_smote = GridSearchCV(estimator=rf_base_no_smote,
                                    param_grid=param_grid_no_smote,
                                    cv=5,
                                    scoring='f1_weighted',
                                    n_jobs=-1,
                                    verbose=1)

print("Inizio ricerca migliori iperparametri (No SMOTE)...")
grid_search_no_smote.fit(X_train_scaled, y_train)

rf_no_smote = grid_search_no_smote.best_estimator_

print(f"\nMigliori parametri (No SMOTE): {grid_search_no_smote.best_params_}")
print(f"Miglior F1-Score (CV No SMOTE): {grid_search_no_smote.best_score_:.4f}")

y_pred_no_smote = rf_no_smote.predict(X_test_scaled)

print("Report di Classificazione (Modello SBILANCIATO - Ottimizzato):")
print(classification_report(y_test, y_pred_no_smote, target_names=target_names))


acc_no = accuracy_score(y_test, y_pred_no_smote)
prec_no = precision_score(y_test, y_pred_no_smote, average='weighted')
rec_no = recall_score(y_test, y_pred_no_smote, average='weighted')
f1_no = f1_score(y_test, y_pred_no_smote, average='weighted')


metrics_df_no = pd.DataFrame({
    'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Valore': [acc_no, prec_no, rec_no, f1_no]
})

plt.figure(figsize=(10, 6))
ax1 = sns.barplot(x='Metrica', y='Valore', data=metrics_df_no)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}',
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9),
                 textcoords='offset points', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.title('Performance Modello - SENZA SMOTE (Ottimizzato)', fontsize=16)
plt.ylabel('Valore (0-1)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(8, 6))
cm_no = confusion_matrix(y_test, y_pred_no_smote)
sns.heatmap(cm_no, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice di Confusione (SENZA SMOTE)')
plt.ylabel('Reale')
plt.xlabel('Predetto')
plt.show()



print("\n" + "="*60)
print("Avvio addestramento Random Forest (Con SMOTE + Tuning + CV)")
print("="*60)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_base,
                           param_grid=param_grid,
                           cv=5,
                           scoring='f1_weighted',
                           n_jobs=-1,
                           verbose=1)

print("Inizio ricerca migliori iperparametri (Grid Search)...")
grid_search.fit(X_train_resampled, y_train_resampled)

rf_model = grid_search.best_estimator_

print(f"\nMigliori parametri trovati: {grid_search.best_params_}")
print(f"Miglior F1-Score (Validation CV): {grid_search.best_score_:.4f}")

cv_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print(f"Accuracy media CV (5-Fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

y_pred = rf_model.predict(X_test_scaled)


acc_smote = accuracy_score(y_test, y_pred)
prec_smote = precision_score(y_test, y_pred, average='weighted')
rec_smote = recall_score(y_test, y_pred, average='weighted')
f1_smote = f1_score(y_test, y_pred, average='weighted')

print("\n--- Risultati Numerici (Modello CON SMOTE Ottimizzato) ---")
print(f"Accuracy:  {acc_smote:.4f}")
print(f"Precision: {prec_smote:.4f}")
print(f"Recall:    {rec_smote:.4f}")
print(f"F1-Score:  {f1_smote:.4f}")

print("\n--- Classification Report Dettagliato ---")
print(classification_report(y_test, y_pred, target_names=target_names))


metrics_df_smote = pd.DataFrame({
    'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Valore': [acc_smote, prec_smote, rec_smote, f1_smote]
})

plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x='Metrica', y='Valore', data=metrics_df_smote)
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}',
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9),
                 textcoords='offset points', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.title('Performance Modello - CON SMOTE (Ottimizzato)', fontsize=16)
plt.ylabel('Valore (0-1)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice di Confusione (CON SMOTE)')
plt.ylabel('Reale')
plt.xlabel('Predetto')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Daily_Gaming_Hours',
    y='Sleep_Hours',
    hue='Performance_Impact',
    palette={'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'},
    style='Performance_Impact',
    s=100,
    alpha=0.7
)
plt.axhline(y=6, color='black', linestyle='--', label='Soglia Sonno (6h)')
plt.axvline(x=4, color='black', linestyle='--', label='Soglia Gioco (4h)')
plt.title('Separazione delle Classi (Dati Originali)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n--- Analisi Feature Importance (Completa) ---")
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importanza di TUTTE le Feature (Random Forest)')
plt.xlabel('Importanza')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()