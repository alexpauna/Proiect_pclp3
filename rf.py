import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


target_column = 'EngagementLevel'
categorical_columns = ['Gender', 'Location', 'GameGenre', 'GameDifficulty','InGamePurchases']
numeric_columns = ['Age', 'PlayTimeHours', 'SessionsPerWeek',
                   'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
all_colums= categorical_columns + numeric_columns


# incarcare dataset-ului
csv_file = 'online_gaming_behavior_dataset.csv'
original = pd.read_csv(csv_file)
gaming=original.copy()

gaming['InGamePurchases'] = gaming['InGamePurchases'].astype('category')


#adaugam elemente lipsa in mod aleator
#######################################################

procent=0.1

elemente_lipsa = int(len(gaming) * procent)

for _ in range(elemente_lipsa):
   random_col=np.random.choice(all_colums)
   random_row=np.random.randint(0, len(gaming))
   gaming.loc[random_row, random_col] = np.nan

#######################################################

# Label Encoding pentru coloanele categorice folosite în rf
rf_data = gaming.copy()
for col in categorical_columns:
    encoder = LabelEncoder()
    rf_data[col] = encoder.fit_transform(rf_data[col])


X = rf_data.drop(columns=[target_column, 'PlayerID'])
y = rf_data[target_column]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df[target_column] = y_train.values
train_df.to_csv('train.csv', index=False)

test_df = X_test.copy()
test_df[target_column] = y_test.values
test_df.to_csv('test.csv', index=False)

#EDA-1
#############################################################
with open("EDA-1_forest.txt", "w", encoding="utf-8") as f:
    def raport(df, name):
        nr_elemente_lipsa = df.isnull().sum()
        procent_elemente_lipsa = 100 * nr_elemente_lipsa / len(df)
        for col in df.columns:
            if nr_elemente_lipsa[col] > 0:
                f.write(f"{name} - {col}: {nr_elemente_lipsa[col]} valori lipsa ({procent_elemente_lipsa[col]:.2f}%)\n")
                f.write("\n")

    raport(X_train, "X_train")
    f.write("\n")
    raport(X_test, "X_test")


for col in all_colums:
    if col in numeric_columns:
        value = X_train[col].mean()
    else:
        value = X_train[col].mode()[0]
    X_train.loc[:, col] = X_train[col].fillna(value)
    X_test.loc[:, col] = X_test[col].fillna(value)

#EDA-2
#############################################################

with open("EDA-2-describe_forest.txt", "w", encoding="utf-8") as f:
   for col in all_colums:
    f.write(f"\nStatistici descriptive pentru {col}:")
    f.write("\n")
    f.write(gaming[col].describe().to_string())
    f.write("\n")

##############################################################

#EDA-3
##############################################################

for col in all_colums:
    valori_unice = len(gaming[col].dropna().unique())
    if valori_unice > 25:
        valori_unice = 25
    if col in numeric_columns:
        plt.hist(gaming[col].dropna(), bins=valori_unice, edgecolor='black', alpha=0.7)
        plt.title(f'Histograma pentru {col}')
        plt.xlabel(col)
        plt.ylabel('Frecvență')
        filename = f"forest/histograma_{col}.png"
    else:
        sns.countplot(x=gaming[col])
        plt.title(f'Countplot pentru {col}')
        plt.xlabel(col)
        plt.ylabel('Număr de apariții')
        filename = f"forest/countplot_{col}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

#############################################################

# EDA-4
#############################################################

for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=gaming[col],showfliers=True)
    plt.title(f'Boxplot - Outlieri în {col}')
    plt.tight_layout()
    plt.savefig(f'forest/boxplot_{col}.png')
    plt.close()

#############################################################


#EDA-5
#############################################################


cor_matrix = gaming[numeric_columns].corr()


sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice de Corelatii Intre Variabilele Numerice")
plt.savefig("forest/matrice_corelatii_numerice.png")
plt.close()

################################################################


#EDA-6
################################################################
for feature in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=gaming, x=target_column, y=feature)
    plt.title(f'Violin Plot: {feature} vs {target_column}')
    plt.tight_layout()
    plt.savefig(f'forest/violin_{feature}_vs_{target_column}.png')
    plt.close()

################################################################


# Normalizam toate coloanele după split
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, rf_y_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_y_pred, average='weighted', zero_division=0)
print(f"Acuratețea Random Forest: {rf_accuracy:.2f}")
print(f"Precizie: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1-score: {rf_f1:.2f}")


rf_cm = confusion_matrix(y_test, rf_y_pred, labels=np.unique(y_test))
plt.figure(figsize=(6, 5))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicții')
plt.ylabel('Valori reale')
plt.title('Matrice de Confuzie Random Forest')
plt.tight_layout()
plt.show()







