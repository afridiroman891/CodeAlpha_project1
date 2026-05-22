import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)


df = pd.read_csv("D:\\creditworthness.csv")


df['Debt_to_Income_Ratio'] = df['Debts'] / df['Income']


label_encoder = LabelEncoder()
df['Payment_History_Encoded'] = label_encoder.fit_transform(df['Payment_History'])


X = df[['Income', 'Debts', 'Debt_to_Income_Ratio']]
y = df['Payment_History_Encoded']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='gini')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 

accuracy = accuracy_score(y_test, y_pred)


precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')



print("================ MODEL PERFORMANCE REVEW ================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Weighted)")
print(f"Recall:    {recall:.4f} (Weighted)")
print(f"F1-Score:  {f1:.4f} (Weighted)")
print(f"ROC-AUC:   {roc_auc:.4f} (OVR Weighted)")
print("========================================================\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=label_encoder.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure for Creditworthiness Prediction")
plt.show()