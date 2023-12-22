from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

import numpy as np
import time
import matplotlib.pyplot as plt

def create_X_and_y():
    file_path = "./chronic_kidney_disease.xlsx"
    
    dataset = pd.read_excel(file_path)
    ornekUzayi = dataset.values;
    
    np.random.shuffle(ornekUzayi)
    
    labels = dataset.columns
     
    X = ornekUzayi[:, 0:len(labels)-1 ]
    y = ornekUzayi[:, len(labels)-1]
    
    return X,y

# Örnek veri kümesi oluşturun
X, y = create_X_and_y()
# K-Fold cross-validation nesnesini oluşturun
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# KNN sınıflandırıcı nesnesini oluşturun
knn = KNeighborsClassifier(n_neighbors=3)  # 3 en yakın komşuyu kullan

# Decision Tree sınıflandırıcı nesnesini oluşturun
dt = DecisionTreeClassifier(random_state=42)

# Random Forest sınıflandırıcı nesnesini oluşturun
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Sınıflandırıcıları eğitip değerlendirin
classifiers = [knn, dt, rf]
results = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    for clf in classifiers:
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed_time = time.time() - start_time

        # Confusion Matrix hesapla
        cm = confusion_matrix(y_test, y_pred)

        # Accuracy hesapla
        accuracy = accuracy_score(y_test, y_pred)

        # Sensitivity ve Specificity hesapla
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        results.append({
            "Algorithm": clf.__class__.__name__,
            "Confusion Matrix": cm,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Elapsed Time (seconds)": elapsed_time
        })

# Sonuçları yazdır
for result in results:
    print(f"{result['Algorithm']} Performansı:")
    print("Confusion Matrix:\n", result['Confusion Matrix'])
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print(f"Sensitivity: {result['Sensitivity']:.2f}")
    print(f"Specificity: {result['Specificity']:.2f}")
    #print(f"Elapsed Time: {result['Elapsed Time (seconds'):.2f} seconds\n")

    import matplotlib.pyplot as plt

# TP, TN, FP, FN değerlerini elde et
tp_values = [result['Confusion Matrix'][1][1] for result in results]
tn_values = [result['Confusion Matrix'][0][0] for result in results]
fp_values = [result['Confusion Matrix'][0][1] for result in results]
fn_values = [result['Confusion Matrix'][1][0] for result in results]

algorithms = [result['Algorithm'] for result in results]

# İndeksler
x = range(len(algorithms))

# Genişlik ayarı
width = 0.2

# TP, TN, FP, FN değerlerini y ekseni üzerinde mum grafiklerle gösterme
plt.bar(x, tp_values, width, label='TP')
plt.bar([i + width for i in x], tn_values, width, label='TN')
plt.bar([i + width * 2 for i in x], fp_values, width, label='FP')
plt.bar([i + width * 3 for i in x], fn_values, width, label='FN')

# X ekseninde etiketlerin ve grafik başlığının ayarlanması
plt.xlabel('Algoritmalar')
plt.ylabel('Sayı Değeri')
plt.xticks([i + width for i in x], algorithms, rotation=15, ha="right")
plt.title('Confusion Matrix Değerleri')

# Grafikteki çubukları gösterme
plt.legend()

# Grafikleri görüntüleme
plt.show()