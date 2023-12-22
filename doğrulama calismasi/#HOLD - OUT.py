#HOLD - OUT çalışması
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

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

# Örnek veri kümesi oluşturma
X, y = create_X_and_y()

# Veriyi eğitim ve test verileri olarak bölmek için train_test_split'i kullanma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN sınıflandırıcı nesnesini oluşturma
knn = KNeighborsClassifier(n_neighbors=3)  # 3 en yakın komşuyu kullanma

# Decision Tree sınıflandırıcı nesnesini oluşturma
dt = DecisionTreeClassifier(random_state=42)

# Random Forest sınıflandırıcı nesnesini oluşturma
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Sınıflandırıcıları eğitip değerlendirinme
classifiers = [knn, dt, rf]
results = []

for clf in classifiers:
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    elapsed_time = time.time() - start_time

    # Confusion Matrix hesaplama
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy hesapla
    accuracy = accuracy_score(y_test, y_pred)

    # Sensitivity ve Specificity hesaplama
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

# Sonuçları yazdırma
for result in results:
    print(f"{result['Algorithm']} Performansı:")
    print("Confusion Matrix:\n", result['Confusion Matrix'])
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print(f"Sensitivity: {result['Sensitivity']:.2f}")
    print(f"Specificity: {result['Specificity']:.2f}")
    print(f"Elapsed Time: {result['Elapsed Time (seconds)']:.2f} seconds\n")


#skdjfbsdıufbıdsuf
# Sonuçları yazdırma ve grafik oluşturma
for result in results:
    print(f"{result['Algorithm']} Performansı:")
    print("Confusion Matrix:\n", result['Confusion Matrix'])
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print(f"Sensitivity: {result['Sensitivity']:.2f}")
    print(f"Specificity: {result['Specificity']:.2f}")

    # Algortima adını ve doğruluk (accuracy) değerini grafikte gösterme
    plt.bar(result['Algorithm'], result['Accuracy'], label='Accuracy')

# Grafik başlığı ve ekseni etiketlerini ayarlama
plt.title('Sınıflandırma Algoritmalarının Performansı')
plt.xlabel('Algoritmalar')
plt.ylabel('Accuracy')

# Grafik üzerindeki x ekseni etiketlerini döndürme
plt.xticks(rotation=15, ha="right")

# Grafikteki çubukları gösterme
plt.legend()

# Grafikleri görüntüleme
plt.show()
