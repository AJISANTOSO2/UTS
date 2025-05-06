# UTS
Project ini saya buat untuk mengerjakan ujian UTS Mata kuliah Data Mining

Ini buat manggil library pandas, alat bantu utama di Python buat ngolah data dalam bentuk tabel (DataFrame).
```python
import pandas as pd
```

Ini buat nyambungin Google Colab ke Google Drive, Supaya bisa akses file (kayak dataset) langsung dari Drive.
```python
from google.colab import drive
drive.mount('/content/drive')
```

Ini bikin variabel folder_path yang isinya alamat folder tempat file dataset disimpan di Google Drive.
```python
folder_path = '/content/drive/My Drive/Dataset/'
```

ini adalah baca file CSV pertama (data attack DoS ICMP Flood), terus simpan datanya ke dalam variabel aji. Baris aji di bawahnya buat nampilin isi datanya di notebook.
```python
aji = pd.read_csv(folder_path + 'DoS ICMP Flood.csv')
aji
```

ini adalah baca file CSV pertama (data attack MQTT Malformed), terus simpan datanya ke dalam variabel aji2. Baris aji2 di bawahnya buat nampilin isi datanya di notebook.
```python
aji2 = pd.read_csv(folder_path + 'MQTT Malformed.csv')
aji2 
```

Gabungin dua dataset tadi jadi satu, lalu pada ignore_index=True artinya index-nya di-reset ulang dari 0 biar rapih.
```python
hasilgabung = pd.concat([aji,aji2],ignore_index=True)
```

Buat liat semua nama kolom dari dataset yang udah digabung.
```python
hasilgabung.columns.values
```

Ambil kolom fitur (fitur = data yang dijadiin bahan buat prediksi). Di sini ngambil kolom dari indeks ke-7 sampai sebelum ke-76.
```python
x = hasilgabung.iloc[:,7: 76]
```
ini adalah untuk menampilkan data dari hasilgabung.iloc
```python
x
```

Ambil kolom label atau target (yaitu jenis attack name) dari kolom ke-83. Ini yang nanti mau diprediksi.
```python
y = hasilgabung.iloc[:,83: 84]
```

Ini untuk menampilkan kolom hasil attack name
```python
y
```

Bagi data jadi dua bagian: training (buat belajar model) dan testing (buat uji model). test_size = 0.2 artinya 20% data buat testing. random_state biar hasil split-nya tetap konsisten.
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
```

Ini bikin model klasifikasi pakai Decision Tree (pohon keputusan). Pakai metode entropy buat ngukur seberapa "ngacak" data-nya. splitter='random' artinya pilih fitur secara acak waktu bikin cabang pohon. Lalu, fit() buat latih modelnya dengan data training.
```python
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

aji = DecisionTreeClassifier(criterion='entropy', splitter = 'random')
aji.fit(x_train,y_train)
```

Menggunakan model yang udah dilatih buat prediksi data testing, hasilnya disimpan di y_pred.
```python
y_pred = aji.predict(x_test)
y_pred
```

Menghitung akurasi model, seberapa banyak prediksi yang benar dibanding total prediksi. Terus ditampilkan dalam bentuk persentase.
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

accuracy_percent = round(accuracy * 100, 2)
print(f"Akurasi: {accuracy_percent}%")
```

Membuat visualisasi pohon keputusan yang udah dilatih tadi. Jadi bisa diliat alur pengambilan keputusan model-nya.
```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (10, 7))
tree.plot_tree(aji, feature_names = x.columns.values, class_names = np.array(['DoS ICMP Flood' ,'MQTT Malformed']), filled = True)
plt.show()
```

Ini untuk membuat dan nampilin confusion matrix pakai heatmap. Ini grafik yang nunjukkin perbandingan antara prediksi model dan nilai aslinya. Biar mudah kelihatan mana yang bener dan salah klasifikasi.
```python
import matplotlib.pyplot as plt
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
lol.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
