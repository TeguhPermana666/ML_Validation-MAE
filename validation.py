"""
validasi model digunakan untuk mengukur kualitas dari model yang anda buat
->kunci untuk meningkatkan model secara iteratir(bertahap)

Anda pasti ingin mengevaluasi hampir setiap model yang pernah Anda buat. Di sebagian besar (meskipun tidak semua) aplikasi, 
ukuran kualitas model yang relevan adalah akurasi prediktif. Dengan kata lain, apakah prediksi model akan mendekati apa yang sebenarnya terjadi.
Banyak orang membuat kesalahan besar saat mengukur akurasi prediksi. 
Mereka membuat prediksi dengan data pelatihan mereka dan membandingkan prediksi tersebut dengan nilai target dalam data pelatihan. 
Anda akan melihat masalah dengan pendekatan ini dan bagaimana menyelesaikannya sebentar lagi, tetapi mari kita pikirkan bagaimana kita akan melakukannya terlebih dahulu.
-Pertama-tama Anda harus meringkas kualitas model menjadi cara yang dapat dimengerti. 

-Jika Anda membandingkan nilai rumah yang diprediksi dan aktual untuk 10.000 rumah, 
kemungkinan besar Anda akan menemukan campuran prediksi yang baik dan yang buruk.
Melihat melalui daftar 10.000 nilai yang diprediksi dan nilai aktual tidak akan ada gunanya. Kita perlu meringkas ini menjadi satu metrik.

Ada banyak metrik untuk meringkas kualitas model, 
tetapi kita akan mulai dengan satu yang disebut Mean Absolute Error (juga disebut MAE).
Mari kita uraikan metrik ini dimulai dengan kata terakhir, kesalahan.

Jadi, jika sebuah rumah berharga $ 150.000 dan Anda memperkirakan biayanya $ 100.000, kesalahannya adalah $ 50.000.

Dengan metrik MAE, kami mengambil nilai absolut dari setiap kesalahan. Ini mengubah setiap kesalahan menjadi angka positif.
Kami kemudian mengambil rata-rata kesalahan absolut tersebut. Ini adalah ukuran kualitas model kami. Dalam bahasa Inggris sederhana, dapat dikatakan sebagai
Rata-rata, prediksi kami meleset sekitar X.
=>On average, our predictions are off by about X.

#the prediction error
#eror = actual-predicted

#To calculate MAE, we first need a model.
"""
 
import pandas as pd

#definition data
file_path = "intro to ml\melb_data.csv"
melbroune_data=pd.read_csv(file_path)

#fillter rows dengan missing values
filltered_melbroune_data=melbroune_data.dropna(axis=0)
print(filltered_melbroune_data.columns)

#pilih y ->predictio target dan X->fitur
y=filltered_melbroune_data.Price
melbroune_fitur=["Rooms","Bathroom","Landsize","Lattitude","Longtitude","BuildingArea","YearBuilt"]
X=filltered_melbroune_data[melbroune_fitur]
print(X)
print(y)

from sklearn.tree import DecisionTreeRegressor
#DEFINE MODEL
melbroune_model=DecisionTreeRegressor(random_state=1)
#fit model
melbroune_model=melbroune_model.fit(X, y)

#calculate MAE VALIDATION EROR
from sklearn.metrics import mean_absolute_error

predicted_model_melbroune=melbroune_model.predict(X)
print("Prediction y \n =>",predicted_model_melbroune)
MEA=mean_absolute_error(y, predicted_model_melbroune)
print("MEA:\n =>",MEA)#IN SAMPEL SCORE

#The Problem with "In-Sample" Scores
"""
Ukuran yang baru saja kita hitung dapat disebut skor "dalam sampel". 
Kami menggunakan satu "sampel" rumah untuk membangun model dan mengevaluasinya. Inilah mengapa ini buruk.
Bayangkan, di pasar real estat yang besar, warna pintu tidak berhubungan dengan harga rumah.
Namun, dalam sampel data yang Anda gunakan untuk membangun model, semua rumah dengan pintu hijau sangat mahal.
Tugas model adalah menemukan pola yang memprediksi harga rumah, sehingga akan melihat pola ini, dan akan selalu memprediksi harga tinggi untuk rumah dengan pintu hijau.
Karena pola ini diturunkan dari data latih(TRAIN DATA), model akan tampak akurat dalam data latih.

Tetapi jika pola ini tidak berlaku ketika model melihat data baru, model tersebut akan sangat tidak akurat saat digunakan dalam praktik.
Karena nilai praktis model berasal dari membuat prediksi pada data baru, 
kami mengukur kinerja pada data yang tidak digunakan untuk membangun model.
Cara paling mudah untuk melakukannya adalah dengan mengecualikan beberapa data dari proses pembuatan model,
dan kemudian menggunakannya untuk menguji akurasi model pada data yang belum pernah dilihat sebelumnya. 
Data ini disebut data validasi.

"""

"""
Pustaka scikit-learn memiliki fungsi train_test_split untuk memecah data menjadi dua bagian.
Kami akan menggunakan beberapa data tersebut sebagai =>data pelatihan agar sesuai dengan model, 
dan kami akan menggunakan data lainnya sebagai data validasi untuk menghitung=> mean_absolute_error.
"""
from sklearn.model_selection import train_test_split
#train data dan testing data

# pisahkan data menjadi data pelatihan dan validasi, baik untuk fitur maupun target
# Pemisahan didasarkan pada random number. Memberikan nilai numerik ke
# argumen random_state menjamin kita mendapatkan pembagian yang sama setiap kali kita
# jalankan skrip ini.

train_x,val_x,train_y,val_y=train_test_split(X,y,random_state=0)
print("x_train:",train_x)
print("x_val:",val_x)
print("y_train",train_y)
print("y_val:",val_y)
#define model
melbroune_data = DecisionTreeRegressor()
#fit model->train data
melbroune_data=melbroune_data.fit(train_x,train_y)
print("Melbroune_data:",melbroune_data)
#get prediction price pada validation data->testing data /prediciton data
val_predictions=melbroune_data.predict(val_x)
print(val_predictions)
MEA=mean_absolute_error(val_y, val_predictions)
print(MEA)
"""
Rata-rata kesalahan absolut Anda untuk data dalam sampel adalah sekitar 500 dolar. Di luar sampel harganya lebih dari 250.000 dolar.
Inilah perbedaan antara model yang hampir tepat, dan yang tidak dapat digunakan untuk sebagian besar tujuan praktis. Sebagai acuan,
nilai rumah rata-rata dalam data validasi adalah 1,1 juta dolar. Jadi kesalahan dalam data baru adalah sekitar seperempat dari nilai rumah rata-rata.
Ada banyak cara untuk meningkatkan model ini, seperti bereksperimen untuk menemukan fitur yang lebih baik atau jenis model yang berbeda.
"""