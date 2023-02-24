import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import os

path = os.path.dirname(__file__)   #dosya konumunu görmek
os.chdir(path)

def load_mnist():     #numpy arraylere dönüştürdüğüm resimleri tekrardan dahil ediyorum
    with open('mnist.pkl', 'rb') as f: #mnist.pkl dosyamızı rb modunda açıyorum, klosörü  f ismiyle kullanıyorum
        mnist = pickle.load(f)  #pickle dosyasını okumak
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']

train_x, train_y, test_x, test_y = load_mnist()  #fonksiyına atayarak değişkenleri değerlere eşitledik.

train_x, train_y, test_x, test_y = [pd.DataFrame(x) for x in [train_x, train_y, test_x, test_y]]
#pandasın dataframe yapısına atıp kendisine eşitledim.

train_x = train_x/255.0 #normalleştirme işlemi yaptım. Resimlerdeki tüm pixel değerlerini 0 ve 1 arasında boyutlandırdım.
test_x = test_x/255.0 #amaç makine öğrenmesi algoritmalarında matriks çarpımı yapılması

svc = SVC() #SVC classında svc örneği oluşturdum.

svc.fit(train_x, train_y.values.flatten())  #pandasın dataframebin numpy array formundaki halini alıyorum. flatten fonksiyonunu çağırarak arrayi bir boyutlu hale getirdim..

filename = "svm_model.pkl"     #modeli kaydettik.
pickle.dump(svc, open(filename, 'wb'))

y_pred = svc.predict(test_x)  #test veri setimizle test ediyorum.
print(classification_report(test_y, y_pred))  #classification report fonksiyonuyla test ettiğimiz verinin doğru cevaplarıyla bizim tahminlerimizi karşılaştırıp metriklerini ekrana bastırıyoruz.
