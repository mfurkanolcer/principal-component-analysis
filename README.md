
# Denetimsiz Öğrenme Algoritmaları (Unsupervised Learning)

Denetimsiz makine öğrenimi; algoritmaları eğitmek için kullanılan bilgilerin sınıflandırılmadığı veya etiketlenmediği durumlarda kullanılır. Denetimsiz öğrenme, sistemlerin etiketlenmemiş verilerden gizli bir yapıyı açıklamak için bir işlevi nasıl çıkarabileceğini inceler.

Modeli denetlemenize gerek olmayan bir makine öğrenme tekniğidir. Bunun yerine, modelin bilgileri keşfetmek için kendi başına çalışmasına izin vermeniz gerekir.

Denetimsiz Öğrenmenin temel görevi verileri sınırlandırmaktır. Google fotoğraflardaki yüzler, hayvanlar ve bitkiler sınıflandırması denetimsiz öğrenmeye örnek gösterilebilir.

![Denetimsiz Öğrenme](https://user-images.githubusercontent.com/58481075/209712293-153bc35a-2bcd-4640-8622-d2af69956eae.png)

## Boyut İndirgeme (Dimensionality Reduction)

Bir veri kümesi için giriş değişkenlerinin veya özelliklerinin sayısına boyutsallık denilmektedir. Fazla girdi özelliği genellikle öngörücü modelleme görevini zorlaştırır, daha genel olarak bu durum, boyutsallığın laneti olarak adlandırılmaktadır.

Veri görselleştirme için genellikle yüksek boyutsallık istatistikleri ve boyutsallık azaltma teknikleri kullanılır. Bununla birlikte, bu teknikler, öngörücü modele daha iyi uyması için sınıflandırma veya regresyon veri kümesini basitleştirmek amacıyla uygulamalı makine öğreniminde de kullanılabilir.

Boyutsal küçültme (dimensionality reduction) yöntemi ise temel olarak, ele alınan rastgele değişken veya öznitelik sayısını azaltma işlemidir.

![Boyut İndirgeme](https://user-images.githubusercontent.com/58481075/209712303-639f8424-cd8f-4e83-88a2-969357b9c17e.png)


Temel Bileşen Analizi (PCA) yüksek boyutlu bir veri setinin boyutunu azaltmak için kullanılan en yaygın yöntemlerden biridir.

### Temel Bileşen Analizi (PCA)

Temel Bileşenler Analizi, çok değişkenli bir veri seti içerisindeki bilgiyi daha az değişkenle ve minimum bilgi kaybıyla açıklamanın bir matematiksel tekniğidir. Başka bir tanımla PCA, çok sayıda birbiri ile ilişkili değişkenler içeren veri setinin boyutunu, veri seti içerisindeki veriyi koruyarak daha küçük boyuta indirgenmesini sağlayan bir dönüşüm tekniğidir.

PCA, büyük boyutlu veri setlerindeki boyutsallığı azaltır. Teknik, boyut küçültme işleminde veri seti içerisindeki değişken sayısını azaltmayı hedefler. Dönüşüm sonrasında elde edilen değişkenler ilk değişkenlerin temel bileşenleri olarak adlandırılır. İlk temel bileşen olarak varyans değeri en büyük olan seçilir ve diğer temel bileşenler varyans değerleri azalacak şekilde sıralanır.



### PCA’nın Özellikleri

- Boyut azaltmada çok faydalı bir yöntemdir.
- Çok boyutlu verileri yaklaşık olarak ve daha az boyutlu veriyle temsil eder.
- Orijinal veriler için dik-olan-en-büyük-varyans-yönleri bulup orijinal verileri bu koordinat sisteminde gösterir.
- Çok boyutlu verilerin görsel gösterilmesi ve incelenmesi için kullanılabilir.
- Makine öğrenmesi olarak, verilerin boyutu azaltabilir–az değişen PCA özellikleri modelleme için önemsiz olabilir, bu şekilde modelleme ile ilgili hesaplama hızlandırabilir.
- Veri sıkıştırma içinde kullanılabilir.

### PCA’nın Amaçları

- Verilerin boyutunu azaltma
- Tahminleme yapma
- Veri setini, bazı analizler için görüntülemek


![Temel Bileşen Analizi](https://user-images.githubusercontent.com/58481075/209712311-cc5853aa-8d1f-4b4f-a770-7ecc005f627e.png)


Çok boyutlu verilere doğru açıdan bakarak genellikle verideki ilişkiler açıklanabilir. PCA’nın amacı bu “doğru açıyı” bulmaktadır.

PCA kilit noktası, problemi çözmek üzere görsel inceleme için uygun bir “açı” yani uygun bir koordinat sistemi seçmektir. Uygun “açıdan” verilere bakmak, bu koordinat sistemi kullanarak verileri incelemek demektir.

### Kaynakça
- https://blog.turhost.com/makine-ogrenmesi-machine-learning-nedir/
- https://www.datasciencearth.com/boyutsal-kucultme-dimensionality-reduction/
- https://www.veribilimiokulu.com/makineler-nasil-ogrenir/
  
