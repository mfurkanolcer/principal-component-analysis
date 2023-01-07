
# Boyut İndirgeme ve Temel Bileşenler Analizi (Principal Component Analysis)

## 1. Boyut İndirgeme (Dimensionality Reduction)

Bir veri kümesi için giriş değişkenlerinin veya özelliklerinin sayısına boyutsallık denilmektedir. Fazla girdi özelliği genellikle öngörücü modelleme görevini zorlaştırır, daha genel olarak bu durum, boyutsallığın laneti olarak adlandırılmaktadır. Veri görselleştirme için genellikle yüksek boyutsallık istatistikleri ve boyutsallık azaltma teknikleri kullanılır. Bununla birlikte, bu teknikler, öngörücü modele daha iyi uyması için sınıflandırma veya regresyon veri kümesini basitleştirmek amacıyla uygulamalı makine öğreniminde de kullanılabilir. Boyutsal küçültme (dimensionality reduction) yöntemi ise temel olarak, ele alınan rastgele değişken veya öznitelik sayısını azaltma işlemidir. İstatistikte, makine öğreniminde ve bilgi teorisindeki kullanımında bir dizi temel değişken oluşturularak dikkate alınan rastgele değişkenlerin sayısının azaltılması sağlanır. Bu yönüyle veri bilimi ve makine öğreniminde oldukça önemli bir konumdadır. Örneğin; makine öğrenimi sınıflandırma problemlerinde, genellikle son sınıflandırmanın yapıldığı temelde çok fazla faktör bulunmaktadır. Bu faktörler temel olarak özellikler (features) adı verilen değişkenlerdir. Özellik sayısı arttıkça, eğitim setini görselleştirmek ve daha sonra üzerinde çalışmak zorlaşmaktadır. Bu özelliklerin çoğu bazen, birbiriyle ilişkilidir ve bu nedenle gereksiz olarak değerlendirilebilir. İşte bu gereksizlik karşısında, boyutsal küçültme algoritmaları devreye girmektedir. Bu yaklaşımların türleri başlıca; özellik seçimi (feature selection) ve özellik çıkarma (feature extraction) bileşenleri olarak ikiye ayrılmaktadır.

![Boyut İndirgeme](https://user-images.githubusercontent.com/58481075/209712303-639f8424-cd8f-4e83-88a2-969357b9c17e.png)

Makine öğrenimi algoritmalarının performansı çok fazla giriş değişkeni ile bozulabilir. N boyutlu bir özellik uzayındaki boyutları temsil eden veri sütunlarını ve veri satırlarını bu alandaki noktalar olarak düşünürsek, özellik alanında çok sayıda boyuta sahip olmak, o alanın hacminin çok büyük olduğu anlamına gelir ve buna karşılık, o alanda sahip olduğumuz noktalar (veri satırları) genellikle küçük ve temsili olmayan bir örneği temsil eder. Bu, bir veri kümesinin yararlı bir geometrik yorumudur.

### 1.1 Boyutsal Küçültmenin Bileşenleri
#### 1.1.1 Özellik Seçimi (Feature Selection)

Özellik seçimi yaklaşımları, giriş değişkenlerinin bir alt kümesini bulmaya çalışır (özellikler veya öznitelikler olarak da adlandırılır).  

Başlıca üç yaklaşım şunlardır:  

- Filtre metodu (Filter method): Bilgi kazancı,
- Sarma metodu (Wrapper method): Doğrulukla yönlendirilen arama,
- Gömülü metodu (Embedded method): Seçilen özellikler model tahminlerine dayanarak oluşturulurken eklenir veya kaldırılır.
Regresyon veya sınıflandırma gibi veri analizleri, azaltmanın gerçekleştirildiği alanda, orijinal alandakinden daha doğru şekilde yapılabilir.  

Özellik seçimi yaklaşımlarının kullanım sebepleri:  
- Model eğitim (model training) sürelerinin daha kısa olmasını sağlamak,
- Araştırmacıların/kullanıcıların yorumlamasını kolaylaştırmak için modellerin basitleştirilmesi,
- Boyutsallığın lanetinden kaçınmak,
- Varyansın azaltılması    
  
#### 1.1.2 Özellik Çıkarma (Feature Extraction) ya da Özellik Projeksiyonu (Feature Projection)  

Makine öğreniminde (machine learning), örüntü tanımada (pattern recognition) ve görüntü işlemede (image processing), özellik çıkarma ilk ölçülen veri kümesinden başlanır ve bilgilendirici ve gereksiz olması amaçlanan türetilmiş değerler (özellikler) oluşturulur, sonraki öğrenme ve genelleme adımlarını kolaylaştırır.

Bir algoritmaya sunulan giriş verilerinin işlenemeyecek kadar büyük olması ve fazlalık olduğundan şüpheleniliyorsa, azaltılmış bir kümeye dönüştürülebilir. Seçilen özelliklerin, giriş verilerinden ilgili bilgileri içermesi beklenir, böylece istenen görev, tam başlangıç verileri yerine bu azaltılmış gösterimi kullanarak gerçekleştirilebilir.  

### 1.2 Boyutsal İndirgeme Avantajları
- Gerekli zamanı ve depolama alanını azaltır.
- Çoklu eşlenimsizliğin (multi-collinearity) giderilmesi, makine öğrenme modelinin parametrelerinin yorumlanmasını geliştirir.
- 2-boyutlu veya 3-boyutlu gibi çok düşük boyutlara küçültüldüğünde verileri görselleştirmek daha kolay hale gelir.
- Yazımızın başında da belirtiğimiz boyutsallığın lanetinden (curse of dimensionality) kaçınmayı sağlar.  

### 1.3 Boyutsal İndirgeme Dezavantajları
- Bir miktar veri kaybına yol açabilir.
- PCA, değişkenler arasında bazen istenmeyen doğrusal korelasyonlar bulma eğilimindedir.
- Ortalama ve kovaryansın veri setlerini tanımlamak için yeterli olmadığı durumlarda PCA başarısız olur.
- Pratikte kaç ana bileşenin saklanacağını bilemeyebiliriz.

## 2. Temel Bileşenler Analizi (PCA)
Temel Bileşenler Analizi, çok değişkenli bir veri seti içerisindeki bilgiyi daha az değişkenle ve minimum bilgi kaybıyla açıklamanın bir matematiksel tekniğidir. Başka bir tanımla PCA, çok sayıda birbiri ile ilişkili değişkenler içeren veri setinin boyutunu, veri seti içerisindeki veriyi koruyarak daha küçük boyuta indirgenmesini sağlayan bir dönüşüm tekniğidir.

PCA, büyük boyutlu veri setlerindeki boyutsallığı azaltır. Teknik, boyut küçültme işleminde veri seti içerisindeki değişken sayısını azaltmayı hedefler. Dönüşüm sonrasında elde edilen değişkenler ilk değişkenlerin temel bileşenleri olarak adlandırılır. İlk temel bileşen olarak varyans değeri en büyük olan seçilir ve diğer temel bileşenler varyans değerleri azalacak şekilde sıralanır.  

Temel Bileşen Analizi (PCA), çok boyutlu uzaydaki bir verinin daha düşük boyutlu bir uzaya izdüşümünü, varyansı maksimizde edecek şekilde bulma yöntemidir. Uzayda bir noktalar kümesi için tüm noktalara ortalama uzaklığı en az olan “en uygun doğru” seçilir. Daha sonra bu doğruya dik olanlar arasından yine en uygun doğru seçilerek bu adımlar, yeni bir boyutun varyansı belirli bir eşiğin altına inene kadar tekrarlanır.   

Bu sürecin sonunda elde edilen doğrular bir doğrusal uzayın tabanlarını oluşturur. Bu taban vektörüne temel bileşen adı verilir. Temel bileşenlerin 3 farklı özelliği vardır ;
- Verinin temel bileşenleri birbirinden bağımsızdır.
- Birinci temel bileşen toplam değişkenliği en çok açıklayan bileşendir.
- Bir sonraki temel bileşen de kalan değişkenliği en çok açıklayan bileşendir.

PCA verideki gerekli bilgileri ortaya çıkarmada oldukça etkili bir yöntemdir. PCA’in arkasında yatan temel mantık çok boyutlu bir veriyi, verideki temel özellikleri yakalayarak daha az sayıda değişken ile göstermektir.


### 2.1 PCA'nın Özellikleri

- Boyut azaltmada çok faydalı bir yöntemdir.
- Çok boyutlu verileri yaklaşık olarak ve daha az boyutlu veriyle temsil eder.
- Orijinal veriler için dik-olan-en-büyük-varyans-yönleri bulup orijinal verileri bu koordinat sisteminde gösterir.
- Çok boyutlu verilerin görsel gösterilmesi ve incelenmesi için kullanılabilir.
- Makine öğrenmesi olarak, verilerin boyutu azaltabilir–az değişen PCA özellikleri modelleme için önemsiz olabilir, bu şekilde modelleme ile ilgili hesaplama hızlandırabilir.
- Veri sıkıştırma içinde kullanılabilir.

### 2.2 PCA’nın Amaçları

PCA'nın amacı, veri kümesindeki en önemli özellikleri veya değişkenleri bulmak ve orijinal değişkenlerin doğrusal kombinasyonları olan yeni bir değişken kümesi oluşturmaktır. Temel bileşenler olarak adlandırılan bu yeni değişkenler, orijinal verileri daha düşük boyutlu bir uzayda temsil etmek için kullanılabilir ve birbirlerine ortogonaldir, yani bağımsızdırlar ve herhangi bir gereksiz bilgi içermezler.  
- Verilerin boyutunu azaltma
- Tahminleme yapma
- Veri setini, bazı analizler için görüntülemek


![Temel Bileşen Analizi](https://user-images.githubusercontent.com/58481075/209712311-cc5853aa-8d1f-4b4f-a770-7ecc005f627e.png)


Çok boyutlu verilere doğru açıdan bakarak genellikle verideki ilişkiler açıklanabilir. PCA’nın amacı bu “doğru açıyı” bulmaktadır.

PCA kilit noktası, problemi çözmek üzere görsel inceleme için uygun bir “açı” yani uygun bir koordinat sistemi seçmektir. Uygun “açıdan” verilere bakmak, bu koordinat sistemi kullanarak verileri incelemek demektir.

### 2.3 PCA'nın Arkasındaki Matematik  
PCA, denetimsiz bir öğrenme sorunu olarak düşünülebilir. Ham bir veri kümesinden temel bileşenleri elde etme sürecinin tamamı altı kısımda basitleştirilebilir:

- d+1 boyutlarından oluşan tüm veri setini alın ve yeni veri setimiz d boyutlu olacak şekilde etiketleri yok sayın.
- Tüm veri kümesinin her boyutu için ortalamayı hesaplayın.
- Tüm veri kümesinin kovaryans matrisini hesaplayın.
- Özvektörleri ve karşılık gelen özdeğerleri hesaplayın.
- Özvektörleri özdeğerleri azaltarak sıralayın ve bir d × k boyutlu W matrisi oluşturmak için en büyük özdeğerlere sahip k özvektörleri seçin.
- Örnekleri yeni alt uzaya dönüştürmek için bu d × k özvektör matrisini kullanın.    

ADIM 1:  

d+1 boyutlarından oluşan tüm veri setini alın ve yeni veri setimiz d boyutlu olacak şekilde etiketleri yok sayın.  

Diyelim ki d+1 boyutlu bir veri setimiz var . Modern makine öğrenimi paradigmasında d' nin X_train ve 1'in y_train (etiketler) olarak düşünülebileceği yer. Böylece, X_train + y_train , eksiksiz tren veri setimizi oluşturur.

Böylece, etiketleri bıraktıktan sonra d boyutlu veri kümesiyle kalırız ve bu, temel bileşenleri bulmak için kullanacağımız veri kümesi olacaktır. Ayrıca, örneğin d = 3 etiketlerini göz ardı ettikten sonra elimizde üç boyutlu bir veri kümesi kaldığını varsayalım.

Örneklerin iki farklı sınıftan kaynaklandığını varsayacağız, burada veri kümemizin bir yarısı sınıf 1 ve diğer yarısı sınıf 2 olarak etiketlenmiştir.

Veri matrisimiz X , üç öğrencinin puanı olsun:

![1_1HD7YIaVhfUjQ2ARKi7gNA](https://user-images.githubusercontent.com/58481075/211147629-a9689f8e-9dea-482c-ad4b-f3182bd5d6b3.png)

ADIM 2:  
Tüm veri kümesinin her boyutunun ortalamasını hesaplayın.  

Yukarıdaki tablodan elde edilen veriler, matristeki her sütunun bir testteki puanları ve her satırın bir öğrencinin puanını gösterdiği A matrisinde temsil edilebilir.  

![1_a0-0h6YJsVtH9lG9A1_-VQ](https://user-images.githubusercontent.com/58481075/211147666-aff179e6-6c7f-42d6-bdc8-a419ab3a0c05.png)  

Yani, A matrisinin ortalaması,  

![1_r1zlnStJxBq8buNZLhyT7A](https://user-images.githubusercontent.com/58481075/211147763-a8c050e7-8336-4831-b878-4cad32cf6ea9.png)  

ADIM 3:  
Tüm veri kümesinin kovaryans matrisini hesaplayın (bazen varyans-kovaryans matrisi olarak da adlandırılır).  

Aşağıdaki formülü kullanarak iki değişken X ve Y'nin kovaryansını hesaplayabiliriz:  

![1_xNSxC6LtyrOwFAe8dfTWgA](https://user-images.githubusercontent.com/58481075/211147836-4bc73e67-c348-4ac9-8b8e-db38c9039919.png)  

Yukarıdaki formülü kullanarak, A'nın kovaryans matrisini bulabiliriz . Ayrıca sonuç , d × d boyutlarında bir kare matris olacaktır.

Orijinal matrisimizi şu şekilde yeniden yazalım  

![1_LB9qyXaROHAKqlaqk_LWdQ](https://user-images.githubusercontent.com/58481075/211147862-b9a6e483-8b01-41dd-b638-2e5ebfc51364.png)  

C kovaryans matrisi,  

![1_E0ImSfZ3Rea3Y-tWnX1cEQ](https://user-images.githubusercontent.com/58481075/211147896-390bfeb3-7c17-4dd8-aff3-be1e43aa9267.png)
 
- Köşegen boyunca Mavi ile gösterilen, her test için puanların varyansını görüyoruz. Sanat testi en büyük varyansa sahiptir (720); ve İngilizce testi, en küçüğü (360). Yani sanat testi puanlarının İngilizce testi puanlarına göre daha fazla değişkenliğe sahip olduğunu söyleyebiliriz.
- Kovaryans, A matrisinin köşegen dışı öğelerinde siyah olarak gösterilir.  

a) Matematik ve İngilizce arasındaki kovaryans pozitiftir (360) ve matematik ile sanat arasındaki kovaryans pozitiftir (180). Bu, puanların olumlu yönde değişme eğiliminde olduğu anlamına gelir. Matematik puanları yükseldikçe, sanat ve İngilizce puanları da yükselme eğilimindedir; ve tersi.

b) Bununla birlikte, İngilizce ve sanat arasındaki kovaryans sıfırdır. Bu, İngilizcenin hareketi ile sanat puanları arasında tahmin edilebilir bir ilişki olmadığı anlamına gelir.

ADIM 4:  
Özvektörleri ve karşılık gelen Özdeğerleri Hesaplayın  

Sezgisel olarak, bir özvektör, kendisine doğrusal bir dönüşüm uygulandığında yönü değişmeden kalan bir vektördür.

Şimdi, yukarıda sahip olduğumuz kovaryans matrisinden özdeğer ve özvektörleri kolayca hesaplayabiliriz.

A bir kare matris, ν bir vektör ve λ A ν = λ ν'yı karşılayan bir skaler olsun , o zaman λ , A'nın özvektörü ν ile ilişkili özdeğer olarak adlandırılır .

A'nın özdeğerleri , karakteristik denklemin kökleridir.  

![1_4NSJKK38x5Db3DlB2TVQcg](https://user-images.githubusercontent.com/58481075/211148358-73933ab6-27a9-46d7-86df-e13e509c8336.png)  

Önce det(A-λI) hesaplandığında , I bir birim matristir:  

![1_DogIPZUWEUpvdT3YZSVwCA](https://user-images.githubusercontent.com/58481075/211148443-8388eb72-7de7-4fd4-9615-ef410ec38c8c.png)  

![1_GrDal75rL0YcMdCNIlijuQ](https://user-images.githubusercontent.com/58481075/211148463-076d37d3-0930-4201-b95e-4cb794b36360.png)

![1_4L8Ip319TwuAdWoLwofxNA](https://user-images.githubusercontent.com/58481075/211148469-4f821ab1-d89f-4f28-9069-71d39bcfc698.png)  

Artık basitleştirilmiş matrisimiz olduğuna göre, bunun determinantını bulabiliriz:  

![1_BzktUJKs11-rFIH6_GAI4w](https://user-images.githubusercontent.com/58481075/211148495-6d4ffc0b-6e84-4248-a578-274c4d481c7a.png)

![1_pclbXVQlph0muCKfuMg4kA](https://user-images.githubusercontent.com/58481075/211148497-9d2511f5-18d2-44d5-a829-b003b366d6ea.png)  

Şimdi denklemimiz var ve matrisin özdeğerini elde etmek için λ'yı çözmemiz gerekiyor. Yani, yukarıdaki denklemi sıfıra eşitlemek:  

![1_Gps62pfonm5ZeC5GwD2X6g](https://user-images.githubusercontent.com/58481075/211148520-5b09a489-fe46-46a3-a28c-763f17859e3e.png)  

Bu denklemi λ değeri için çözdükten sonra aşağıdaki değeri elde ederiz.  

![1_6KgzE41U5GTPcFG187zt0g](https://user-images.githubusercontent.com/58481075/211148539-bc4b801c-3f7a-4e85-a5b6-b702ff457bfe.png)  

Şimdi, yukarıdaki özdeğerlere karşılık gelen özvektörleri hesaplayabiliriz. Özvektörleri çözdükten sonra , karşılık gelen özdeğerler için aşağıdaki çözümü elde ederiz.  

![1_cfDBspXxFBGJ3yIhm5tXGA](https://user-images.githubusercontent.com/58481075/211148573-907a3d68-28d5-4ec7-94a5-601ab56c66e1.png)  

ADIM 5:  
Özvektörleri azalan özdeğerlere göre sıralayın ve d × k boyutlu bir W matrisi oluşturmak için en büyük özdeğerlere sahip k özvektörleri seçin.

Özellik uzayımızın boyutsallığını azaltmak, yani özvektörlerin bu yeni özellik alt uzayının eksenlerini oluşturacağı daha küçük bir alt uzaya PCA aracılığıyla özellik uzayını yansıtmak amacıyla başladık. Bununla birlikte, özvektörler yalnızca yeni eksenin yönlerini tanımlar, çünkü hepsi aynı birim uzunluğa 1 sahiptir.

Bu nedenle, alt boyutlu alt uzayımız için hangi özvektörleri bırakmak istediğimize karar vermek için, özvektörlerin karşılık gelen özdeğerlerine bir göz atmalıyız. Kabaca söylemek gerekirse, en düşük özdeğere sahip özvektörler, verinin dağılımı hakkında en az bilgiyi taşırlar ve bırakmak istediklerimiz bunlardır.

Ortak yaklaşım, özvektörleri en yüksekten en düşüğe karşılık gelen özdeğere göre sıralamak ve en üstteki k özvektörleri seçmektir.

Böylece, özdeğerleri azalan düzende sıraladıktan sonra,  

![1_ZJOOlx0T7JtqIiyD0cgVgw](https://user-images.githubusercontent.com/58481075/211148616-328d5b3a-7682-46b3-a633-ee63cf5dde10.png)  

3 boyutlu bir özellik uzayını 2 boyutlu bir özellik alt uzayına indirgediğimiz basit örneğimiz için, d×k boyutlu özvektör matrisimizi W oluşturmak için en yüksek özdeğerlere sahip iki özvektörü birleştiriyoruz.

Dolayısıyla, iki maksimum özdeğere karşılık gelen özvektörler :

![1_LdwD0hzCfPuvrbpuZMcB4A](https://user-images.githubusercontent.com/58481075/211148631-ab6f2123-9eb5-4ac2-8f40-423456021fbc.png)  

ADIM 6:  
Örnekleri yeni alt uzaya dönüştürün

Son adımda, az önce hesapladığımız 3x2 boyutlu W matrisini y = W' × x denklemi aracılığıyla örneklerimizi yeni alt uzaya dönüştürmek için kullanıyoruz ; burada W' , W matrisinin devriktir .

Son olarak, iki ana bileşenimizi hesapladık ve veri noktalarını yeni alt uzaya yansıttık.

### 2.4 PCA'nın Kullanım Alanları


#### 2.4.1 Genetik

Genetik çeşitliliğin coğrafi konum ve etnik kökene göre dağılımı,bir ırkın yaşadığı tarihsel demografik olaylar ve süreçler hakkında geniş bir bilgi kaynağı sağlar.Bununla birlikte kolonizasyon, izolasyon, göç ve karışım gibi süreçlerin doğası ve zamanlaması hakkında çıkarımlar yapmak çok zorlaşabilir. Temel Bileşen Analizi de genetik varyasyonun coğrafi konum ve etnik kökene dağılımındaki yapıyı belirlemek için yaygın olarak kullanılır. 
          
#### 2.4.2 Sağlık

Elektronik Sağlık Hizmeti kayıtları kullanan klinik araştırmalar genellikle çok sayıda değişken sunar. Bu değişkenler sıklıkla birbirleriyle ilişkilidir ve bu da  regresyon modellerinde çoklu bağlantıya neden olur. Çoklu bağlantıdan etkilenen  tahminlerin büyük standart hataları olabilir ve bu tür tahminler üzerindeki çıkarımı daha az kesin hale getirir.
Bu tip bir sorun klinik çalışmalarda mevcuttur ve bu sorunla başa çıkmak için kullanılan yöntemlerden bir tanesi de Temel Bileşen Analizidir.   

#### 2.4.3 Enerji

Günümüzde fosil yakıtlar nedeniyle   artan küresel ısınma sorununa karşılık güneş enerjisi benzeri yenilenebilir enerjilere  yönelim artmaktadır. Ancak güneş enerjisi sistemlerinin sorunsuz  ve de sürekli çalışabilmesi için güneş ışınımının yoğunluğu ile ilgili birkaç dakika önceden bilgi alınmalıdır. Bunun için çeşitli modeller olsa da bu modeller çoğunlukla yüksek hesaplama süresi gerektirir. Hesaplama sürelerini azaltmak amacıyla veri boyutunu küçültülür ve bunun için temel bileşen analizi kullanılır.

#### 2.4.4 Makine Öğrenmesinde Kullanımı

Veri Bilimi çalışmalarında çok sayıda değişken ile çalışılması gerekebilir.Bu durum eğitim(training) süresinin fazla olması, aşırı öğrenme(overlifting) ve çoklu doğrusal bağlantı(multicollinearity) gibi sorunları beraberinde getirir.Hazırlanan modellerin optimum sürede ve performansla çalışması gerekecektir.

Bu problemleri aşmak için değişken seçimi ve boyut indirgeme yöntemleri kullanılabilir. Değişken seçiminde veri setindeki değişken korunur ya da tamamen kaldırılır. Boyut indirgemede ise mevcut değişkenlerin kombinasyonlarından oluşan yeni değişkenler yaratılarak değişken sayısı azaltılır. Böylece veri setindeki tüm özellikler hala mevcut ancak değişken sayısı azaltılmış olur.

Analizlerde yaşanan bu tip sorunları aşmak için en çok tercih edilen boyut indirgeme yöntemlerinden  birisi de Temel Bileşenler Analizidir.
Ayrıca Temel Bileşenler Analizi, yüz tanıma, resim sıkıştırma ve örüntü tanıma gibi alanlarda yaygın olarak kullanılmaktadır.

   ![image](https://user-images.githubusercontent.com/75726215/208492447-391ec063-983f-46c4-9de4-105d442f0d89.png)
   
#### 2.4.5 Görüntüler için PCA 

Bir makinenin görüntüleri okuyabildiğini veya sayı kullanmadan sadece görüntüleri kullanarak bazı hesaplamalar yapabileceğini birçok kez merak ediyor olmalısınız. Şimdi bunun bir kısmını cevaplamaya çalışacağız. Basit olması için, tartışmamızı yalnızca kare resimlerle sınırlayacağız. NxN piksel boyutunda herhangi bir kare görüntü, her öğenin görüntünün yoğunluk değeri olduğu bir NxN matrisi olarak temsil edilebilir. (Görüntü, tek bir görüntü oluşturacak şekilde piksel sıralarının birbiri ardına yerleştirilmesiyle oluşturulmuştur.) Yani bir dizi görüntünüz varsa, bu matrislerden bir matris oluşturabiliriz, bir dizi pikseli bir vektör olarak kabul ederek, biz üzerinde temel bileşen analizine başlamaya hazırdır. Nasıl faydalıdır?

Bir önceki kümenin parçası olmayan, tanımanız için size bir görüntü verildiğini varsayalım. Makine, tanınacak görüntü ile temel bileşenlerin her biri arasındaki farkları kontrol eder. PCA uygulanırsa ve farklılıklar 'dönüştürülmüş' matristen alınırsa, sürecin iyi performans gösterdiği ortaya çıkar. Ayrıca, PCA'yı uygulamak, fazla bilgi kaybetmeden bazı bileşenleri dışarıda bırakma ve böylece sorunun karmaşıklığını azaltma özgürlüğü verir.

Görüntü sıkıştırma için, daha az anlamlı özvektörleri çıkararak, aslında depolama için görüntünün boyutunu azaltabiliriz. Ancak, orijinal görüntünün çoğaltılmasından bahsetmek, bariz nedenlerden dolayı bazı bilgileri kaybedecektir.

### 2.5 PCA Örnek Uygulama
Görüntü sıkıştırma PCA’nın  yaygın bir uygulamasıdır. Şekilde 1200 x 795 piksel çözünürlükte çekilmiş bir ay resmi vardır. 

![Resim1](https://user-images.githubusercontent.com/58481075/211149248-c6f96728-c92a-44a5-8ec6-b9845c31cb1d.png)

```
moon <-- readJPEG(\moon.jpg")
moon_pca <- prcomp(moona center=FALSE)
percent_variance = sum((moon_pca$sdev[1:k])^2)/sum((moon_pca$sdev^2))

```

PCA sonuçları, temel bileşenlerle ilişkili standart sapmaları içeren moon pca nesnesinde depolanıyor. Kodun son satırındaki k, dikkate alınan temel bileşenlerin sayısını gösterir ve dolayısıyla varyans yüzdesini etkiler. 

Şekildeki tablo,  k değerine bağlı olarak temel bileşenler tarafından yakalanan yüzde varyansı gösterir (yani, dikkate alınan temel bileşenlerin sayısı)
Bu sonuçlara göre, yalnızca 10 Temel Bileşen seçerek görüntüyü  %98 doğrulukla yeniden oluşturabilirsiniz. Bu da, 10/795’lik bir sıkıştırmadır.

![Resim2](https://user-images.githubusercontent.com/58481075/211149338-e9cbe89f-35e6-43de-b6ef-e9ae05a9d1f6.png)  

Orijinal veriler, aşağıdaki kod alıntısı kullanılarak PC’lerden çoğaltılabilir.

5 temel bileşen ve 20 temel bileşen ile yeniden yapılandırılmış görseller aşağıda verilmiştir.

![Resim3](https://user-images.githubusercontent.com/58481075/211149371-c0d4ba97-c6af-4639-9f91-25960a44e002.png) 
5PC'li Görüntü  

![Resim4](https://user-images.githubusercontent.com/58481075/211149388-6e05754c-8fa8-48de-a7de-9f3c635a6881.png)
20PC'li Görüntü


### Kaynakça
- https://www.datasciencearth.com/boyutsal-kucultme-dimensionality-reduction/
- https://medium.com/@santoshshirol/principal-component-analysis-pca-4aa813c3cdd3
- https://www.cs.cmu.edu/~mgormley/courses/10701-f16/slides/lecture14-pca.pdf
- https://blog.turhost.com/makine-ogrenmesi-machine-learning-nedir/
- https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
- https://www.veribilimiokulu.com/makineler-nasil-ogrenir/
- https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec12-slides.pdf
