# Noisy-MNIST
Python CNN
[Düğün](https://dugun.com/) şirketi tarafından talep edilen bu kod mücadelesi. Buradaki aday __"Ali Saadat"__, yalnızca takip kodlarını geliştirdiğini ve topladığını onaylar. Görev, adayın aşağıdaki görevleri tamamlamasını gerektirir:


* Beyaz arka plan üzerine el yazısı ile yazilan rakamı ne oldugu tahmin edilecek.

* Bonus: Karelli-cizgili arka plan ile aynı işlem yapılacak.

Görevleri başarmak için meseleye böl ve yaklaşma yöntemiyle yaklaşıyorum. İlk önce, MNIST veri setini tanıyalım. Bu sayfadan elde edilebilen MNIST el yazısı rakamları veri tabanı, 60.000 örnek eğitim setine ve 10.000 örnek test setine sahiptir. MNIST'ten alınabilecek daha büyük bir setin alt kümesidir. Rakamlar boyut normalleştirildi ve sabit boyutlu bir görüntüde ortalandı.

![MNIST](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset.png)
---
## Getting Started
Ön işleme ve biçimlendirme konusunda asgari çaba harcayarak gerçek dünyadaki veriler üzerinde öğrenme tekniklerini ve örüntü tanıma yöntemlerini denemek isteyen insanlar için iyi bir veritabanıdır.

It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.


ilk önce sorunun basit versiyonuna değinilecek, daha sonra gürültülü veri olan daha zorlu bölüme devam edeceğim. Aşağıdaki gibi, MNIST veri setinin gürültülü veri varyantlarından bazılarını görüntüleyebilirsiniz.

at first the simple version of the problem will be addressed, next, I will continue to the more challenging part which is noisy data. As following you can view some of the noisy data variants of the MNIST data set. 


mnist-back-image: a patch from a black and white image was used as the background for the digit image. The patches were extracted randomly from a set of 20 images downloaded from the internet. Patches which had low pixel variance (i.e. contained little texture) were ignored;

```
