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

![mnist-back-image](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/_/rsrc/1392048832517/variations-on-the-mnist-digits/mnist_back_image.png)

mnist-back-rand: a random background was inserted in the digit image. Each pixel value of the background was generated uniformly between 0 and 255;

![mnist-back-rand](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/_/rsrc/1392048835888/variations-on-the-mnist-digits/mnist_back_random.png)

mnist-rot-back-image: the perturbations used in mnist-rot and mnist-back-image were combined.

![mnist-rot-back-image](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/_/rsrc/1392048838794/variations-on-the-mnist-digits/mnist_rot_back_image.png)

```
the workflow will be as follows:

Simple MNIST data set
* library input
* data gatherig
* data visualization
* data exploration
* data tranformation
* data loading
* model building
* model copiling
* model evaluation
* accuracy, precision, recall, F1, kappa, confusion_matrix


Noisy MNIST data set
* library input
* data gatherig
* data visualization
* data exploration
* data tranformation
* data loading
* model building
* model copiling
* model evaluation
* accuracy, precision, recall, F1, kappa, confusio

İletişim detayları: Ali.saadat81@gmail.com
