# Doğrulama — imagehub 0.1

4 Ekim 2026'da CPU üzerinde doğrulandı. Ortam: Python 3.13.11 / PyTorch 2.14.1.
Referans: diffusers 0.40.0, transformers 5.18.0. Makine: Apple M2 Max, 96 GB birleşik bellek,
12 CPU thread'i. Kaynaklar, commit'ler ve dosya parmak izleri [validation.json](validation.json) dosyasındadır.

## Kapsam

- **5 checkpoint, tam ağırlık (`full`):** `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4`, `sd-v1-5`.
  Yayımlanmış tensörler strict yüklendi. Aynı başlangıç gürültüsüyle 50 adımlık üretim diffusers ile
  karşılaştırıldı.
- Bütün karşılaştırmalarda gözlenen fark **tam olarak 0**. Yerel uygulama, CPU'da float32 ile
  diffusers'ın ürettiği görselin bit düzeyinde aynısını üretiyor.
- Bu ifade bir kalite benchmark'ı anlamına gelmez.

## Karşılaştırmalar (CPU, float32)

| Karşılaştırma | Checkpoint | Gözlenen en büyük fark | Tolerans |
| --- | --- | --- | --- |
| Tokenizer token id'leri | v1.4 (14 prompt) ve v1.1–v1.5 (her biri 1.000 rastgele prompt) | birebir | birebir |
| CLIP metin kodlayıcı | v1.4 | 0 | 0 |
| U-Net tek adım (t = 981, 501, 1) | v1.4 | 0 | 0 |
| VAE decode / encode | v1.4 | 0 / 0 | 0 |
| Seed'den başlangıç gürültüsü | — | birebir | birebir |
| Uçtan uca: son latent / piksel | v1.1 | 0 / 0 | 0 / 0 |
| Uçtan uca: son latent / piksel | v1.2 | 0 / 0 | 0 / 0 |
| Uçtan uca: son latent / piksel | v1.3 | 0 / 0 | 0 / 0 |
| Uçtan uca: son latent / piksel | v1.4 | 0 / 0 | 0 / 0 |
| Uçtan uca: son latent / piksel | v1.5 | 0 / 0 | 0 / 0 |
| Uçtan uca, seed 42 (piksel) | v1.5 | 0 | 0 |

Uçtan uca testlerin ayarları: prompt "a photograph of an astronaut riding a horse", 50 adım,
guidance 7.5, 512×512, başlangıç gürültüsü `seed=1234`. diffusers tarafında `safety_checker=None`
ve `feature_extractor=None` kullanıldı.

Toleranslar şu kuralla belirlenir: ölçülen fark 0 ise tolerans 0, değilse bir üst onluk basamak.
Bütün ölçümler 0 çıktığı için hiçbir toleransın gevşetilmesi gerekmedi.

## Cihaz duman testleri (parity iddiası yok)

| Cihaz | Sonuç | CPU'ya göre en büyük latent farkı | En büyük piksel farkı | PSNR |
| --- | --- | --- | --- | --- |
| MPS (Apple M2 Max) | sonlu, 512×512, 20 adım | 2,7e-3 | 1 | 72,8 dB |
| CUDA | henüz çalıştırılmadı | | | |

## Sınırlar

- Parity iddiası yalnızca CPU'da float32 için, diffusers 0.40.0 ve transformers 5.18.0'a karşıdır.
- MPS ve CUDA çalıştırmaları duman testidir: sonlu çıktı ve doğru boyut kontrol edilir, CPU'ya göre
  fark kaydedilir. Bu cihazlarda parity iddiası yoktur.
- float16 ve bfloat16 için parity iddiası yoktur.
- Safety checker ve invisible watermark yoktur. Referans `safety_checker=None` ile çalıştırıldı.
- Yalnızca checkpoint'lerle yayımlanan PNDM (PLMS) zamanlayıcısı uygulanmıştır.
- Görsel kalite benchmark'ı (FID, CLIP score) yapılmadı.
- Yalnızca metinden görsel üretimi desteklenir. SD 1.x'in img2img kullanımı uygulanmadı.

## Otomatik testler

- Normal testler: 133 geçti, 36 atlandı (atlananlar isteğe bağlı referans testleridir). Testler ağdan
  model indirmez.
- Referans testleri (CPU): 34 geçti. Bunlara rastgele ağırlıklı yapısal parity, küçük modelle
  uçtan uca parity, gerçek checkpoint parity'si ve tokenizer fuzz testleri dahildir.
- MPS duman testi: 1 geçti.

## Doğrulama sırasında bulunan ve düzeltilen farklar

Gerçek checkpoint'lerle yapılan ilk koşu, rastgele ağırlıklı testlerin gösteremediği iki tokenizer
farkını ortaya çıkardı. İkisi de düzeltildi ve kalıcı fuzz testleriyle korunuyor:

1. **Özel token'ların `normalized` bayrağı.** `<|startoftext|>` normalize edilmiş metinde eşleşir,
   `<|endoftext|>` ise yalnızca ham metindeki tam yazımıyla eşleşir. Bunun sebebi, transformers'ın
   özel token'ları sırayla kaydetmesi ve son kaydın geçerli olmasıdır.
2. **Özel token olarak tanınmayan dizilerin bölünmesi.** Bu diziler, referans ön ayırıcıdaki gibi
   `<|`, `endoftext`, `|>` biçiminde üçe bölünür.
