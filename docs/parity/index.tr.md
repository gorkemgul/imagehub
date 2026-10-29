---
title: Parity
---

# Parity

imagehub bir modeli ancak özgün uygulamasının çıktısını yeniden ürettikten sonra ekler. Bu bölüm
neyin, nasıl ölçüldüğünü ve iddianın nerede sona erdiğini gösterir.

<div class="ih-stats" markdown>
<div class="ih-stat" markdown><strong>{{ ih.validated }} / {{ ih.models }}</strong><span>kayıtlı uçtan uca kanıtı olan checkpoint</span></div>
<div class="ih-stat" markdown><strong>{{ ih.max-pixel-diff }}</strong><span>kaydedilen en büyük piksel farkı</span></div>
<div class="ih-stat" markdown><strong>CPU · fp32</strong><span>parity iddiasının kapsamı</span></div>
<div class="ih-stat" markdown><strong>{{ ih.gallery-count }}</strong><span>yan yana galerideki model</span></div>
</div>

## Yöntem

Her checkpoint için sırasıyla:

1. **Sıkı yükleme.** Yayımlanmış her tensör yüklenir; eksik, fazla ya da adı veya shape'i
   uyuşmayan herhangi bir tensör modeli başarısız kılar.
2. **Bileşenler.** Tokenizer, metin encoder'ları, transformer veya U-Net, VAE ve scheduler aynı
   girdilerle referansa karşı ayrı ayrı karşılaştırılır.
3. **Uçtan uca.** Aynı başlangıç gürültüsünden tam üretim referansla karşılaştırılır: son latent'lar
   ve son pikseller.
4. **Tolerans.** Kabul edilen fark çalıştırmadan önce yazılır; çoğu modelde tam olarak sıfırdır.

Kayıtlı referans, modelin diffusers {{ ih.diffusers }} ve transformers {{ ih.transformers }}
içindeki uygulamasıdır; bu kitaplıkların kapsamadığı modellerde resmî uygulama kullanılır. Ölçümler
PyTorch {{ ih.torch }} ile yapılır.

## Sonuçları okuma

| Sayfa | İçerik |
| --- | --- |
| [Galeri](gallery.md) | Referans ve imagehub görüntülerinin kaydırıcıyla yan yana karşılaştırması |
| [Sonuçlar](results.md) | Adım, boyut ve kaydedilen en büyük farkla checkpoint başına bir satır |
| [Tam doğrulama kaydı](../validation.md) | Yöntem, toleranslar, aygıt testleri, sınırlar ve bulunan hatalar |

## İddianın sınırları

!!! warning "Parity iddiasının belirli bir kapsamı vardır"
    - Belirtilen kitaplık sürümlerine karşı yalnızca **CPU'da float32** için geçerlidir.
    - MPS ve CUDA çalıştırmaları parity değil, duman testidir.
    - float16 veya bfloat16 için parity iddiası yoktur.
    - Bu bir kalite benchmark'ı değildir; FID veya CLIP skoru ölçülmemiştir.
    - Safety checker veya görünmez watermark yoktur; referans bunlar olmadan çalıştırılmıştır.

Tam liste [sınırlar](../validation.md#limits) bölümündedir.
