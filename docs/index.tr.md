---
title: imagehub
hide:
  - navigation
  - toc
---

# imagehub

<div class="ih-hero" markdown>

<!-- ih:banner -->

<p class="ih-eyebrow">Açık ağırlıklı metinden görüntüye modeller · saf PyTorch</p>

## Çok model. Tek API. Aynı pikseller.

<p class="ih-lead">
imagehub, açık metinden-görüntüye mimarilerini PyTorch ile baştan yazar ve her modeli referans
uygulamasıyla karşılaştırır. CPU'da float32 ile çıktı, diffusers'ın çıktısıyla bit düzeyinde
aynıdır.
</p>

<div class="ih-cta" markdown>
[Başla](quickstart.md){ .md-button .md-button--primary }
[Modellere göz at](models/index.md){ .md-button }
[Eşdeğerlik kaydı](parity/index.md){ .md-button }
</div>

</div>

<div class="ih-stats" markdown>
<div class="ih-stat" markdown><strong>{{ ih.models }}</strong><span>yayımlanmış checkpoint</span></div>
<div class="ih-stat" markdown><strong>{{ ih.families }}</strong><span>mimari ailesi</span></div>
<div class="ih-stat" markdown><strong>{{ ih.validated }}</strong><span>kayıtlı eşdeğerlik kanıtı olan</span></div>
<div class="ih-stat" markdown><strong>1</strong><span>çalışma bağımlılığı: PyTorch</span></div>
</div>

<div class="ih-split" markdown>
<div markdown>

## Her model için tek çağrı

Model kimliği mimariyi, klasör ağırlıkları seçer. Stable Diffusion'dan FLUX'a ya da
Qwen-Image'e geçmek tek bir metni değiştirmek demektir.

- Nesneyi oluşturmak ağa dokunmaz; indirme ayrı ve açık bir adımdır.
- Ayarlar, hiçbir ağırlık yüklenmeden önce doğrulanır; bilinmeyen ayar hata verir.
- Aynı tohum, referans uygulamayla aynı görüntüyü üretir.

</div>
<div markdown>

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
    result = pipe("a photograph of an astronaut riding a horse", seed=42)

result.images[0].save("astronaut.png")
print(result.metadata)
```

</div>
</div>

## Neden imagehub

<div class="ih-features" markdown>
<div markdown>
### Yaklaşıklık değil, eşdeğerlik
Bir model ancak tüm tensörler sıkı biçimde yüklendiğinde ve aynı gürültüden tam bir üretim
referansla eşleştiğinde kabul edilir. Farklar iddia edilmez, kaydedilir.
</div>
<div markdown>
### Okunabilir mimariler
Her model bu depoda düz PyTorch olarak durur; çıkarım sırasında diffusers veya transformers'a
gizli bir bağımlılık yoktur.
</div>
<div markdown>
### Sabitlenmiş, doğrulanmış ağırlıklar
Katalog kayıtları bir commit'i ve dosya listesini sabitler. İndirme boyutları ve özetleri denetler,
ne aldığınızı bir manifestoya yazar.
</div>
<div markdown>
### Dürüst sınırlar
Eşdeğerlik CPU ve float32 için iddia edilir. Diğer aygıtlar ve hassasiyetler duman testiyle
denenir ve öyle etiketlenir.
</div>
</div>

## Modeller

<!-- ih:model-grid -->

## Aynı tohum, aynı görüntü

Kaydırıcıyı sürükleyin: sol taraf referans uygulama, sağ taraf imagehub.

<div class="ih-gallery" markdown>
<!-- ih:compare:pixart-sigma-xl-1024 -->
<!-- ih:compare:kolors -->
<!-- ih:compare:flux1-krea-dev -->
</div>

[Tüm galeri](parity/gallery.md){ .md-button }
[Eşdeğerlik nasıl ölçülür](parity/index.md){ .md-button }
