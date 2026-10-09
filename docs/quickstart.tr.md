---
title: Hızlı başlangıç
---

# Hızlı başlangıç

Sıfırdan bir görüntüye dört adımda. Python 3.11 veya yenisi gerekir; tek zorunlu paket PyTorch'tur.

## 1. Kurulum

=== "pip"

    ```bash
    pip install imagehub            # yalnızca PyTorch
    pip install "imagehub[image]"   # Pillow ekler (Image.to_pil() için)
    ```

=== "Kaynaktan"

    ```bash
    git clone https://github.com/gorkemgul/imagehub
    cd imagehub
    pip install -e ".[image]"
    ```

## 2. Modeli indirin

İndirme yalnızca bu komutu çalıştırdığınızda olur. Doğrulanmış, sabitlenmiş revizyonu getirir ve
her dosyayı denetler.

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

!!! note "Kapılı ve büyük modeller"
    Bazı depolar (örneğin `krea2-turbo`) kapılıdır: Hugging Face'te şartları kabul edin ve
    `HF_TOKEN` değişkenini ayarlayın. Ağırlık lisansları modele göre değişir; ticari kullanmadan önce
    [kataloğa](catalog.md) bakın.

## 3. Üretin

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
    result = pipe("a photograph of an astronaut riding a horse", seed=42)

result.images[0].save("astronaut.png")
print(result.metadata)   # tohum, adım, rehberlik, boyut, revizyon, aygıt, dtype
```

`with` bloğu girişte ağırlıkları yükler, çıkışta bırakır. `pipe(...)` bloğun dışında da çalışır; model
ilk kullanımda yüklenir.

## 4. Tek şeyi değiştirin

=== "Başka model"

    ```python
    with pipeline("t2i", backend="flux1-schnell", model="checkpoints/flux1-schnell") as pipe:
        result = pipe("a lighthouse at dusk", seed=7)
    ```

=== "Başka aygıt"

    ```python
    pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps")
    ```

=== "Üretim ayarları"

    ```python
    pipe("a cat, oil painting", seed=42, steps=30, guidance=6.0, negative_prompt="blurry")
    ```

Ayarlar mimariye göre biraz değişir (örneğin damıtılmış bir modelde rehberlik yoktur). Her
[model sayfası](models/index.md) kendi ayarlarını listeler; bilinmeyen ayarlar yok sayılmaz, hata verir.

## Sırada ne var

<div class="ih-features" markdown>
<div markdown>
### Rehberler
[Aygıtlar ve hassasiyet](guides/devices.md), [görüntü girdisi](guides/image-input.md) ve
[tekrarlanabilirlik](guides/reproducibility.md).
</div>
<div markdown>
### Modeller
Ayarları, lisansları ve notlarıyla [{{ ih.models }} checkpoint](models/index.md).
</div>
<div markdown>
### Eşdeğerlik
[Neyin ölçüldüğü](parity/index.md) ve bakmak için [galeri](parity/gallery.md).
</div>
</div>
