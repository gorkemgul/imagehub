# Tekrarlanabilirlik

## Aynı seed, aynı görüntü

`seed` iletirseniz başlangıç gürültüsü, modeli hangi aygıt çalıştırırsa çalıştırsın, referanstaki
gibi **CPU** üzerinde `torch.Generator("cpu").manual_seed(seed)` ile üretilir (örnek Stable
Diffusion içindir; mimari farkları ilgili [model sayfasında](../models/index.md) belirtilir).

```python
a = pipe("a cat", seed=1).images[0]
b = pipe("a cat", seed=1).images[0]
assert a.tensor.equal(b.tensor)
```

`seed` vermezseniz rastgele bir değer seçilip `result.metadata` içine kaydedilir; böylece görüntü
sonradan yeniden üretilebilir. `seed` veya kendi gürültü tensörünüz olan `latents` değerinden
yalnızca birini iletin.

## Referansla aynı görüntü

imagehub, kayıtlı checkpoint'ler için CPU'da float32 kullanarak her modelin kayıtlı referans
uygulamasıyla **aynı pikselleri** üretir. Doğrulama kaydı kullanılan diffusers, transformers veya
resmî kod revizyonunu tam olarak belirtir. İddianın kanıtları şunlardır:

- [Sonuçlar](../parity/results.md): checkpoint başına bir satırla kayıtlı farklar.
- [Galeri](../parity/gallery.md): referans ve imagehub görüntülerinin yan yana karşılaştırması.
- [Doğrulama kaydı](../validation.md): yöntem, toleranslar ve süreçte bulunan hatalar.

## Yine de neler farklı olabilir

- Farklı aygıt veya hassasiyet: [Aygıtlar ve hassasiyet](devices.md) sayfasına bakın.
- Referans kitaplığının farklı bir sürümü: kayıt, ölçümde kullanılan sürümleri belirtir.
- Farklı ağırlıklar: katalog bir revizyonu sabitler; başka revizyonlar kapsam dışındadır.

!!! info "Hizalama önemlidir"
    Safetensors dosyaları memory-map ile açılır; böylece ağırlıklar yayımlandıkları hizalamayla
    okunur. CPU BLAS, farklı hizalanmış verileri farklı sırada toplayabilir. Bu, doğrulama sırasında
    bulunan gerçek uyuşmazlık nedenlerinden biriydi.
