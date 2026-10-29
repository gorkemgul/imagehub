---
title: Katkıda bulunma
---

# Katkıda bulunma

En faydalı katkı yeni bir modeldir. Ölçüt, mevcut tüm modellerle aynıdır: model referans uygulamayı
yeniden üretmeli ve kanıtı depoda bulunmalıdır.

## Kurulum

```bash
uv venv -p 3.13 .venv && uv pip install --python .venv -e ".[dev,image,docs]"
.venv/bin/python scripts/fetch_fixtures.py
.venv/bin/python -m pytest
.venv/bin/mkdocs serve        # preview this site
```

Kabul ettiğiniz depo lisansları kapılı fixture'lara erişim veriyorsa `HF_TOKEN` ayarlayın. Kısıtlı
dosyalar `tests/fixtures/RESTRICTED.json` ile doğrulanır ve yok sayılan
`tests/fixtures/_restricted` cache'inde kalır.

## Model ekleme

1. Modeli, deposunu ve ağırlık lisansını belirten **bir issue açın**. Her model için ayrı issue,
   incelemenin odaklı kalmasını sağlar.
2. Önce **gerçek config dosyalarını indirin** ve test fixture'ı olarak saklayın
   (`tests/fixtures/<model>_configs/`). Bileşenleri varsayımlara değil, gerçekten yayımlanmış
   config'lere göre yazın.
3. Bileşenleri `imagehub/models/` altında; scheduler'ları `imagehub/schedulers/`, tokenizer'ları
   `imagehub/tokenization_*.py` altında saf PyTorch ile **uygulayın**. Bir bileşeni mevcut aileyle
   ancak kod gerçekten aynıysa paylaşın.
4. `imagehub/backends/` altında **backend ekleyin**: seçenekleri yüklemeden önce doğrulayan, tembel
   yüklenen ve metadata'sı çalıştırmayı kaydeden `ImageResult` döndüren bir `TextToImageBackend`.
   Bunu `imagehub/registry.py` içinde kaydedin.
5. `imagehub/catalog.py` içine **katalog kaydı ekleyin**: model kimliği, sabitlenmiş 40 karakterlik
   revizyon, dosya listesi, lisans, provenance ve
   [awesome-text-to-image-architectures](https://github.com/gorkemgul/awesome-text-to-image-architectures)
   bağlantısını kuran `catalog_id`.
6. `tests/reference/` altındaki referans testleriyle **parity'yi kanıtlayın**: sıkı yükleme, her
   bileşen, ardından CPU'da float32 ve aynı gürültüyle uçtan uca üretim. Bu testler yalnızca
   `IMAGEHUB_REFERENCE_TESTS=1` ayarlandığında çalışır ve hiçbir şey indirmez.
7. Sonucu `docs/validation.json` içine (`scripts/build_validation.py` ile) ve
   `docs/validation.md` dosyasına **kaydedin**; modeli `docs/models.md`, `docs/catalog.md` ve README
   tablosuna ekleyin.
8. `main` dalına karşı, issue'yu kapatan ve yalnızca bir model içeren **pull request açın**.

??? tip "Bir models.md bölümünde bulunması gerekenler"
    Mimari adını taşıyan 1. düzey başlığın ardından
    `Catalog: [...] · Models: \`id\`, ... · Task: \`t2i\`` satırı; sonra **Architecture**,
    **Usage**, **Settings**, **Seed compatibility**, **Out of scope** ve **License** paragrafları
    gelmelidir. Site aile sayfasını, model grid'ini ve navigasyonu bu dosya ile katalogdan üretir;
    başka bir yeri düzenlemek gerekmez.

## Dokümantasyon sitesinin veri kaynakları

Bu sitede hiçbir bilgi elle tekrarlanmaz.

| Sayfa | Kaynak |
| --- | --- |
| Ana sayfadaki sayılar ve model grid'i | `imagehub/catalog.py`, `docs/validation.json` |
| Aile sayfaları | katalog ailesine göre bölünen `docs/models.md` |
| Sonuçlar | `docs/validation.json` |
| Galeri | `docs/assets/gallery/*.json` ve yanlarındaki görüntü çiftleri |
| Katalog, Doğrulama kaydı | `docs/catalog.md`, `docs/validation.md` |

Üretici `scripts/docs_hooks.py` dosyasıdır; `mkdocs serve` ve `mkdocs build` sırasında çalışır.
`mkdocs build --strict` başarılı olmalıdır.

### Galeri çifti ekleme

`<model-id>.reference.webp`, `<model-id>.imagehub.webp` ve `<model-id>.json` (ad, aile, ayarlar,
`max_pixel_diff`) dosyalarını `docs/assets/gallery/` içine koyun. Çift hem galeri sayfasında hem
modelin kendi sayfasında görünür.

### Çeviri

İngilizce sayfalar kaynaktır. Çeviri, uzantıdan önce locale eklenmiş bir kardeş dosyadır; örneğin
`quickstart.md` yanındaki `quickstart.tr.md`. Çevirisi olmayan sayfalar İngilizceye geri döner;
böylece bir dil sayfa sayfa büyüyebilir.

## Stil

Kod ruff ile biçimlendirilir (satır uzunluğu 100) ve mypy ile type-check edilir. Dokümantasyon açık
ve somut olmalıdır: neyin, hangi ortamda ölçüldüğünü ve neyin iddia edilmediğini belirtin.
