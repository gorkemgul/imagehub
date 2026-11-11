# Çeviriler

Dokümantasyonun kaynak dili İngilizcedir ve sitenin kökünde durur. Türkçe sürüm `/tr/` altında
yayımlanır. Türkçe çevirisi olmayan bir sayfa Türkçe sitede İngilizce gösterilir; böylece her
sayfaya iki dilden de ulaşılır.

## Kapsam

| Sayfalar | İngilizce | Türkçe |
| --- | --- | --- |
| Ana sayfa, Hızlı başlangıç, Rehberler, Eşdeğerlik özeti, Galeri, Sonuçlar, API, Katkıda bulunma, Çeviriler | kaynak | çevrildi |
| Model sayfaları (`docs/models.md` dosyasından üretilir) | kaynak | İngilizce gösterilir |
| Model kataloğu (`docs/catalog.md`) | kaynak | İngilizce gösterilir |
| Tam doğrulama kaydı (`docs/validation.md`) | kaynak | İngilizce gösterilir |
| Menü etiketleri | kaynak | çevrildi |

Model sayfaları, katalog ve doğrulama kaydı her yeni modelle değişen başvuru belgeleridir; bu
yüzden yalnızca İngilizce tutulur.

## Dosya adları

Site, `mkdocs-static-i18n` eklentisinin sonek düzenini kullanır. İngilizce dosyanın adını koruyun
ve `.md` uzantısından önce dil kodunu ekleyin:

```text
docs/
├── index.md
├── index.tr.md
└── guides/
    ├── training.md
    └── training.tr.md
```

Bir dil için ayrı bir klasör ağacı oluşturmayın.

## Kurallar

- Çevrilmiş bir sayfadaki değişiklik, aynı pull request'te iki dosyayı da günceller.
- Başlık seviyelerini, kod bloklarını, anchor'ları, HTML özniteliklerini ve göreli link hedeflerini
  değiştirmeyin.
- Düz metni, başlıkları, tablo metinlerini ve görsel açıklamalarını çevirin.
- Model id'lerini, Python adlarını, seçenekleri, komutları, dosya yollarını ve lisans adlarını
  çevirmeyin.
- Sayıları ve destek iddialarını İngilizce sayfaya ve `imagehub/catalog.py` dosyasına göre kontrol
  edin.
- Menüye eklenen yeni bir sayfanın etiketi `mkdocs.yml` içindeki `nav_translations` listesine de
  eklenir.

## Derleme

```bash
mkdocs build --strict
```

Derleme İngilizce siteyi köke, Türkçe siteyi `tr/` altına yazar. Pull request açmadan önce iki
siteyi ve dil değiştiriciyi kontrol edin.
