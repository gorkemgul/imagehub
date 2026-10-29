# Ağırlıkları indirme

imagehub kendiliğinden hiçbir şey indirmez. `pipeline()` yalnızca bir nesne oluşturur ve yükleme
yerel dosyaları okur. Ağırlıkları indirmek, bilinçli olarak çalıştırdığınız ayrı bir komuttur.

```bash
python -m imagehub.pretrained <model-id> <empty-directory>
python -m imagehub.pretrained flux1-schnell checkpoints/flux1-schnell
```

Python'dan kullanımı:

```python
from imagehub.pretrained import download_pretrained

download_pretrained("sd-v1-5", "checkpoints/sd-v1-5")
```

## İndiricinin garantileri

- **Revizyon**, parity testlerini geçen commit'e sabitlenir ([kataloğa](../catalog.md) bakın).
  Başka revizyon istemek mümkündür ancak doğrulanmamış olarak kaydedilir.
- Her dosyanın boyutu denetlenir. Ağırlık dosyaları ayrıca Hub'ın yayımladığı sha256 özetiyle,
  diğer dosyalar git blob hash'iyle doğrulanır.
- Hedef boş olmalıdır; `imagehub_snapshot.json` manifestosu tam olarak nelerin indirildiğini kaydeder.
- Yalnızca HTTPS kabul edilir ve başka host'a yönlendirmelerde kimlik bilgileri kaldırılır.

## Kapılı depolar

Bazı depolarda önce şartları kabul etmeniz gerekir. Hugging Face'teki model sayfasında şartları
kabul edin, bir okuma token'ı oluşturun ve ortam üzerinden iletin:

```bash
export HF_TOKEN=...
python -m imagehub.pretrained krea2-turbo checkpoints/krea2-turbo
```

## Disk alanı

Checkpoint'ler büyüktür; çoğu birkaç gigabayt yer kaplar. Yalnızca kullandığınız modelleri tutun.
Snapshot sıradan bir klasördür; alan açmak için klasörü silmeniz yeterlidir.

## Lisanslar

imagehub Apache-2.0 lisanslıdır ancak **model ağırlıkları değildir**. Her modelin kendi lisansı
vardır ve bunlar [katalogda](../catalog.md) ve model sayfalarında listelenir. Bazıları ticari kullanıma
kapalıdır, bazılarıysa bölgeleri veya kullanım biçimlerini kısıtlar. Lisans, ağırlıkları indirip
kullanan kişiyi bağlar.
