# imagehub: çekirdek altyapı ve Stable Diffusion 1.x tasarımı

- Tarih: 2026-10-04
- Durum: onaylandı, uygulama planı bekliyor
- Bağlam: [kickoff notları](../../research/2026-10-04-kickoff.md)

## 1. Amaç ve kapsam

İlk kilometre taşında imagehub'ın çekirdek altyapısı kurulur. Bu altyapının üzerinde Stable Diffusion 1.x
mimarisi yerel PyTorch ile yeniden yazılır ve beş checkpoint parity kanıtıyla uçtan uca çalıştırılır.
Kilometre taşı PyPI'da `imagehub 0.1.0` sürümüyle kapanır.

**Temel yaklaşım:** Düz bir paket düzeni izlenir: `models/`, `backends/`, `pipelines/`, elle yazılmış bir
katalog ve lazy bir kayıt defteri. Mimari başına ayrı spec dosyaları ve model başına `SOURCE.json`
kullanılmaz. Bu temel düzenden sapılan her nokta bu dokümanda **[sapma]** olarak işaretlenmiştir.

### Kapsamda

- Çekirdek: katalog, kayıt defteri, indirme, checkpoint yükleme, pipeline, görsel tipi, hatalar.
- SD 1.x mimarisi: CLIP BPE tokenizer, CLIP metin kodlayıcı, `UNet2DConditionModel` eşdeğeri U-Net,
  `AutoencoderKL` eşdeğeri VAE ve PNDM zamanlayıcı (`skip_prk_steps=True`, yalnızca PLMS adımları).
- Checkpoint'ler: `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4` (CompVis, yazarın resmi depoları) ve
  `sd-v1-5` (`stable-diffusion-v1-5/stable-diffusion-v1-5`, topluluk aynası).
- Görev: yalnızca `t2i`.
- CPU fp32 parity, MPS ve CUDA duman testleri, belgeler, PyPI sürümü.

### Kapsam dışında

img2img ve `edit` görevi, `understand` görevi, PNDM dışındaki zamanlayıcılar, safety checker ve
invisible watermark, fp16 parity, hız optimizasyonları, eğitim, LlamaGen (bir sonraki kilometre taşı).

## 2. Paket yapısı

```
imagehub/
├── pyproject.toml
├── LICENSE                     # Apache-2.0
├── THIRD_PARTY_NOTICES.md      # diffusers, transformers, CompVis atıf tablosu
├── licenses/                   # Diffusers-Apache-2.0.txt, Transformers-Apache-2.0.txt, OpenAI-CLIP-MIT.txt
├── src/imagehub/
│   ├── __init__.py             # dışa açık API, __version__
│   ├── types.py                # Task, T2IRequest, ImageResult
│   ├── image.py                # Image tipi, stdlib (zlib) ile PNG okuma ve yazma
│   ├── exceptions.py
│   ├── catalog.py              # ModelSpec, MODEL_CATALOG
│   ├── registry.py             # BackendRegistry
│   ├── factory.py              # pipeline()
│   ├── download.py             # HF API, commit sabitleme, sha256
│   ├── pretrained.py           # python -m imagehub.pretrained
│   ├── checkpoints.py          # stdlib safetensors okuyucu, NativeModel
│   ├── tokenization.py         # CLIPTokenizer (yerel BPE)
│   ├── models/
│   │   ├── clip_text.py
│   │   ├── unet_2d.py
│   │   └── autoencoder_kl.py
│   ├── schedulers/             # [sapma] diffusion aileleri paylaşacak
│   │   └── pndm.py
│   ├── backends/
│   │   ├── base.py
│   │   └── stable_diffusion.py
│   └── pipelines/
│       ├── base.py
│       └── text_to_image.py
├── tests/
└── docs/
    ├── catalog.md
    ├── models.md
    ├── validation.md
    └── validation.json
```

`pyproject.toml`:

- Derleme aracı `hatchling`. `requires-python = ">=3.11"`. `dependencies = ["torch>=2.2"]`.
- Extra'lar:
  - `image = ["pillow>=10"]`
  - `reference = ["diffusers==0.40.0", "transformers==5.18.0"]`. Sürümler sabittir; uygulama
    sırasında kurulum çalışmazsa bir alt sürüme inilir ve seçilen sürüm `validation.json`'a yazılır.
  - `dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.9", "mypy>=1.13", "build>=1.2"]`
- Araç ayarları: ruff `line-length = 100`, kurallar `E, F, I, UP, B`. mypy'de
  `disallow_untyped_defs = true`. pytest'te `reference` adlı bir marker tanımlıdır.
- Paket adı `imagehub`. PyPI'da boş (2026-10-04'te kontrol edildi).

Torch'un alt sınırı 2.2'dir. Geliştirme Python 3.13 ve torch 2.13 ile yapılır.
Python 3.11 ile torch 2.2 kombinasyonu sürümden önce bir kez test edilir.

## 3. Katalog ve kayıt defteri

### 3.1 `ModelSpec`

Katalog, `catalog.py` içinde elle yazılır ve sürüm kontrolünde tutulur. Bir model,
**yayımlanmış bir ağırlık setidir**. Aynı ailedeki modeller aynı kodu paylaşır.

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str                         # "sd-v1-5"
    task: Task                      # Task.T2I
    family: str                     # "stable-diffusion"
    source: str                     # "https://huggingface.co/<repo_id>"
    repo_id: str | None
    files: tuple[str, ...]          # indirilecek dosyaların açık listesi
    format: str                     # "diffusers"
    config: Mapping[str, Any]
    gated: bool
    license: str                    # "CreativeML-OpenRAIL-M"
    catalog_id: str                 # [sapma] awesome kataloğundaki id: "stable-diffusion-1"
    revision: str                   # [sapma] doğrulanmış 40 karakterlik commit
    provenance: str                 # [sapma] "author" | "community-mirror"
```

| id | repo_id | revision | provenance |
|---|---|---|---|
| `sd-v1-1` | `CompVis/stable-diffusion-v1-1` | `574555a657f0c8d6e970a02c7e095e9d4220dcbf` | author |
| `sd-v1-2` | `CompVis/stable-diffusion-v1-2` | `f519eebf1372598b85ab317ec273a63ebeddd5eb` | author |
| `sd-v1-3` | `CompVis/stable-diffusion-v1-3` | `159f26fccae6e3ea26c797ad24c5e7033da5b1ee` | author |
| `sd-v1-4` | `CompVis/stable-diffusion-v1-4` | `133a221b8aa7292a167afc5127cb63fb5005638b` | author |
| `sd-v1-5` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | community-mirror |

### 3.2 `files` seçim kuralı

Her checkpoint için şu dosyalar indirilir:

- `model_index.json`
- `scheduler/scheduler_config.json`
- `tokenizer/vocab.json`, `tokenizer/merges.txt`, `tokenizer/tokenizer_config.json`,
  `tokenizer/special_tokens_map.json`
- `text_encoder/config.json`, `unet/config.json`, `vae/config.json`
- Her bileşenin fp32 ağırlığı:
  - **v1.4 ve v1.5:** `text_encoder/model.safetensors`, `unet/diffusion_pytorch_model.safetensors`,
    `vae/diffusion_pytorch_model.safetensors`
  - **v1.1, v1.2 ve v1.3:** Bu depolarda fp32 safetensors yok. Yerine
    `text_encoder/pytorch_model.bin`, `unet/diffusion_pytorch_model.bin` ve
    `vae/diffusion_pytorch_model.bin` indirilir ve `torch.load(..., weights_only=True)` ile yüklenir
    (tensor-only `.bin` yolu).

**Hiçbir zaman indirilmeyenler:** fp16 dosyaları, `non_ema` dosyaları, tek dosyalık `*.ckpt` ve
`v1-5-pruned*` checkpoint'leri, `safety_checker/`, `feature_extractor/`, `.py` dosyaları, görseller.

Yerel modüller, diffusers ve transformers'ın state-dict anahtar adlarını aynen kullanır. Bu yüzden
`format="diffusers"` için ayrı bir dönüştürme adımı gerekmez.

### 3.3 Kayıt defteri

`BackendRegistry`:

- Anahtar `(Task, isim)` çiftidir, değer bir callable ya da `"module:Class"` string'idir.
- Kayıt sırasında modül import edilmez. Import, `create()` çağrısında yapılır.
- Aile kaydı:
  `(Task.T2I, "stable-diffusion", "imagehub.backends.stable_diffusion:StableDiffusionBackend")`.
- `MODEL_CATALOG`'daki her model id'si, ailesinin factory'siyle otomatik olarak kaydedilir.
  `backend="sd-v1-5"` ile `backend="stable-diffusion"` aynı sınıfı seçer.
- Katalog id'si dosyaların yerini tutmaz. Kullanılacak ağırlıkları `model=` ile verilen yerel klasör
  belirler.
- Dışa açık fonksiyonlar: `available_models()`, `model_info()`, `available_backends()`.

### 3.4 Awesome kataloğuyla bağ [sapma]

`tests/test_catalog_link.py` şu yolu arar:
`../awesome-text-to-image-architectures/data/models.json`.

- Dosya varsa test her `catalog_id`'nin orada bulunduğunu ve `license.weights` değerinin
  `ModelSpec.license` ile tutarlı olduğunu kontrol eder. Karşılaştırma büyük/küçük harf ve ayraç
  farklarını yok sayar.
- Dosya yoksa test atlanır.
- Çalışma anında `models.json`'a bağımlılık yoktur.

Katalog reposundaki `stable-diffusion-1` kaydına `sd-v1-5` varyantının ayna notuyla eklenmesi ayrı bir
iştir, bu kilometre taşının parçası değildir.

## 4. Checkpoint indirme ve doğrulama

```bash
python -m imagehub.pretrained sd-v1-5 ./sd15 [--revision <ref>]
```

Akış:

1. `model_info(id)` ile spec okunur. Hedef klasör boş değilse `FileExistsError` verilir.
2. Şu adres çağrılır: `GET https://huggingface.co/api/models/<repo>/revision/<rev>?blobs=true`.
   Dönen `sha` 40 karakterlik bir hex değilse hata verilir. `files` listesindeki her dosya
   `siblings` içinde aranır. Eksik dosya varsa indirme başlamadan hata verilir.
3. Her dosya `https://huggingface.co/<repo>/resolve/<commit>/<file>` adresinden, hedef klasördeki
   geçici bir dosyaya 1 MB'lık parçalar hâlinde indirilir. İnerken sha256 hesaplanır.
   - LFS sha256 değeri ya da `size` uyuşmazsa geçici dosya silinir ve hata verilir.
   - Uyuşursa dosya `os.replace` ile yerine taşınır.
4. `imagehub_snapshot.json` yazılır. İçeriği:
   `{model_id, repo_id, revision, verified_revision: bool, files: {path: sha256}}`.
   - **[sapma]** Varsayılan revision, `ModelSpec.revision` değeridir (`main` değil).
   - `--revision` ile başka bir ref verilirse ve çözülen commit `ModelSpec.revision`'dan farklıysa
     `verified_revision: false` yazılır ve bir uyarı basılır.

Güvenlik kuralları:

- Yalnızca HTTPS kullanılır.
- Başka bir host'a yönlendirmede `Authorization` başlığı silinir. `HF_TOKEN` yalnızca
  huggingface.co'ya gönderilir.
- Mutlak yollar ve `..` içeren dosya yolları reddedilir.
- Python kodu indirilmez. Pickle yalnızca `weights_only=True` ile açılır.
- `pipeline()` ve backend'ler hiçbir zaman ağa çıkmaz.

Yükleme (`checkpoints.py`):

- `load_safetensors` safetensors başlığını, offset'leri, örtüşmeleri ve sondaki fazla veriyi
  doğrular. Yalnızca stdlib ve `torch.frombuffer` kullanır.
- `load_weights` safetensors, tensor-only `.bin` ve `.pt` dosyalarını ve sharded index'leri okur.
- `NativeModel.from_pretrained` modeli meta-device üzerinde kurar. Ardından
  `load_state_dict(strict=True, assign=True)` çağırır. Eksik ya da fazla anahtar hata sebebidir.
  `prepare_state_dict` yalnızca belgelenmiş anahtar dönüşümleri için override edilir.

Bilinçli olarak yapılmayanlar: kaldığı yerden devam eden indirme ve her yüklemede yeniden hash
hesaplama.

## 5. Pipeline API'si

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="./sd15", device="mps") as pipe:
    result = pipe("bir kedi, yağlı boya", seed=42)
    result.images[0].save("kedi.png")
```

- `pipeline(task, *, backend, options=None, registry=None, **backend_options)`.
  `backend_options` modelin kurulumunu belirler (`model`, `device`, `dtype`).
  `options` ise istek varsayılanlarını belirler.
- **Pipeline** (`pipelines/base.py`): `preprocess` girdiyi doğrular, `forward`
  backend'i çağırır, `postprocess` sonucun tipini kontrol eder. Ayrıca `run`, `__call__`, `batch`,
  `map`, `arun`, `load`, `close` ve context manager desteği vardır. `stream` desteklenmez.
- **Backend** (`backends/base.py`): Model ilk istekte yüklenir. `RLock` ile her
  instance'ta tek inference çalışır. Yükleme başarısız olursa temizlenir ve yeniden denenebilir.
  `unload` belleği boşaltır.

İstek ayarları (varsayılanlar diffusers `StableDiffusionPipeline` tarifiyle aynıdır):

| Ayar | Varsayılan | Kural |
|---|---|---|
| `prompt` | — | Boş olmayan bir string |
| `negative_prompt` | `None` | |
| `steps` | 50 | Pozitif bir int |
| `guidance` | 7.5 | Sonlu ve ≥ 0. 1'den büyükse classifier-free guidance uygulanır |
| `width`, `height` | 512 | 8'in katı ve pozitif |
| `num_images` | 1 | Pozitif bir int |
| `seed` | `None` | `None` ise rastgele bir seed seçilir ve metadata'ya yazılır |
| `latents` | `None` | `(num_images, 4, H/8, W/8)` şeklinde bir tensör. Verilirse seed kullanılmaz |

- Tanınmayan bir ayar `TypeError` verir.
- **Seed uyumu:** Başlangıç latent'leri diffusers'taki gibi üretilir:
  `torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=float32)`, ardından
  cihaza taşınır ve zamanlayıcının `init_noise_sigma` değeriyle ölçeklenir. Böylece CPU'da aynı seed,
  diffusers'ta `generator=torch.Generator("cpu").manual_seed(seed)` ile alınan görselin aynısını verir.
- **Cihaz kuralları:** Model kodunda cihaza özel bir dal yoktur. Gürültü CPU'da üretilir.
  fp64 kullanılmaz (MPS desteklemez). Zamanlayıcı hesapları diffusers'taki dtype'larla yapılır.

Tipler:

- `T2IRequest(prompt, parameters)`
- `ImageResult(images: tuple[Image, ...], metadata)`. Metadata alanları: `model_id`, `revision`,
  `seed`, `steps`, `guidance`, `width`, `height`, `scheduler`.
- `Image`. `.tensor` `(H, W, 3)` şeklinde bir uint8 tensördür. Yöntemler: `.save(path)` (stdlib PNG),
  `.to_pil()` (`[image]` extra'sı gerektirir), `Image.open(path)` (stdlib PNG okuyucu, 8 bit RGB ve
  RGBA).
- Metadata'daki `revision` değeri, model klasöründeki `imagehub_snapshot.json`'dan okunur. Dosya
  yoksa `None` olur.

**Safety checker yok.** diffusers'taki NSFW filtresi ayrı bir sınıflandırıcıdır, üretim mimarisinin
parçası değildir. Bu durum `validation.md`'de ve model sayfasında açıkça yazılır. Ağırlıkların
CreativeML OpenRAIL-M kullanım kısıtları kullanıcıyı bağlar ve bu bilgi `docs/models.md`'de yer alır.

## 6. Parity testi altyapısı

### 6.1 Normal testler (ağ yok, her `pytest` çalıştırmasında)

- Katalog: id'ler benzersizdir, revision'lar 40 karakterdir, her ailenin bir backend kaydı vardır.
- Kayıt defteri: lazy import çalışır, bilinmeyen isim hata verir, model id'leri seçici olarak kayıtlıdır.
- `checkpoints`: bozuk safetensors (başlık, offset, örtüşme, sondaki fazla veri) reddedilir. Strict
  yüklemede eksik veya fazla anahtar hata verir. `.bin` yolu `weights_only` kullanır.
- `download`: sahte bir HTTP sunucusuyla test edilir. Kontrol edilenler: sha256 ve boyut uyuşmazlığı,
  eksik dosya, HTTPS dışı yönlendirme, başka host'a yönlendirmede token'ın silinmesi, boş olmayan
  hedef klasör, `verified_revision`.
- `image`: PNG yazıp geri okuma (round-trip), Pillow ile çapraz kontrol (Pillow kuruluysa).
- Pipeline: girdi doğrulama, tanınmayan ayar, seed'in metadata'ya yazılması.
- Rastgele ağırlıklı küçük bir SD konfigürasyonu uçtan uca çalışır ve doğru şekilde sonlu bir görsel
  üretir. `save_pretrained` ile kaydedilen model `from_pretrained` ile geri yüklenir.
- `test_catalog_link.py` (bkz. 3.4).

### 6.2 Referans testleri (`@pytest.mark.reference`, isteğe bağlı)

```bash
IMAGEHUB_REFERENCE_TESTS=1 IMAGEHUB_SD_V1_4=./sd14 IMAGEHUB_SD_V1_5=./sd15 pytest -m reference
```

- `IMAGEHUB_REFERENCE_TESTS=1` yoksa testler atlanır. Referans paketler `pytest.importorskip` ile
  yüklenir.
- Checkpoint yolları `IMAGEHUB_SD_V1_1` … `IMAGEHUB_SD_V1_5` ortam değişkenleriyle verilir. Bir
  değişken yoksa yalnızca o checkpoint'in testleri atlanır.
- Testler hiçbir şey indirmez. Referans modeller aynı yerel klasörden şu şekilde açılır:
  `diffusers.StableDiffusionPipeline.from_pretrained(dir, safety_checker=None, local_files_only=True)`.
- Bütün testler CPU'da ve fp32'de çalışır. `torch.set_num_threads` sabitlenir ve kayda geçer.

| Karşılaştırma | Checkpoint | Tolerans |
|---|---|---|
| Tokenizer token id'leri. Prompt'lar ASCII, Türkçe, emoji, boşluk ve noktalama içerir, ayrıca 77 token'dan uzun olanlar vardır | v1.4 | `==` (birebir) |
| PNDM `timesteps` | v1.4 | `==` (birebir) |
| PNDM `step` çıktıları, rastgele girdilerle bir dizi adım | v1.4 | ölçülür |
| CLIP metin kodlayıcı `last_hidden_state` | v1.4 | ölçülür |
| U-Net tek forward (latent, t, metin gömmesi) | v1.4 | ölçülür |
| VAE decode | v1.4 | ölçülür |
| Seed'den başlangıç latent'i | — | `==` (birebir) |
| Uçtan uca: sabit `latents`, 50 adım, guidance 7.5, sabit prompt. Son latent ve uint8 görsel | v1.1–v1.5 (her biri) | ölçülür |
| Uçtan uca: aynı seed ile görsel | v1.5 | ölçülür |

**Tolerans kuralı:** İlk çalıştırmada referansla aradaki en büyük mutlak ve göreli fark ölçülür.
Test toleransı bu değerin biraz üstüne konur, tipik olarak bir sonraki "1e-k" ya da "2e-k" basamağı.
Ölçülen değer `validation.json`'a yazılır. Kesikli çıktılarda (token id'leri, timesteps) tolerans
her zaman 0'dır.

### 6.3 Doğrulama seviyeleri

Üç doğrulama seviyesi vardır:

- `full`: Tam ağırlıklar strict yüklenir ve referansla parity sağlanır.
- `header_only`: Yayımlanmış bütün tensör adları ve şekilleri, meta-device üzerinde yerel mimariyle
  eşleşir. Üretim test edilmez.
- `experimental`: Mimari ve birim testleri vardır, özgün ağırlıklarla test yapılmamıştır (örneğin
  erişim onayı gerektiren checkpoint'ler).

Seviye `validation.json`'da ve `docs/catalog.md`'de her model için yazılır. Yalnızca `full`
seviyesindeki modeller parity iddiası taşır. Bu kilometre taşındaki beş checkpoint'in hepsi `full`
hedefler.

### 6.4 Cihaz duman testleri

- **MPS:** Testin işaretleri `reference` ve `mps`'tir; MPS yoksa test atlanır. v1.5 ile 512×512
  çözünürlükte, 20 adımlık bir üretim yapılır. Sonuç sonlu olmalı ve şekli doğru olmalıdır. CPU
  çıktısına olan en büyük fark ve PSNR yalnızca bilgi olarak kaydedilir, eşik değildir.
- **CUDA:** Aynı test `cuda` işaretiyle çalışır; CUDA yoksa atlanır. Her PyPI sürümünden önce
  RunPod'da (RTX 4090) çalıştırılır ve sonucu `validation.json`'a yazılır.

### 6.5 Kayıt: `docs/validation.json` ve `validation.md`

Kayda şunlar girer:

- Tarih ve ortam: Python, torch, diffusers ve transformers sürümleri, makine, CPU thread sayısı.
- Her checkpoint için `repo_id`, `revision`, dosya sha256'ları ve `status`.
- Her karşılaştırmada ölçülen en büyük fark ve kullanılan tolerans.
- Cihaz duman testlerinin sonuçları.
- `limits` listesi. Örnekler: "MPS ve CUDA için parity iddiası yok", "fp16 parity yok",
  "safety checker yok", "kalite benchmark'ı (FID, CLIP score) yok", "PNDM dışında zamanlayıcı yok".
- Test sayıları.

## 7. Hata yönetimi

`exceptions.py`:

- `ImageHubError(Exception)`
- `MissingDependencyError(ImageHubError, ImportError)`
- `BackendBusyError(ImageHubError, RuntimeError)`
- `BackendContractError(ImageHubError, TypeError)`
- `CheckpointError(ImageHubError, ValueError)`: bozuk dosya, eksik veya fazla anahtar, eksik dosya.
- `DownloadError(ImageHubError, RuntimeError)`: checksum, boyut, eksik artefakt, güvensiz URL.

| Durum | Davranış |
|---|---|
| Bilinmeyen model ya da backend | `ValueError`. Mevcut isimler listelenir |
| Model klasörü yok ya da dosyası eksik | `CheckpointError`. Eksik dosya adı ve `python -m imagehub.pretrained <id> <dir>` önerisi verilir |
| Eksik veya fazla state-dict anahtarı | `CheckpointError`. İki listenin ilk 20 elemanı gösterilir |
| İndirmede checksum veya boyut uyuşmazlığı | Geçici dosya silinir, `DownloadError` verilir |
| Geçersiz ayar değeri | Model yüklenmeden `ValueError` verilir |
| Tanınmayan ayar | `TypeError` |
| Pillow yokken `.to_pil()` | `MissingDependencyError`, `pip install imagehub[image]` önerilir |
| İstenen cihaz yok | `ValueError`, kullanılabilir cihazlar listelenir |

Hiçbir hata yutulmaz. Yükleme sırasında oluşan hata, temizlik başarısız olsa bile özgün tipiyle
yükselir.

## 8. Teslimatlar ve kabul ölçütleri

1. Bölüm 2–7'deki çekirdek modüller ve SD 1.x mimarisi.
2. Normal testlerin tamamı Python 3.13 / torch 2.13 ve Python 3.11 / torch 2.2 ortamlarında geçer.
   ruff, biçim kontrolü ve mypy temizdir.
3. Beş checkpoint'in hepsi `full` seviyesindedir. Referans testleri CPU fp32'de geçer, ölçülen farklar
   `validation.json`'da kayıtlıdır.
4. MPS duman testi Mac'te, CUDA duman testi RunPod'da geçer.
5. Belgeler: `README.md` (kurulum, indirme, örnek), `docs/catalog.md`, `docs/models.md` (SD 1.x
   sayfası, lisans ve kısıtlar), `docs/validation.md` ve `.json`, `THIRD_PARTY_NOTICES.md`,
   `licenses/`.
6. `imagehub 0.1.0` PyPI'da yayımlanır. Wheel temiz bir ortamda kurulur ve indirme yapılmadan
   import edilip normal testler çalıştırılır.

## 9. Uygulama notları

- Kod yazma görevleri Sonnet modelli alt ajanlara verilir. Tasarım ve inceleme ana oturumda kalır.
- Önerilen uygulama sırası:
  1. Çekirdek iskelet (tipler, hatalar, görsel, katalog, kayıt defteri, checkpoints, download).
  2. Tokenizer.
  3. Metin kodlayıcı.
  4. VAE.
  5. U-Net.
  6. PNDM zamanlayıcı.
  7. Backend ve pipeline.
  8. Referans testleri.
  9. Belgeler ve sürüm.
- Her mimari bileşen, referans testi geçmeden bir sonraki bileşene geçilmez.
