---
title: API referansı
---

# API referansı

Herkese açık her öğe üst düzey `imagehub` paketinden import edilebilir.

```python
import imagehub
imagehub.__version__   # "{{ ih.version }}"
```

## `pipeline`

```python
pipeline(task, *, backend, options=None, registry=None, **backend_options) -> Pipeline
```

Bir pipeline oluşturur. Nesneyi oluşturmak ağa erişmez ve ağırlıkları yüklemez.

| Argüman | Anlamı |
| --- | --- |
| `task` | `"t2i"` (metinden görüntüye) |
| `backend` | `"sd-v1-5"` gibi katalog model kimliği, `"flux"` gibi aile adı veya mevcut bir `Backend` |
| `options` | her isteğe uygulanan varsayılan üretim ayarları |
| `model` | *(backend seçeneği)* indirilmiş snapshot'ın bulunduğu klasör |
| `device` | *(backend seçeneği)* `"cpu"` (varsayılan), `"mps"`, `"cuda"` |
| `dtype` | *(backend seçeneği)* `torch.float32` (varsayılan), `torch.float16`, `torch.bfloat16` |
| `lora`, `lora_scale`, `lora_fuse` | *(backend seçeneği, LoRA adaptörleri bölümünde listelenen backend'ler)* LoRA adaptör dosyaları ([ayrıntılar](models/index.md#lora-adapters)) |

## `Pipeline`

| Üye | Açıklama |
| --- | --- |
| `pipe(prompt, **settings)` | bir istek çalıştırıp `ImageResult` döndürür |
| `pipe.batch(prompts, **settings)` | tek model oturumunda birden fazla istek çalıştırır |
| `pipe.map(prompts, **settings)` | bir iterable'ı sınırlı bellekle tembel olarak işler |
| `await pipe.arun(prompt, **settings)` | event loop dışında çalıştırır |
| `pipe.load()` / `pipe.close()` | ağırlıkları açıkça yükler veya serbest bırakır |
| `with pipe:` | girişte yükler, çıkışta serbest bırakır |

Çağrı sırasında verilen ayarlar, isteğin kendi parametrelerinden; onlar da pipeline'ın `options`
değerinden önceliklidir. Kabul edilen ayarlar mimariye bağlıdır (genellikle `steps`, `guidance`,
`width`, `height`, `negative_prompt`, `num_images`, `seed`, `latents` ve görüntü kabul eden
modellerde `image`). Model yüklenmeden önce doğrulanırlar; bilinmeyen ayar hata verir.

## Sonuçlar

`ImageResult`

| Alan | Açıklama |
| --- | --- |
| `images` | `Image` tuple'ı |
| `metadata` | çalıştırmayı açıklayan mapping: model kimliği, revizyon, seed, adımlar, boyut, aygıt, dtype |

`Image`, 8 bit RGB veya RGBA görüntüyü `.tensor` içinde `(height, width, channels)` boyutlu `uint8`
tensör olarak tutar.

| Üye | Açıklama |
| --- | --- |
| `width`, `height`, `channels` | boyutlar |
| `save(path)` | PNG yazar (Pillow gerekmez) |
| `to_png()` | PNG byte'ları |
| `Image.open(path)`, `Image.from_png(data)` | PNG okur |
| `to_pil()` | `PIL.Image.Image` nesnesine dönüştürür (`image` ek paketi gerekir) |

## Keşif

```python
imagehub.available_models()          # {"t2i": ("amused-256", ..., "z-image-turbo")}
imagehub.available_backends("t2i")   # selectors accepted as `backend=`
spec = imagehub.model_info("sd-v1-5")
spec.repo_id, spec.revision, spec.license, spec.image_input
```

`ModelSpec` her model için `id`, `task`, `family`, `repo_id`, `revision` (doğrulanmış commit),
`files`, `license`, `gated`, `provenance` ve `image_input` değerlerini kaydeder.

## İndirme

```python
from imagehub.pretrained import download_pretrained
download_pretrained(model_id, destination, *, revision=None, token=None, progress=None) -> Path
```

Aynı fonksiyon `python -m imagehub.pretrained <model-id> <empty-directory> [--revision REF]`
komutunu da çalıştırır. [Ağırlıkları indirme](guides/weights.md) rehberine bakın.

## Hatalar ve uyarılar

Kitaplığa özgü operasyonel hataları ele almak için `ImageHubError` yakalayın. Daha özel hata
yönetimi için kullanılamayan isteğe bağlı bağımlılıklarda `MissingDependencyError`, eksik veya
uyumsuz checkpoint dosyalarında `CheckpointError`, indirme ya da doğrulama hatalarında
`DownloadError`, geçersiz eşzamanlı backend kullanımında `BackendBusyError`, adapter görev
sözleşmesini ihlal ettiğinde `BackendContractError` kullanılabilir. Argüman doğrulaması standart
Python hataları üretmeye devam edebilir.

Kısıtlı checkpoint'ler `ImageHubLicenseWarning` üretir; gerektiğinde Python'ın `warnings`
filtreleriyle yapılandırın. `ImagehubLicenseWarning` uyumluluk alias'ı olarak korunur.

## Genişletme

| Ad | Amaç |
| --- | --- |
| `Backend`, `TextToImageBackend` | temel sınıflar: tembel yükleme, seçenek doğrulama, serileştirilmiş inference |
| `Pipeline`, `TextToImagePipeline` | ön işleme, forward, son işleme |
| `BackendRegistry`, `backend_registry` | seçicileri backend factory'leriyle eşler |
| `T2IRequest`, `Task` | istek ve görev türleri |

Yeni bir modelin nasıl ekleneceği için [Katkıda bulunma](contributing.md) sayfasına bakın.
