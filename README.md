# imagehub

Ağırlığı açık görsel üretim mimarileri için, tek bir API altında, yalnızca PyTorch ile yeniden
yazılmış inference kütüphanesi. **Tek zorunlu çalışma bağımlılığı PyTorch'tur.** Her model,
özgün uygulamayla aynı çıktıyı verdiği kanıtlanarak eklenir ([doğrulama](docs/validation.md)).
CPU çıktıları diffusers ile birebir aynıdır (bit düzeyinde özdeş görseller); ayrıntılar
[doğrulama](docs/validation.md) sayfasındadır.
Mimariler [awesome-text-to-image-architectures](https://github.com/gorkemgul/awesome-text-to-image-architectures)
kataloğuna bağlıdır.

![sd-v1-5, seed 42](artifacts/sd-v1-5-seed42.png)

## Kurulum

    pip install imagehub            # yalnızca torch
    pip install "imagehub[image]"   # Image.to_pil() için Pillow

## Kullanım

    python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5

    from imagehub import pipeline

    with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
        result = pipe("a photograph of an astronaut riding a horse", seed=42)
    result.images[0].save("astronot.png")
    print(result.metadata)

İndirme yalnızca `imagehub.pretrained` ile ve açıkça yapılır: doğrulanmış commit sabitlenir,
her dosyanın sha256'sı kontrol edilir, `imagehub_snapshot.json` makbuzu yazılır. `pipeline()` hiçbir
zaman ağa çıkmaz.

## Modeller

| Aile | Checkpoint'ler | Görev | Doğrulama |
| --- | --- | --- | --- |
| Stable Diffusion 1.x | `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4`, `sd-v1-5` | `t2i` | CPU fp32 parity |

[Katalog](docs/catalog.md) · [Model sayfaları](docs/models.md) · [Doğrulama](docs/validation.md) ·
[Üçüncü taraf bildirimleri](THIRD_PARTY_NOTICES.md)

## Geliştirme

    uv venv -p 3.13 .venv && uv pip install --python .venv -e ".[dev,image]"
    .venv/bin/python -m pytest

Referans karşılaştırmaları isteğe bağlıdır ve hiçbir şey indirmez:

    uv pip install --python .venv -e ".[dev,image,reference]"
    IMAGEHUB_REFERENCE_TESTS=1 IMAGEHUB_SD_V1_5=checkpoints/sd-v1-5 .venv/bin/python -m pytest tests/reference

## Lisans

Kod Apache-2.0. Model ağırlıkları kendi lisanslarına tabidir ve imagehub ile dağıtılmaz.
