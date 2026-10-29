# Kurulum

imagehub için Python 3.11 veya yenisi gerekir. Tek zorunlu bağımlılık PyTorch'tur.

```bash
pip install imagehub
```

| Ek paket | Eklediği | Gerektiği yer |
| --- | --- | --- |
| `image` | Pillow | `Image.to_pil()` ve PIL görüntülerini girdi olarak iletmek |
| `reference` | diffusers, transformers, torchvision, sentencepiece, protobuf | referansa karşı isteğe bağlı parity testlerini çalıştırmak |
| `dev` | pytest, ruff, mypy, build | imagehub üzerinde çalışmak |
| `docs` | MkDocs Material, static i18n | bu siteyi derlemek |

```bash
pip install "imagehub[image]"
pip install "imagehub[docs]"
```

PNG kaydetme ve okuma yerleşik olarak gelir; dolayısıyla `Image.save("out.png")` Pillow olmadan çalışır.

!!! tip "Önce donanımınıza uygun PyTorch'u kurun"
    CUDA veya yalnızca CPU destekli bir sürüm istiyorsanız `pip install imagehub` komutundan önce
    [resmî seçiciyi](https://pytorch.org/get-started/locally/) kullanarak PyTorch'u kurun.

## Kurulumu denetleyin

```python
import imagehub
print(imagehub.__version__)
print(len(imagehub.available_models()["t2i"]), "models in the catalog")
```
