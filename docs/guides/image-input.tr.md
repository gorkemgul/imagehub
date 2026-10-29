# Görüntü girdisi

Çoğu model yalnızca metin prompt'u alır. Görüntü girdisi de kabul eden modeller bunu kendi
sayfalarında ve aşağıdaki tabloda belirtir; başka bir modele `image` iletmek hata verir.

<!-- ih:image-input -->

```python
from PIL import Image
from imagehub import pipeline

with pipeline("t2i", backend="flux1-kontext-dev", model="checkpoints/flux1-kontext-dev") as edit:
    result = edit("Add a red hat to the scene", image=Image.open("photo.png"), seed=0)

result.images[0].save("edited.png")
```

- `image`, `PIL.Image.Image` veya `imagehub.Image` kabul eder. Pillow yalnızca ilki için gerekir.
- Seçenek `optional` ise boş bırakıldığında yalnızca metinden görüntü üretilir.
- Ayrı bir "edit" görevi yoktur; aynı `t2i` pipeline'ı ek seçeneği alır.
