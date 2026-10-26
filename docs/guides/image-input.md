# Image input

Most models take a text prompt only. A model that also accepts an input image says so on its page and in
the table below; passing `image` to any other model raises an error.

<!-- ih:image-input -->

```python
from PIL import Image
from imagehub import pipeline

with pipeline("t2i", backend="flux1-kontext-dev", model="checkpoints/flux1-kontext-dev") as edit:
    result = edit("Add a red hat to the scene", image=Image.open("photo.png"), seed=0)

result.images[0].save("edited.png")
```

- `image` takes a `PIL.Image.Image` or an `imagehub.Image`. Pillow is only needed for the first.
- When the option is `optional`, leaving it out generates from the text alone.
- There is no separate "edit" task: the same `t2i` pipeline takes the extra option.
