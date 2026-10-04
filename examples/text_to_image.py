"""Offline text-to-image with a downloaded snapshot.

python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
python examples/text_to_image.py checkpoints/sd-v1-5 "a photograph of an astronaut riding a horse"
python examples/text_to_image.py checkpoints/sd-v1-4 "a cat" sd-v1-4

The optional third argument is the catalog id (default: sd-v1-5).
"""

import sys

from imagehub import pipeline

model, prompt = sys.argv[1], sys.argv[2]
backend = sys.argv[3] if len(sys.argv) > 3 else "sd-v1-5"
with pipeline("t2i", backend=backend, model=model) as pipe:
    result = pipe(prompt, seed=42)
path = result.images[0].save("output.png")
print(path, result.metadata)
