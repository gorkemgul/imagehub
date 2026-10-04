"""Offline text-to-image with a downloaded snapshot.

python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
python examples/text_to_image.py checkpoints/sd-v1-5 "a photograph of an astronaut riding a horse"
"""

import sys

from imagehub import pipeline

model, prompt = sys.argv[1], sys.argv[2]
with pipeline("t2i", backend="sd-v1-5", model=model) as pipe:
    result = pipe(prompt, seed=42)
path = result.images[0].save("output.png")
print(path, result.metadata)
