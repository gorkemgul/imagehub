"""LoRA fine-tuning (``pip install "imagehub[training]"``).

```python
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments

trainer = Trainer(
    model="sd-v1-5",
    checkpoint="checkpoints/sd-v1-5",
    args=TrainingArguments(output_dir="runs/my-dog", max_steps=1000),
    train_dataset=ImageFolderDataset("my_images"),
)
trainer.train()
trainer.save_model("runs/my-dog/final")
```
"""

from imagehub.training.arguments import TrainingArguments
from imagehub.training.data import ImageFolderDataset, StableDiffusionCollator, preprocess_image
from imagehub.training.sd1 import StableDiffusionLoRAObjective
from imagehub.training.specs import (
    TrainingSpec,
    TrainingSupport,
    get_training_spec,
    list_training_specs,
)
from imagehub.training.trainer import Trainer, TrainOutput

__all__ = [
    "ImageFolderDataset",
    "StableDiffusionCollator",
    "StableDiffusionLoRAObjective",
    "TrainOutput",
    "Trainer",
    "TrainingArguments",
    "TrainingSpec",
    "TrainingSupport",
    "get_training_spec",
    "list_training_specs",
    "preprocess_image",
]
