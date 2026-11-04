# LoRA eğitimi

`imagehub.training`, kendi görüntülerinizle bir LoRA adaptörü eğitir. Kendi LoRA katmanlarıyla ve
kendi eğitim döngüsüyle gelir (PEFT veya Accelerate gerekmez) ve isteğe bağlı olarak kurulur;
çekirdek paket yalnızca PyTorch'a bağlı kalır.

## 1. Kurulum ve indirme

```bash
pip install "imagehub[training]"
```

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

## 2. Görüntüleri hazırlama

Görüntüleri tek bir klasöre koyun; yanlarına her görüntü için bir satır içeren bir
`metadata.jsonl` dosyası ekleyin:

```
my_images/
  metadata.jsonl
  0001.png
  0002.jpg
```

```json
{"file_name": "0001.png", "text": "a photo of sks dog on the beach"}
{"file_name": "0002.jpg", "text": "a photo of sks dog in the snow"}
```

Eksik dosyalar, boş alt yazılar ve okunamayan görüntüler veri seti oluşturulurken bildirilir.
Öğeleri `{"image": PIL görüntüsü, "text": str}` olan, uzunluğu bilinen her veri seti de çalışır;
örneğin bir Hugging Face `datasets` veri seti. imagehub o pakete bağımlı değildir.

## 3. Eğitme

```python
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments

trainer = Trainer(
    model="sd-v1-5",
    checkpoint="checkpoints/sd-v1-5",
    args=TrainingArguments(output_dir="runs/my-dog", max_steps=1000),
    train_dataset=ImageFolderDataset("my_images"),
)
result = trainer.train()
print(result.training_loss)
trainer.save_model("runs/my-dog/final")
```

Loss her `logging_steps` adımda yazdırılır. Her `save_steps` adımda ve sonda, eğitici
`output_dir/checkpoint-N/` klasörüne adaptörü ve eğitimin tam durumunu yazar. Eğitim yarıda
kalırsa son kayıttan devam edin:

```python
trainer.train(resume_from_checkpoint=True)
```

Devam ettirilen eğitim, kesintisiz eğitimle birebir aynı adaptörü verir.

## 4. Adaptörü kullanma

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5",
              lora="runs/my-dog/final/adapter.safetensors") as pipe:
    image = pipe("a photo of sks dog in a bucket", seed=0).images[0]
```

`adapter.safetensors` diffusers düzenindedir; diffusers `load_lora_weights` de okur.

## Ayarlar

Yalnızca `output_dir` ve `max_steps` zorunludur. Varsayılanlar diffusers'ın
`train_text_to_image_lora.py` betiğini izler.

| Argüman | Varsayılan | Anlamı |
| --- | --- | --- |
| `learning_rate` | `1e-4` | AdamW öğrenme hızı |
| `train_batch_size` | `1` | mikro-batch başına görüntü |
| `gradient_accumulation_steps` | `1` | optimizer adımı başına mikro-batch |
| `lr_scheduler`, `lr_warmup_steps` | `"constant"`, `0` | ayrıca `constant_with_warmup`, `linear`, `cosine` |
| `max_grad_norm` | `1.0` | gradyan kırpma; `0` kapatır |
| `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_epsilon` | `0.9`, `0.999`, `1e-2`, `1e-8` | AdamW ayarları |
| `mixed_precision` | `"no"` | `"fp16"` veya `"bf16"`; adaptör ve optimizer float32 kalır |
| `resolution`, `center_crop`, `random_flip` | `512`, `False`, `False` | görüntü dönüşümü |
| `lora_rank`, `lora_alpha`, `lora_dropout` | `4`, `4`, `0.0` | adaptör boyutu ve ölçeği (`alpha / rank`) |
| `lora_target_modules` | `to_q`, `to_k`, `to_v`, `to_out.0` | adaptör takılan U-Net katmanları |
| `noise_offset`, `snr_gamma`, `prediction_type` | `0.0`, `None`, zamanlayıcınınki | hedef seçenekleri |
| `logging_steps`, `save_steps` | `10`, `500` | kayıt ve checkpoint sıklığı |
| `report_to`, `run_name` | `"none"`, `None` | `"wandb"`, kuruluysa Weights & Biases'e de yazar |
| `seed`, `device` | `0`, `"cpu"` | rastgelelik tohumu; GPU için `"cuda"` veya `"mps"` |

## Desteklenen modeller

`get_training_spec(model_id)` bir modelin eğitilip eğitilemeyeceğini söyler;
`list_training_specs()` katalogdaki her modeli listeler.

```python
from imagehub.training import get_training_spec

spec = get_training_spec("sd-v1-5")
print(spec.support.value, spec.family)   # lora stable-diffusion
```

<!-- training-support:start -->
| Model | Destek |
| --- | --- |
| `sd-v1-1` | lora |
| `sd-v1-2` | lora |
| `sd-v1-3` | lora |
| `sd-v1-4` | lora |
| `sd-v1-5` | lora |
| `dreamshaper-8` | lora |
| `segmind-small-sd` | lora |
| `segmind-tiny-sd` | lora |
| `bk-sdm-base` | lora |
| `bk-sdm-small` | lora |
| `bk-sdm-tiny` | lora |
| `sd-v2-1` | lora |
| `bk-sdm-v2-base` | lora |
| `bk-sdm-v2-small` | lora |
| `bk-sdm-v2-tiny` | lora |
<!-- training-support:end -->

Diğer bütün modeller şimdilik `inference_only`; birkaç adımlık damıtılmış `sd-turbo`,
`lcm-dreamshaper-v7` ve `sdxs-512-0.9` de dahil. Sırada SDXL ve transformer aileleri var.

## Neler doğrulandı

CPU'da float32 ile ilk adımın loss'u ve bütün LoRA gradyanları, küçük modellerde ve `sd-v1-5`'te
diffusers eğitim adımıyla birebir aynıdır. Isınma, cosine planı ve kırpma içeren dört eğitim adımı
diffusers + PEFT döngüsüyle aynıdır. Kaydedilen bir adaptör imagehub'da ve diffusers'ta aynı
görüntüyü verir. Ayrıntılar [doğrulama kaydında](../validation.md#lora-training).
