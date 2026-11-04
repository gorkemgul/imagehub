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

## Değerlendirme ve örnek görüntüler

```python
trainer = Trainer(
    model="sd-v1-5",
    checkpoint="checkpoints/sd-v1-5",
    args=TrainingArguments(
        output_dir="runs/my-dog",
        max_steps=1000,
        eval_strategy="steps",
        eval_steps=200,
        validation_prompts=["a photo of sks dog in a bucket"],
    ),
    train_dataset=ImageFolderDataset("my_images"),
    eval_dataset=ImageFolderDataset("my_val_images"),
)
trainer.train()
print(trainer.evaluate())   # {"eval_loss": 0.1234}
```

`eval_loss`, değerlendirme görüntüleri üzerinde gradyansız hesaplanan eğitim loss'udur. Difüzyon
loss'u rastgele gürültüye ve zaman adımına bağlıdır; bu yüzden bunlar görüntünün sırasına göre
tohumlanır ve görüntüler ortadan kırpılır. Aynı adaptörü iki kez değerlendirmek aynı sayıyı verir;
eğitim boyunca eğri zara değil adaptöre göre değişir. Her değerlendirme noktasında (ve sonda) her
doğrulama prompt'u o anki adaptörle `output_dir/samples/step-N/<i>.png` olarak çizilir; görüntü,
`seed`, `validation_num_inference_steps` ve `validation_guidance_scale` ile
`pipeline(..., lora=...)` çıktısının aynısıdır.

## Özel batch'ler

`data_collator`, veri setinden gelen öğe listesini bir batch'e çevirir. Varsayılanı görüntüleri
yeniden boyutlandırır, kırpar, çevirir, normalize eder ve alt yazıları tokenize eder. Kendi
collator'ınız `{"pixel_values": [-1, 1] aralığında (B, 3, H, W), "input_ids": (B, 77)}`
döndürmelidir:

```python
trainer = Trainer(..., data_collator=my_collator)
```

Birebir devam ettirme, rastgelelik durumu checkpoint'te saklanan varsayılan collator için
garantilidir.

## Birden fazla GPU

Aynı betiği `torchrun` ile çalıştırın; içinde hiçbir şey değişmez:

```bash
torchrun --nproc_per_node=4 train.py
```

Her süreç her batch'in kendi payıyla eğitilir ve adaptör gradyanlarının ortalaması alınır; böylece
bütün süreçlerde aynı adaptör kalır. Etkin batch `per_device_train_batch_size ×
gradient_accumulation_steps × süreç sayısı`dır. Yalnız ilk süreç yazdırır; checkpoint'leri, örnek
görüntüleri ve `save_model` çıktısını o yazar. Checkpoint aynı süreç sayısıyla devam ettirilir.
Dağıtık eğitim CUDA (NCCL) veya CPU (gloo) ile çalışır; MPS ile çalışmaz.

## Ayarlar

Yalnızca `output_dir` ve `max_steps` zorunludur. Varsayılanlar diffusers'ın
`train_text_to_image_lora.py` betiğini izler.

| Argüman | Varsayılan | Anlamı |
| --- | --- | --- |
| `learning_rate` | `1e-4` | AdamW öğrenme hızı |
| `per_device_train_batch_size`, `per_device_eval_batch_size` | `1`, `1` | her süreçte mikro-batch başına görüntü |
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
| `eval_strategy`, `eval_steps` | `"no"`, `logging_steps` | `"steps"` her `eval_steps` adımda değerlendirir |
| `validation_prompts` | yok | her değerlendirme noktasında çizilen prompt'lar |
| `validation_num_inference_steps`, `validation_guidance_scale` | `25`, `7.5` | bu görüntülerin örnekleme ayarları |
| `report_to`, `run_name` | `"none"`, `None` | `"wandb"`, kuruluysa Weights & Biases'e de yazar |
| `seed`, `device` | `0`, `"cpu"` | rastgelelik tohumu; GPU için `"cuda"` veya `"mps"` |

## Rank ve alpha seçimi

Adaptörün güncellemesi `lora_alpha / lora_rank` ile ölçeklenir. Difüzyon araçları küçük rank ve 1
civarı ölçek kullanır; dil ve konuşma eğiticileri genelde daha büyüklerini kullanır:

| Kaynak | Rank | Alpha | Ölçek |
| --- | --- | --- | --- |
| imagehub varsayılanı, diffusers eğitim betikleri (SD 1.5, SDXL, FLUX.1) | 4 | 4 | 1 |
| kohya sd-scripts varsayılanı | 4 | 1 | 0.25 |
| PEFT varsayılanı | 8 | 8 | 1 |
| voicehub varsayılanı (konuşma) | 8 | 16 | 2 |

Bir stil ya da basit bir konu için rank 4 yeter. Daha fazla ayrıntı için rank'ı 16-32'ye çıkarın ve
alpha'yı rank'a eşit tutun; böylece ölçek ve uygun öğrenme hızı değişmez.

## Desteklenen modeller

`get_training_spec(model_id)` bir modelin eğitilip eğitilemeyeceğini söyler;
`list_training_specs()` katalogdaki her modeli listeler.

```python
from imagehub.training import get_training_spec

spec = get_training_spec("sd-v1-5")
print(spec.support.value, spec.methods)   # native ('lora',)
```

`native`, modelin görüntü ve alt yazılarla, referans eğitim betiğine karşı doğrulanmış bir hedefle
eğitildiği anlamına gelir; `inference_only` henüz eğitim yolu olmadığını söyler. `methods` neyin
eğitilebildiğini listeler (bugün `lora`).

<!-- training-support:start -->
| Model | Destek | Yöntemler |
| --- | --- | --- |
| `sd-v1-1` | native | lora |
| `sd-v1-2` | native | lora |
| `sd-v1-3` | native | lora |
| `sd-v1-4` | native | lora |
| `sd-v1-5` | native | lora |
| `dreamshaper-8` | native | lora |
| `segmind-small-sd` | native | lora |
| `segmind-tiny-sd` | native | lora |
| `bk-sdm-base` | native | lora |
| `bk-sdm-small` | native | lora |
| `bk-sdm-tiny` | native | lora |
| `sd-v2-1` | native | lora |
| `bk-sdm-v2-base` | native | lora |
| `bk-sdm-v2-small` | native | lora |
| `bk-sdm-v2-tiny` | native | lora |
<!-- training-support:end -->

Diğer bütün modeller şimdilik `inference_only`; birkaç adımlık damıtılmış `sd-turbo`,
`lcm-dreamshaper-v7` ve `sdxs-512-0.9` de dahil. Sırada SDXL ve transformer aileleri var.

## Neler doğrulandı

CPU'da float32 ile ilk adımın loss'u ve bütün LoRA gradyanları, küçük modellerde ve `sd-v1-5`'te
diffusers eğitim adımıyla birebir aynıdır. Isınma, cosine planı ve kırpma içeren dört eğitim adımı
diffusers + PEFT döngüsüyle aynıdır. Kaydedilen bir adaptör imagehub'da ve diffusers'ta aynı
görüntüyü verir. Ayrıntılar [doğrulama kaydında](../validation.md#lora-training).
