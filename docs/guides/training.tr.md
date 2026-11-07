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

with pipeline(
    "t2i",
    backend="sd-v1-5",
    model="checkpoints/sd-v1-5",
    lora="runs/my-dog/final/adapter.safetensors",
) as pipe:
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
print(trainer.evaluate())  # {"eval_loss": 0.1234}
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
| `lora_rank`, `lora_alpha`, `lora_dropout` | model betiğinin varsayılanı, model betiğinin varsayılanı, `0.0` | adaptör boyutu ve ölçeği (`alpha / rank`); Qwen-Image 2.1 (16 / 16) hariç 4 / 4 |
| `lora_target_modules` | ailenin varsayılanı | adaptör takılan katmanlar: U-Net'lerde, Qwen-Image, Z-Image ve Lumina 2'de `to_q`, `to_k`, `to_v`, `to_out.0`; HiDream-I1 görüntü dikkatinde `to_q`, `to_k`, `to_v`, `to_out`; SD 3'te sekiz ortak dikkat izdüşümü; Sana'da `to_k`, `to_q`, `to_v` |
| `gradient_checkpointing` | `False` | belleği azaltmak için U-Net ya da transformer bloklarını geri yayılımda yeniden hesaplar |
| `noise_offset`, `snr_gamma`, `prediction_type` | `0.0`, `None`, zamanlayıcınınki | U-Net (DDPM) hedef seçenekleri |
| `weighting_scheme`, `logit_mean`, `logit_std`, `mode_scale` | ailenin varsayılanı (SD 3 için `"logit_normal"`, FLUX.1, Sana, Qwen-Image, Z-Image, Lumina 2 ve HiDream-I1 için `"none"`), `0.0`, `1.0`, `1.29` | flow matching zaman adımı örneklemesi: ayrıca `mode`, `sigma_sqrt`, `cosmap`, `none` |
| `precondition_outputs` | `True` | flow matching: `noise - latents` yerine ön koşullanmış çıktıyı latent'lerle karşılaştırır |
| `max_sequence_length` | `77` (SD 3), `512` (FLUX.1, Qwen-Image ve Z-Image), `300` (Sana), `256` (Lumina 2), `128` (HiDream-I1) | metin prompt uzunluğu |
| `system_prompt` | `None` | Lumina 2'nin isteğe bağlı Gemma sistem prompt'u; diğer aileler reddeder |
| `complex_human_instruction` | `None` | Sana'nın isteğe bağlı prompt iyileştirme yönergesi; diğer aileler reddeder |
| `guidance_scale` | `3.5` | FLUX.1: guidance embedding'i olan checkpoint'lerde eğitim sırasındaki değeri |
| `logging_steps`, `save_steps` | `10`, `500` | kayıt ve checkpoint sıklığı |
| `eval_strategy`, `eval_steps` | `"no"`, `logging_steps` | `"steps"` her `eval_steps` adımda değerlendirir |
| `validation_prompts` | yok | her değerlendirme noktasında çizilen prompt'lar |
| `validation_num_inference_steps`, `validation_guidance_scale` | `25`, `7.5` | bu görüntülerin örnekleme ayarları |
| `report_to`, `run_name` | `"none"`, `None` | `"wandb"`, `"clearml"` ya da ikisi bir liste olarak: `train_loss`, `lr`, `eval_loss` ve örnek görüntüleri gönderir (paket kurulu olmalı) |
| `seed`, `device` | `0`, `"cpu"` | rastgelelik tohumu; GPU için `"cuda"` veya `"mps"` |

## Rank ve alpha seçimi

Adaptörün güncellemesi `lora_alpha / lora_rank` ile ölçeklenir. Difüzyon araçları küçük rank ve 1
civarı ölçek kullanır; dil ve konuşma eğiticileri genelde daha büyüklerini kullanır:

| Kaynak | Rank | Alpha | Ölçek |
| --- | --- | --- | --- |
| imagehub varsayılanı, diffusers eğitim betikleri (SD 1.5, SDXL, FLUX.1) | 4 | 4 | 1 |
| kohya sd-scripts varsayılanı | 4 | 1 | 0.25 |
| PEFT varsayılanı | 8 | 8 | 1 |

Bir stil ya da basit bir konu için rank 4 yeter. Daha fazla ayrıntı için rank'ı 16-32'ye çıkarın ve
alpha'yı rank'a eşit tutun; böylece ölçek ve uygun öğrenme hızı değişmez.

## Desteklenen modeller

`get_training_spec(model_id)` bir modelin eğitilip eğitilemeyeceğini söyler;
`list_training_specs()` katalogdaki her modeli listeler.

```python
from imagehub.training import get_training_spec

spec = get_training_spec("sd-v1-5")
print(spec.support.value, spec.methods)  # native ('lora',)
```

`native`, modelin görüntü ve alt yazılarla, referans eğitim betiğine karşı doğrulanmış bir hedefle
eğitildiği anlamına gelir; `inference_only` henüz eğitim yolu olmadığını söyler. `methods` neyin
eğitilebildiğini listeler (bugün `lora`).

<!-- training-support:start -->
| Model | Destek | Yöntemler |
| --- | --- | --- |
| `qwen-image` | native | lora |
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
| `sdxl-base-1.0` | native | lora |
| `juggernaut-xl-v9` | native | lora |
| `animagine-xl-4.0` | native | lora |
| `dreamshaper-xl-1.0` | native | lora |
| `z-image` | native | lora |
| `qwen-image-2.1` | native | lora |
| `sd3-medium` | native | lora |
| `sd3.5-medium` | native | lora |
| `sd3.5-large` | native | lora |
| `flux1-dev` | native | lora |
| `flux1-krea-dev` | native | lora |
| `flux-1-lite-8b` | native | lora |
| `flux1-kontext-dev` | native | lora |
| `sana-1600m-1024` | native | lora |
| `sana-600m-1024` | native | lora |
| `sana-1.5-4.8b-1024` | native | lora |
| `lumina-image-2.0` | native | lora |
| `ssd-1b` | native | lora |
| `segmind-vega` | native | lora |
| `koala-700m` | native | lora |
| `koala-1b` | native | lora |
| `playground-v2` | native | lora |
| `hidream-i1-dev` | native | lora |
| `hidream-i1-full` | native | lora |
<!-- training-support:end -->

Diğer bütün modeller şimdilik `inference_only`: birkaç adımlık damıtılmış checkpoint'ler
(`sd-turbo`, `lcm-dreamshaper-v7`, `sdxs-512-0.9`, SDXL Lightning, Hyper-SD, DMD2, `sdxl-turbo`,
KOALA Lightning), SDXL refiner (görüntüden görüntüye), Playground v2.5 (DDPM değil, EDM) ve
`sd3.5-large-turbo` (birkaç adımlık damıtılmış) ve `sana-sprint-1.6b-1024` (damıtılmış).
Diğer transformer aileleri sırada (#202).

Sana yalnızca transformer'ını, betikteki `to_k`, `to_q`, `to_v` hedefleriyle eğitir. DC-AE'nin
latentleri deterministiktir; bu nedenle rastgele akış önce gürültüyü, sonra flow zaman adımını
çeker ve karışık hassasiyette otomatik kodlayıcı float32 kalır. Gemma 2 koşullaması Sana'nın
attention maskesini ve 300 token varsayılanını kullanır; complex-human yönergesi isteğe bağlıdır
ve betikteki gibi varsayılanı `None`'dır. Gradient checkpointing Sana transformer bloklarını geri
yayılımda yeniden hesaplar.

Qwen-Image ve Qwen-Image 2.1 yalnızca transformer LoRA adaptörlerini `to_k`, `to_q`, `to_v` ve
`to_out.0` üzerinde eğitir. Her ikisi `noise - latents` flow hedefini, betiklerin `"none"`
yoğunluk varsayılanını ve VAE posterior normalizasyonunu kullanır. Özgün Qwen-Image 512 tokenlık
betimleyici sohbet prompt'unu, tek-kare latenti ve 2×2 uzamsal paketlemeyi kullanır; 2.1 sistem
prompt tokenlarını çıkarır ve birleşik prompt/görüntü dizisini image-slot maskesiyle eğitir.
`max_sequence_length=512` yalnızca özgün sürüm içindir. Rank ve alpha ayarlanmadığında Qwen-Image
betiğinin 4 / 4, Qwen-Image 2.1 ise 16 / 16 varsayılanlarını kullanır. Her iki backend de transformer
adaptörlerini `lora=`, `lora_scale=` ve `lora_fuse=` ile yükler.

Z-Image yalnızca transformer LoRA adaptörlerini `to_k`, `to_q`, `to_v` ve `to_out.0` üzerinde
eğitir; betikteki rank / alpha varsayılanı 4 / 4, ağırlıklandırma `"none"`, sohbet şablonu prompt
uzunluğu 512'dir. KL VAE'nin deterministik posterior ortalaması yayınlanan shift ve scale ile
normalleştirilir; modele `(1000 - timestep) / 1000` verilir ve ters işaretli tahmin `noise -
latents` ile karşılaştırılır. Gradient checkpointing her Z-Image transformer bloğunu yeniden
hesaplar. Backend, transformer adaptörlerini `lora=`, `lora_scale=` ve `lora_fuse=` ile yükler.
`z-image-turbo` damıtılmıştır ve `inference_only` kalır.

Lumina-Image 2.0 yalnızca transformer LoRA adaptörlerini `to_k`, `to_q`, `to_v` ve `to_out.0`
üzerinde eğitir; betikteki rank / alpha varsayılanı 4 / 4, ağırlıklandırma `"none"`, Gemma prompt
uzunluğu 256 ve sistem prompt'u varsayılandır. Float32 KL VAE posteriörü kaydırılıp ölçeklendikten
sonra ters flow matching kullanılır: transformer `timestep / 1000` alır ve `latents - noise`
tahmin eder. Gradient checkpointing Lumina 2 transformer bloklarını yeniden hesaplar. Backend
adaptörleri `lora=`, `lora_scale=` ve `lora_fuse=` ile yükler; Lumina-Next ve Lumina-DiMOO
`inference_only` kalır.
Referans testleri betik varsayılanlarını, görüntü dönüşümünü, varsayılan sistem prompt'lu token
kimlikleri ve dikkat maskesini, her ağırlıklandırmadaki adım çekilişlerini/kayıp/gradientleri ve
diffusers yüklemesinden sonra rastgele veya üç adımda eğitilmiş adaptörlerin son latentleri ile
uint8 piksellerini tam olarak karşılaştırır.

HiDream-I1 Dev ve Full yalnızca transformer LoRA adaptörlerini görüntü akışının `to_k`, `to_q`,
`to_v` ve `to_out` izdüşümlerinde eğitir; rank / alpha 4 / 4, ağırlıklandırma `"none"`, T5 ve
Llama için prompt uzunluğu 128'dir. Dört metin kodlayıcısı ve VAE eğitim ağırlık dtype'ını kullanır.
Örneklenen, kaydırılan ve ölçeklenen VAE latenti `(1 - sigma) * latents + sigma * noise` ile
karıştırılır; ham zaman adımı inference-MoE transformer'a verilir ve ters işaretli tahmin `noise -
latents` ile karşılaştırılır. Gradient checkpointing çift ve tek akış bloklarını kapsar. Backend
transformer adaptörlerini `lora=`, `lora_scale=` ve `lora_fuse=` ile yükler. Referans testleri,
diffusers içinde rastgele tekli, yığılı (0.8 / 0.5) ve birleştirilmiş adaptörlerle üç adımda
eğitilmiş bir adaptörü yükler; kayıtlı hedefleri, değişen çıktıları ve aynı açık başlangıç
latentlerinden elde edilen son latentleri ile uint8 pikselleri karşılaştırır. `hidream-i1-fast`
damıtılmıştır ve `inference_only` kalır.

SDXL'de dönüşüm, diffusers'ın SDXL betiğindeki gibi önce yeniden boyutlandırır, sonra çevirir,
sonra kırpar; her görüntünün özgün boyutunu ve kırpma köşesini U-Net'in boyut koşullaması olarak
kaydeder. Karışık hassasiyette otomatik kodlayıcı float32 kalır. `gradient_checkpointing=True`,
bellekten kazanmak için U-Net'in transformer ve resnet bloklarını geri yayılımda yeniden hesaplar;
gradyanlar aynıdır.

SD 3 / 3.5, diffusers'ın `train_dreambooth_lora_sd3.py` betiğindeki flow matching hedefiyle
eğitilir: zaman adımları kaydırılmış plandan `weighting_scheme`'e göre çekilir, gürültülü latent
`(1 - sigma) * latents + sigma * noise` olur ve ön koşullanmış çıktı latent'lerle karşılaştırılır.
Prompt, çıkarımdaki gibi CLIP-L ve CLIP-G'nin sondan bir önceki katmanlarını T5 ile birleştirir.
DDPM seçenekleri (`snr_gamma`, `prediction_type`, `noise_offset`) SD 3'te, flow matching
seçenekleri de U-Net ailelerinde reddedilir.

FLUX.1 dev, Krea dev ve Lite, diffusers'ın karşılık gelen FLUX DreamBooth adımıyla eğitilir:
float32 VAE latentleri kaydırılır ve ölçeklenir, FLUX image id'leriyle paketlenir ve doğrudan
`noise - latents` ile karşılaştırılır. CLIP havuzlanmış izdüşümü ve T5 prompt'u varsayılan olarak
512 token kullanır. Guidance embedding yalnızca kullanan checkpoint'lerde `guidance_scale` alır.
Adaptör hedefleri hem çift-akışlı hem tek-akışlı blokları kapsar. `flux1-kontext-dev`, Kontext
betiğinin koşul görüntüsü olmadan yaptığı gibi aynı adımla eğitilir ve tek akışlı `proj_mlp`
katmanlarını da hedefler; görüntü koşullu Kontext eğitimi dahil değildir. `flux1-schnell` birkaç
adımlık damıtılmış olduğundan `inference_only` kalır.

```python
trainer = Trainer(
    model="sd3.5-medium",
    checkpoint="checkpoints/sd3.5-medium",
    args=TrainingArguments(
        output_dir="runs/my-dog-sd35",
        max_steps=1000,
        resolution=1024,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        weighting_scheme="logit_normal",
    ),
    train_dataset=ImageFolderDataset("my_images"),
)
```

## Neler doğrulandı

CPU'da float32 ile ilk adımın loss'u ve bütün LoRA gradyanları, küçük modellerde ve `sd-v1-5`'te
diffusers eğitim adımıyla birebir aynıdır. Isınma, cosine planı ve kırpma içeren dört eğitim adımı
diffusers + PEFT döngüsüyle aynıdır. Kaydedilen bir adaptör imagehub'da ve diffusers'ta aynı
görüntüyü verir. Ayrıntılar [doğrulama kaydında](../validation.md#lora-training).
