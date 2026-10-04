# Stable Diffusion 1.x

Katalog: [`stable-diffusion-1`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Modeller: `sd-v1-1` … `sd-v1-5` · Görev: `t2i`

**Mimari.** f=8 KL autoencoder (512×512×3 görsel ↔ 64×64×4 latent), cross-attention ile koşullanan
860M parametreli U-Net ve donmuş CLIP ViT-L/14 metin kodlayıcı (son katman, havuzlanmamış çıktı).
Zamanlayıcı, checkpoint'lerle yayımlanan PNDM'dir (`skip_prk_steps=True`, yani PLMS):
50 adım 51 U-Net değerlendirmesi yapar.

**Kullanım.**

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps") as pipe:
    result = pipe("bir kedi, yağlı boya", seed=42, negative_prompt="bulanık")
    result.images[0].save("kedi.png")
```

**Ayarlar.** `steps` (50), `guidance` (7.5; 1 ve altı guidance'ı kapatır), `width`/`height`
(512, 8'in katı), `negative_prompt`, `num_images` (1), `seed`, `latents`. Tanınmayan ayarlar hata verir.

**Seed uyumu.** CPU'da aynı seed, diffusers'ta `generator=torch.Generator("cpu").manual_seed(seed)`
ile üretilen görselin aynısını verir ([doğrulama](validation.md)). MPS'te duman testi yapıldı (CPU'ya göre en fazla 1 piksel fark, PSNR 72,8 dB); CUDA henüz test edilmedi. Bu cihazlarda birebir
aynılık iddiası yoktur.

**Kapsam dışı.** Safety checker ve invisible watermark yoktur. img2img, diğer zamanlayıcılar ve fp16
parity bu sürümde yoktur.

**Lisans.** Ağırlıklar CreativeML OpenRAIL-M lisanslıdır; lisansın kullanım kısıtları ağırlıkları
indiren ve kullanan kişiyi bağlar. imagehub ağırlık dağıtmaz.
