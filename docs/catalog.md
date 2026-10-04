# Model kataloğu — imagehub 0.1

**1 mimari ailesi, 5 yayımlanmış checkpoint.** Bir model, ayrı yayımlanmış bir ağırlık setidir;
fp16 kopyalar, EMA olmayan ağırlıklar ve tek dosyalık `.ckpt` sürümleri ayrı model sayılmaz.

`pipeline("t2i", backend=model_id, model="yerel/klasor")` katalog kimliğiyle aileyi seçer; hangi
ağırlıkların kullanılacağını `model` klasörü belirler.

**Doğrulama seviyeleri:** `full` — tensörler strict yüklendi ve CPU fp32'de diffusers ile parity
kanıtlandı. `header_only` — tensör adları ve şekilleri eşleşti, üretim test edilmedi.
`experimental` — özgün ağırlıklarla test edilmedi. Bu sürümde bütün modeller `full`.

| Model kimliği | Aile | Kaynak | Revision | Köken | Doğrulama | Ağırlık lisansı | Awesome kataloğu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sd-v1-1` | `stable-diffusion` | [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1) | `574555a` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-2` | `stable-diffusion` | [CompVis/stable-diffusion-v1-2](https://huggingface.co/CompVis/stable-diffusion-v1-2) | `f519eeb` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-3` | `stable-diffusion` | [CompVis/stable-diffusion-v1-3](https://huggingface.co/CompVis/stable-diffusion-v1-3) | `159f26f` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-4` | `stable-diffusion` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | `133a221` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-5` | `stable-diffusion` | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `451f4fe` | topluluk aynası | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |

`sd-v1-5`'in özgün deposu (`runwayml/stable-diffusion-v1-5`) kaldırılmıştır; kullanılan depo bir
topluluk aynasıdır. v1.1–v1.3 fp32 ağırlıkları yalnızca `.bin` olarak yayımlandığı için
`weights_only=True` ile, yalnızca tensör içerecek şekilde yüklenir.
