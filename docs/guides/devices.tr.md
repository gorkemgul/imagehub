# Aygıtlar ve hassasiyet

```python
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="cpu")   # default
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps")   # Apple silicon
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="cuda")  # NVIDIA
```

Kullanılamayan bir aygıt isterseniz hata mesajı kullanılabilir seçenekleri listeler.

## Hassasiyet

Model destekliyorsa `dtype`; `torch.float32` (varsayılan), `torch.float16` veya `torch.bfloat16`
değerlerini kabul eder:

```python
import torch
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="cuda", dtype=torch.float16)
```

## Nerede ne iddia ediliyor

| Aygıt | Hassasiyet | İddia |
| --- | --- | --- |
| CPU | float32 | **Parity**: kayıtlı checkpoint'ler için referansla bit düzeyinde aynı |
| MPS | float32 | Duman testi: sonlu çıktı, doğru boyut; CPU farkı kaydedilir |
| CUDA | float32 | Henüz test edilmedi |
| herhangi biri | float16 / bfloat16 | Parity iddiası yok |

!!! warning "Hızlı olmak aynı olmak değildir"
    Farklı donanımlar kayan noktalı sayıları farklı sırada toplar; bu nedenle GPU çıktısı CPU
    çıktısından birkaç piksel değeri kadar farklı olabilir. imagehub ölçülen farkı kaydeder ve bunu
    parity olarak adlandırmaz.

Tüm sayılar [doğrulama kaydında](../validation.md#device-smoke-tests-no-parity-claim) bulunur.
