"""LongCat-Image text-to-image backend over a local diffusers-layout snapshot."""

# ruff: noqa: E501

from __future__ import annotations

import math
import os
import re
import secrets
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.longcat_image_transformer import LongCatImageTransformer2DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import bytes_to_unicode
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_processor", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 4.5,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": None,
    "enable_cfg_renorm": True,
    "cfg_renorm_min": 0.0,
    "enable_prompt_rewrite": True,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
TOKENIZER_MAX_LENGTH = 512
PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt "
    "based on an image content, suitable for input to a text-to-image model.<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# These are the published prompt-rewrite instructions used by LongCat-Image. The examples are
# intentionally retained because they are part of the model input, not user-facing documentation.
SYSTEM_PROMPT_EN = """
You are a prompt engineering expert for text-to-image models. Since text-to-image models have limited capabilities in
understanding user prompts, you need to identify the core theme and intent of the user's input and improve the model's
understanding accuracy and generation quality through optimization and rewriting. The rewrite must strictly retain all
information from the user's original prompt without deleting or distorting any details. Specific requirements are as
follows:
1. The rewrite must not affect any information expressed in the user's original prompt; the rewritten prompt should use
   coherent natural language, avoid low-information redundant descriptions, and keep the rewritten prompt length as
   concise as possible.
2. Ensure consistency between input and output languages: Chinese input yields Chinese output, and English input yields
   English output. The rewritten token count should not exceed 512.
3. The rewritten description should further refine subject characteristics and aesthetic techniques appearing in the
   original prompt, such as lighting and textures.
4. If the original prompt does not specify an image style, ensure the rewritten prompt uses a **realistic photography
   style**. If the user specifies a style, retain the user's style.
5. When the original prompt requires reasoning to clarify user intent, use logical reasoning based on world knowledge
   to convert vague abstract descriptions into specific tangible objects (e.g., convert "the tallest animal" to "a
   giraffe").
6. When the original prompt requires text generation, please use double quotes to enclose the text part (e.g., `"50%
   OFF"`).
7. When the original prompt requires generating text-heavy scenes like webpages, logos, UIs, or posters, and no
   specific text content is specified, you need to infer appropriate text content and enclose it in double quotes. For
   example, if the user inputs: "A tourism flyer with a grassland theme," it should be rewritten as: "A tourism flyer
   with the image title 'Grassland'."
8. When negative words exist in the original prompt, ensure the rewritten prompt does not contain negative words. For
   example, "a lakeside without boats" should be rewritten such that the word "boat" does not appear at all.
9. Except for text content explicitly requested by the user, **adding any extra text content is prohibited**.
Here are examples of rewrites for different types of prompts: # Examples (Few-Shot Learning)
  1. User Input: An animal with nine lives.
    Rewrite Output: A cat bathed in soft sunlight, its fur soft and glossy. The background is a comfortable home
    environment with light from the window filtering through curtains, creating a warm light and shadow effect. The
    shot uses a medium distance perspective to highlight the cat's leisurely and stretched posture. Light cleverly hits
    the cat's face, emphasizing its spirited eyes and delicate whiskers, adding depth and affinity to the image.
  2. User Input: Create an anime-style tourism flyer with a grassland theme.
    Rewrite Output: In the lower right of the center, a short-haired girl sits sideways on a gray, irregularly shaped
    rock. She wears a white short-sleeved dress and brown flat shoes, holding a bunch of small white flowers in her
    left hand, smiling with her legs hanging naturally. The girl has dark brown shoulder-length hair with bangs
    covering her forehead, brown eyes, and a slightly open mouth. The rock surface has textures of varying depths. To
    the girl's left and front is lush grass, with long, yellow-green blades, some glowing golden in the sunlight. The
    grass extends into the distance, forming rolling green hills that fade in color as they recede. The sky occupies
    the upper half of the picture, pale blue dotted with a few fluffy white clouds. In the upper left corner, there is
    a line of text in italic, dark green font reading "Explore Nature's Peace". Colors are dominated by green, blue,
    and yellow, fluid lines, and distinct light and shadow contrast, creating a quiet and comfortable atmosphere.
  3. User Input: A Christmas sale poster with a red background, promoting a Buy 1 Get 1 Free milk tea offer.
    Rewrite Output: The poster features an overall red tone, embellished with white snowflake patterns on the top and
    left side. The upper right features a bunch of holly leaves with red berries and a pine cone. In the upper center,
    golden 3D text reads "Christmas Heartwarming Feedback" centered, along with red bold text "Buy 1 Get 1". Below, two
    transparent cups filled with bubble tea are placed side by side; the tea is light brown with dark brown pearls
    scattered at the bottom and middle. Below the cups, white snow piles up, decorated with pine branches, red berries,
    and pine cones. A blurry Christmas tree is faintly visible in the lower right corner. The image has high clarity,
    accurate text content, a unified design style, a prominent Christmas theme, and a reasonable layout, providing
    strong visual appeal.
  4. User Input: A woman indoors shot in natural light, smiling with arms crossed, showing a relaxed and confident
     posture.
    Rewrite Output: The image features a young Asian woman with long dark brown hair naturally falling over her
    shoulders, with some strands illuminated by light, showing a soft sheen. Her features are delicate, with long
    eyebrows, bright and spirited dark brown eyes looking directly at the camera, revealing peace and confidence. She
    has a high nose bridge, full lips with nude lipstick, and corners of the mouth slightly raised in a faint smile.
    Her skin is fair, with cheeks and collarbones illuminated by warm light, showing a healthy ruddiness. She wears a
    black spaghetti strap tank top revealing graceful collarbone lines, and a thin gold necklace with small beads and
    metal bars glinting in the light. Her outer layer is a beige knitted cardigan, soft in texture with visible
    knitting patterns on the sleeves. Her arms are crossed over her chest, hands covered by the cardigan sleeves, in a
    relaxed posture. The background is a pure dark brown without extra decoration, making the figure the absolute
    focus. The figure is located in the center of the frame. Light enters from the upper right, creating bright spots
    on her left cheek, neck, and collarbone, while the right side is slightly shadowed, creating a three-dimensional
    and soft tone. Image details are clear, showcasing skin texture, hair, and clothing materials well. Colors are
    dominated by warm tones, with the combination of beige and dark brown creating a warm and comfortable atmosphere.
    The overall style is natural, elegant, and artistic.
  5. User Input: Create a series of images showing the growth process of an apple from seed to fruit. The series should
     include four stages: 1. Sowing, 2. Seedling growth, 3. Plant maturity, 4. Fruit harvesting.
    Rewrite Output: A 4-panel exquisite illustration depicting the growth process of an apple, capturing each stage
    precisely and clearly. 1. "Sowing": A close-up shot of a hand gently placing a small apple seed into fertile dark
    soil, with visible soil texture and the seed's smooth surface. The background is a soft-focus garden dotted with
    green leaves and sunlight filtering through. 2. "Seedling Growth": A young apple sapling breaks through the soil,
    stretching tender green leaves toward the sky. The scene is set in a vibrant garden illuminated by warm golden
    light, highlighting the seedling's delicate structure. 3. "Plant Maturity": A mature apple tree, lush with branches
    and leaves, covered in tender green foliage and developing small apples. The background is a vibrant orchard under
    a clear blue sky, with dappled sunlight creating a peaceful atmosphere. 4. "Fruit Harvesting": A hand reaches into
    the tree to pick a ripe red apple, its smooth skin glistening in the sun. The scene shows the abundance of the
    orchard, with baskets of apples in the background, giving a sense of fulfillment. Each illustration uses a
    realistic style, focusing on details and harmonious colors to showcase the natural beauty and development of the
    apple's life cycle.
  6. User Input: If 1 represents red, 2 represents green, 3 represents purple, and 4 represents yellow, please generate
     a four-color rainbow based on this rule. The color order from top to bottom is 3142.
    Rewrite Output: The image consists of four horizontally arranged colored stripes, ordered from top to bottom as
    purple, red, yellow, and green. A white number is centered on each stripe. The top purple stripe features the
    number "3", the red stripe below it has the number "1", the yellow stripe further down has the number "4", and the
    bottom green stripe has the number "2". All numbers use a sans-serif font in pure white, forming a sharp contrast
    with the background colors to ensure good readability. The stripes have high color saturation and a slight texture.
    The overall layout is simple and clear, with distinct visual effects and no extra decorative elements, emphasizing
    the numerical information. The image is high definition, with accurate colors and a consistent style, offering
    strong visual appeal.
  7. User Input: A stone tablet carved with "Guan Guan Ju Jiu, On the River Isle", natural light, background is a
     Chinese garden.
    Rewrite Output: An ancient stone tablet carved with "Guan Guan Ju Jiu, On the River Isle", the surface covered with
    traces of time, the writing clear and deep. Natural light falls from above, softly illuminating every detail of the
    stone tablet and enhancing its sense of history. The background is an elegant Chinese garden featuring lush bamboo
    forests, winding paths, and quiet pools, creating a serene and distant atmosphere. The overall picture uses a
    realistic style with rich details and natural light and shadow effects, highlighting the cultural heritage of the
    stone tablet and the classical beauty of the garden.
# Output Format Please directly output the rewritten and optimized Prompt content. Do not include any explanatory
language or JSON formatting, and do not add opening or closing quotes yourself."""

SYSTEM_PROMPT_ZH = """
你是一名文生图模型的prompt
engineering专家。由于文生图模型对用户prompt的理解能力有限，你需要识别用户输入的核心主题和意图，并通过优化改写提升模型的理解准确性和生成质量。改写必须严格保留用户原始prompt的所有信息，不得删减或曲解任何细节。
具体要求如下：
1. 改写不能影响用户原始prompt里表达的任何信息，改写后的prompt应该使用连贯的自然语言表达,不要出现低信息量的冗余描述，尽可能保持改写后prompt长度精简。
2. 请确保输入和输出的语言类型一致，中文输入中文输出，英文输入英文输出，改写后的token数量不要超过512个;
3. 改写后的描述应当进一步完善原始prompt中出现的主体特征、美学技巧，如打光、纹理等；
4. 如果原始prompt没有指定图片风格时，确保改写后的prompt使用真实摄影风格，如果用户指定了图片风格，则保留用户风格；
5. 当原始prompt需要推理才能明确用户意图时，根据世界知识进行适当逻辑推理，将模糊抽象描述转化为具体指向事物（例：将"最高的动物"转化为"一头长颈鹿"）。
6. 当原始prompt需要生成文字时，请使用双引号圈定文字部分，例：`"限时5折"`）。
7. 当原始prompt需要生成网页、logo、ui、海报等文字场景时，且没有指定具体的文字内容时，需要推断出合适的文字内容，并使用双引号圈定，如用户输入：一个旅游宣传单，以草原为主题。应该改写成：一个旅游宣传单，图片标题为“草原”。
8. 当原始prompt中存在否定词时，需要确保改写后的prompt不存在否定词，如没有船的湖边，改写后的prompt不能出现船这个词汇。
9. 除非用户指定生成品牌logo，否则不要增加额外的品牌logo.
10. 除了用户明确要求书写的文字内容外，**禁止增加任何额外的文字内容**。
以下是针对不同类型prompt改写的示例：

# Examples (Few-Shot Learning)
  1. 用户输入: 九条命的动物。
    改写输出:
    一只猫，被柔和的阳光笼罩着，毛发柔软而富有光泽。背景是一个舒适的家居环境，窗外的光线透过窗帘，形成温馨的光影效果。镜头采用中距离视角，突出猫悠闲舒展的姿态。光线巧妙地打在猫的脸部，强调它灵动的眼睛和精致的胡须，增加画面的层次感与亲和力。
  2. 用户输入: 制作一个动画风格的旅游宣传单，以草原为主题。
    改写输出:
    画面中央偏右下角，一个短发女孩侧身坐在灰色的不规则形状岩石上，她穿着白色短袖连衣裙和棕色平底鞋，左手拿着一束白色小花，面带微笑，双腿自然垂下。女孩的头发为深棕色，齐肩短发，刘海覆盖额头，眼睛呈棕色，嘴巴微张。岩石表面有深浅不一的纹理。女孩的左侧和前方是茂盛的草地，草叶细长，呈黄绿色，部分草叶在阳光下泛着金色的光芒，仿佛被阳光照亮。草地向远处延伸，形成连绵起伏的绿色山丘，山丘的颜色由近及远逐渐变浅。天空占据了画面的上半部分，呈淡蓝色，点缀着几朵白色蓬松的云彩。画面的左上角有一行文字，文字内容是斜体、深绿色的“Explore
    Nature's Peace”。色彩以绿色、蓝色和黄色为主，线条流畅，光影明暗对比明显，营造出一种宁静、舒适的氛围。
  3. 用户输入: 一张以红色为背景的圣诞节促销海报，主要宣传奶茶买一送一的优惠活动。
    改写输出: 海报整体呈现红色调，上方和左侧点缀着白色雪花图案，右上方有一束冬青叶和红色浆果，以及一个松果。海报中央偏上位置，金色立体字样“圣诞节
    暖心回馈”居中排列，和红色粗体字“买1送1”。海报下方，两个装满珍珠奶茶的透明杯子并排摆放，杯中奶茶呈浅棕色，底部和中间散布着深棕色珍珠。杯子下方，堆积着白色雪花，雪花上装饰着松枝、红色浆果和松果。右下角隐约可见一棵模糊的圣诞树。图片清晰度高，文字内容准确，整体设计风格统一，圣诞主题突出，排版布局合理，具有较强的视觉吸引力。
  4. 用户输入: 一位女性在室内以自然光线拍摄，她面带微笑，双臂交叉，展现出轻松自信的姿态。
    改写输出:
    画面中是一位年轻的亚洲女性，她拥有深棕色的长发，发丝自然地垂落在双肩，部分发丝被光线照亮，呈现出柔和的光泽。她的五官精致，眉毛修长，眼睛明亮有神，瞳孔呈深棕色，眼神直视镜头，流露出平和与自信。鼻梁挺拔，嘴唇丰满，涂有裸色系唇膏，嘴角微微上扬，展现出浅浅的微笑。她的肤色白皙，脸颊和锁骨处被暖色调的光线照亮，呈现出健康的红润感。她穿着一件黑色的细吊带背心，肩带纤细，露出优美的锁骨线条。脖颈上佩戴着一条金色的细项链，项链由小珠子和几个细长的金属条组成，在光线下闪烁着光泽。她的外搭是一件米黄色的针织开衫，材质柔软，袖子部分有明显的针织纹理。她双臂交叉在胸前，双手被开衫的袖子覆盖，姿态放松。背景是纯粹的深棕色，没有多余的装饰，使得人物成为画面的绝对焦点。人物位于画面中央。光线从画面的右上方射入，在人物的左侧脸颊、脖颈和锁骨处形成明亮的光斑，右侧则略显阴影，营造出立体感和柔和的影调。图像细节清晰，人物的皮肤纹理、发丝以及衣物材质都得到了很好的展现。色彩以暖色调为主，米黄色和深棕色的搭配营造出温馨舒适的氛围。整体呈现出一种自然、优雅且富有亲和力的艺术风格。
  5. 用户输入：创作一系列图片，展现苹果从种子到结果的生长过程。该系列图片应包含以下四个阶段：1. 播种，2. 幼苗生长，3. 植物成熟，4. 果实采摘。
    改写输出：一个4宫格的精美插图，描绘苹果的生长过程，精确清晰地捕捉每个阶段。1.“播种”：特写镜头，一只手轻轻地将一颗小小的苹果种子放入肥沃的深色土壤中，土壤的纹理和种子光滑的表面清晰可见。背景是花园的柔焦画面，点缀着绿色的树叶和透过树叶洒下的阳光。2.“幼苗生长”：一棵幼小的苹果树苗破土而出，嫩绿的叶子向天空舒展。场景设定在一个生机勃勃的花园中，温暖的金光照亮了它。幼苗的纤细结构。3.“植物的成熟”：一棵成熟的苹果树，枝繁叶茂，挂满了嫩绿的叶子和正在萌发的小苹果。背景是一片生机勃勃的果园，湛蓝的天空下，斑驳的阳光营造出宁静祥和的氛围。4.“采摘果实”：一只手伸向树上，摘下一个成熟的红苹果，苹果光滑的果皮在阳光下闪闪发光。画面展现了果园的丰收景象，背景中摆放着一篮篮的苹果，给人一种圆满满足的感觉。每幅插图都采用写实风格，注重细节，色彩和谐，展现了苹果生命周期的自然之美和发展过程。
  6. 用户输入： 如果1代表红色，2代表绿色，3代表紫色，4代表黄色，请按照此规则生成四色彩虹。它的颜色顺序从上到下是3142
    改写输出：图片由四个水平排列的彩色条纹组成，从上到下依次为紫色、红色、黄色和绿色。每个条纹上都居中放置一个白色数字。最上方的紫色条纹上是数字“3”，其下方红色条纹上是数字“1”，再下方黄色条纹上是数字“4”，最下方的绿色条纹上是数字“2”。所有数字均采用无衬线字体，颜色为纯白色，与背景色形成鲜明对比，确保了良好的可读性。条纹的颜色饱和度高，且带有轻微的纹理感，整体排版简洁明了，视觉效果清晰，没有多余的装饰元素，强调了数字信息本身。图片整体清晰度高，色彩准确，风格一致，具有较强的视觉吸引力。
  7. 用户输入：石碑上刻着“关关雎鸠，在河之洲”，自然光照，背景是中式园林
    改写输出：一块古老的石碑上刻着“关关雎鸠，在河之洲”，石碑表面布满岁月的痕迹，字迹清晰而深刻。自然光线从上方洒下，柔和地照亮石碑的每一个细节，增强了其历史感。背景是一座典雅的中式园林，园林中有翠绿的竹林、蜿蜒的小径和静谧的水池，营造出一种宁静而悠远的氛围。整体画面采用写实风格，细节丰富，光影效果自然，突出了石碑的文化底蕴和园林的古典美。
# 输出格式 请直接输出改写优化后的 Prompt 内容，不要包含任何解释性语言或 JSON 格式，不要自行添加开头或结尾的引号。
"""


def get_prompt_language(prompt: str) -> str:
    pattern = re.compile(r"[\u4e00-\u9fff]")
    return "zh" if bool(pattern.search(prompt)) else "en"


def split_quotation(
    prompt: str, quote_pairs: list[tuple[str, str]] | None = None
) -> list[tuple[str, bool]]:
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote = []
    for index, word_src in enumerate(set(matches_word_internal_quote_pattern)):
        word_tgt = "longcat_$##$_longcat" * (index + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])
    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
    pattern = "|".join(
        re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2)
        for q1, q2 in quote_pairs
    )
    parts = re.split(f"({pattern})", prompt)
    result = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if part:
            result.append((part, bool(re.match(pattern, part))))
    return result


def calculate_shift(image_seq_len: int, config: Mapping[str, Any] | None = None) -> float:
    config = config or {}
    base_seq_len = int(config.get("base_image_seq_len", 256))
    max_seq_len = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 1.15))
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "longcat-image"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _special_token_ids(directory: Path) -> set[int]:
    data = read_json(directory / "tokenizer.json")
    return {
        int(token["id"])
        for token in data.get("added_tokens", [])
        if isinstance(token, dict) and token.get("special", False)
    }


@dataclass
class LongCatImageComponents:
    tokenizer: Qwen2Tokenizer
    text_processor: Qwen2Tokenizer
    text_processor_special_ids: set[int]
    text_encoder: Qwen25VLTextModel
    generation_config: dict[str, Any]
    transformer: LongCatImageTransformer2DModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> LongCatImageComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a LongCat-Image snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_processor=Qwen2Tokenizer(root / "text_processor"),
                text_processor_special_ids=_special_token_ids(root / "text_processor"),
                text_encoder=Qwen25VLTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                generation_config=read_json(root / "text_encoder/generation_config.json"),
                transformer=LongCatImageTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler/scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _decode_tokens(tokenizer: Qwen2Tokenizer, token_ids: Tensor, special_ids: set[int]) -> str:
    decoder = {value: key for key, value in tokenizer.encoder.items()}
    decoder.update({value: key for key, value in tokenizer._raw_added.items()})
    decoder.update({value: key for key, value in tokenizer._normalized_added.items()})
    byte_decoder = {value: key for key, value in bytes_to_unicode().items()}
    pieces: list[str] = []
    encoded = bytearray()
    for token_id in token_ids.tolist():
        if token_id in special_ids:
            continue
        token = decoder[int(token_id)]
        if all(character in byte_decoder for character in token):
            encoded.extend(byte_decoder[character] for character in token)
        else:
            if encoded:
                pieces.append(encoded.decode("utf-8", errors="replace"))
                encoded.clear()
            pieces.append(token)
    if encoded:
        pieces.append(encoded.decode("utf-8", errors="replace"))
    return "".join(pieces)


@torch.inference_mode()
def _qwen2_vl_chat_prompt(text: str) -> str:
    """The text processor's Qwen2-VL chat template for one text-only user message.

    The template adds a default system turn when the conversation does not start with one, then
    the generation prompt.
    """
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )


def rewrite_prompt(components: LongCatImageComponents, prompt: str, device: torch.device) -> str:
    if get_prompt_language(prompt) == "zh":
        question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{prompt}\n改写后的prompt为："
    else:
        question = SYSTEM_PROMPT_EN + f"\nUser Input: {prompt}\nRewritten prompt:"
    rendered = _qwen2_vl_chat_prompt(question)
    ids = components.text_processor.tokenize(rendered)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated = components.text_encoder.generate(
        input_ids,
        attention_mask,
        max_new_tokens=TOKENIZER_MAX_LENGTH,
        generation_config=components.generation_config,
    )
    return _decode_tokens(
        components.text_processor,
        generated[0, input_ids.shape[1] :],
        components.text_processor_special_ids,
    )


@torch.inference_mode()
def encode_prompt(
    components: LongCatImageComponents,
    prompt: str,
    device: torch.device,
    num_images: int = 1,
) -> tuple[Tensor, Tensor]:
    all_tokens: list[int] = []
    for clean_prompt_sub, matched in split_quotation(prompt):
        if matched:
            for sub_word in clean_prompt_sub:
                all_tokens.extend(components.tokenizer.tokenize(sub_word))
        else:
            all_tokens.extend(components.tokenizer.tokenize(clean_prompt_sub))
    all_tokens = all_tokens[:TOKENIZER_MAX_LENGTH]
    padding = TOKENIZER_MAX_LENGTH - len(all_tokens)
    content_ids = torch.tensor(
        [all_tokens + [components.tokenizer.pad_id] * padding], dtype=torch.long
    )
    content_mask = torch.tensor([[True] * len(all_tokens) + [False] * padding], dtype=torch.bool)
    prefix = components.tokenizer.tokenize(PROMPT_TEMPLATE_ENCODE_PREFIX)
    suffix = components.tokenizer.tokenize(PROMPT_TEMPLATE_ENCODE_SUFFIX)
    input_ids = torch.cat(
        (
            torch.tensor([prefix], dtype=torch.long),
            content_ids,
            torch.tensor([suffix], dtype=torch.long),
        ),
        dim=-1,
    ).to(device)
    attention_mask = torch.cat(
        (
            torch.ones(1, len(prefix), dtype=torch.bool),
            content_mask,
            torch.ones(1, len(suffix), dtype=torch.bool),
        ),
        dim=-1,
    ).to(device)
    prompt_embeds = components.text_encoder(input_ids, attention_mask)[-1]
    prompt_embeds = prompt_embeds[:, len(prefix) : -len(suffix), :].detach()
    _, sequence_length, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images, 1)
    prompt_embeds = prompt_embeds.view(num_images, sequence_length, -1)
    text_ids = torch.zeros(sequence_length, 3, device=device)
    positions = torch.arange(sequence_length, device=device)
    text_ids[:, 1] = positions
    text_ids[:, 2] = positions
    return prompt_embeds, text_ids


def _prepare_latent_image_ids(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    ids = torch.zeros(height, width, 3)
    ids[..., 0] = 1
    ids[..., 1] += torch.arange(height)[:, None] + TOKENIZER_MAX_LENGTH
    ids[..., 2] += torch.arange(width)[None, :] + TOKENIZER_MAX_LENGTH
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def _pack_latents(
    latents: Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> Tensor:
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def _unpack_latents(latents: Tensor, height: int, width: int, vae_scale_factor: int) -> Tensor:
    batch_size, _, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height, width)


def _sigmas(steps: int) -> list[float]:
    if steps == 1:
        return [1.0]
    stop = 1.0 / steps
    step = (stop - 1.0) / (steps - 1)
    values = [1.0 + index * step for index in range(steps)]
    values[-1] = stop
    return values


def _apply_cfg(
    noise_pred_text: Tensor,
    noise_pred_uncond: Tensor,
    guidance: float,
    enable_cfg_renorm: bool,
    cfg_renorm_min: float,
) -> Tensor:
    noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
    if enable_cfg_renorm:
        cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        noise_pred = noise_pred * scale
    return noise_pred


@torch.inference_mode()
def denoise(
    components: LongCatImageComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    latents: Tensor,
    enable_cfg_renorm: bool,
    cfg_renorm_min: float,
    enable_prompt_rewrite: bool,
) -> Tensor:
    dtype = cast(torch.dtype, components.transformer.x_embedder.weight.dtype)
    if enable_prompt_rewrite:
        prompt = rewrite_prompt(components, prompt, latents.device)
    negative_prompt = "" if negative_prompt is None else negative_prompt
    prompt_embeds, text_ids = encode_prompt(components, prompt, latents.device)
    prompt_embeds = prompt_embeds.to(dtype)
    do_cfg = guidance > 1
    negative_prompt_embeds = negative_text_ids = None
    if do_cfg:
        negative_prompt_embeds, negative_text_ids = encode_prompt(
            components, negative_prompt, latents.device
        )
        negative_prompt_embeds = negative_prompt_embeds.to(dtype)
    batch_size, channels, latent_height, latent_width = latents.shape
    latents = _pack_latents(latents, batch_size, channels, latent_height, latent_width)
    latent_image_ids = _prepare_latent_image_ids(
        latent_height // 2, latent_width // 2, latents.device, latents.dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(
        steps,
        sigmas=_sigmas(steps),
        mu=calculate_shift(latents.shape[1], components.scheduler_config),
    )
    scheduler.set_begin_index(0)
    for timestep in scheduler.timesteps:
        model_timestep = timestep.expand(batch_size).to(latents.dtype) / 1000
        noise_pred_text = components.transformer(
            latents.to(dtype),
            prompt_embeds,
            model_timestep,
            latent_image_ids,
            text_ids,
        )
        if do_cfg:
            assert negative_prompt_embeds is not None and negative_text_ids is not None
            noise_pred_uncond = components.transformer(
                latents.to(dtype),
                negative_prompt_embeds,
                model_timestep,
                latent_image_ids,
                negative_text_ids,
            )
            noise_pred = _apply_cfg(
                noise_pred_text,
                noise_pred_uncond,
                guidance,
                enable_cfg_renorm,
                cfg_renorm_min,
            )
        else:
            noise_pred = noise_pred_text
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor, height: int, width: int) -> Tensor:
    vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)
    latents = _unpack_latents(latents, height, width, vae_scale_factor)
    latents = latents / vae.scaling_factor + (vae.shift_factor or 0.0)
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class LongCatImageBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source = Path(model)
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: LongCatImageComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int, int]:
        packed_channels = 64
        vae_scale = 8
        transformer_config = self.source / "transformer/config.json"
        vae_config = self.source / "vae/config.json"
        if transformer_config.is_file():
            packed_channels = int(read_json(transformer_config).get("in_channels", 64))
        if vae_config.is_file():
            blocks = read_json(vae_config).get("block_out_channels", [1, 1, 1, 1])
            vae_scale = 2 ** (len(blocks) - 1)
        return (
            1,
            packed_channels // 4,
            options["height"] // vae_scale,
            options["width"] // vae_scale,
        )

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown LongCat-Image options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("enable_cfg_renorm", "enable_prompt_rewrite"):
            if type(o[name]) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in ("steps", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if o[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        cfg_min = o["cfg_renorm_min"]
        if (
            isinstance(cfg_min, bool)
            or not isinstance(cfg_min, int | float)
            or not math.isfinite(cfg_min)
            or not 0 <= cfg_min <= 1
        ):
            raise ValueError("cfg_renorm_min must be a finite number in [0, 1]")
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = self._latent_shape(o)
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        o["latent_shape"] = shape
        return o

    def _load(self) -> None:
        self.components = LongCatImageComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the LongCat-Image architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            latents = torch.randn(
                o["latent_shape"],
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            width=o["width"],
            height=o["height"],
            latents=latents.to(torch.float32).to(self.device),
            enable_cfg_renorm=o["enable_cfg_renorm"],
            cfg_renorm_min=float(o["cfg_renorm_min"]),
            enable_prompt_rewrite=o["enable_prompt_rewrite"],
        )
        pixels = decode_latents(self.components.vae, final, o["height"], o["width"])
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "negative_prompt": o["negative_prompt"],
            "enable_cfg_renorm": o["enable_cfg_renorm"],
            "cfg_renorm_min": float(o["cfg_renorm_min"]),
            "enable_prompt_rewrite": o["enable_prompt_rewrite"],
            "width": o["width"],
            "height": o["height"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
