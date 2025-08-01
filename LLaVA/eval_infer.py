import re
import torch
from PIL import Image
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)

def eval_model_single_image(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    query: str,
    conv_mode: str = None,
    temperature: float = 0.2,
    top_p: float = None,
    num_beams: int = 1,
    max_new_tokens: int = 512
):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if model.config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        if model.config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query

    # conversation 模板选择
    if conv_mode is None:
        conv_mode = "llava_v1" if hasattr(model.config, "mm_use_im_start_end") else "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = process_images(
        [image], image_processor, model.config
    ).to(model.device, dtype=torch.float16)
    image_sizes = [image.size]

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs
