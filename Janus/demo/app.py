import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import numpy as np

# Detect device (CPU fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model config and force eager attention
model_path = "deepseek-ai/Janus-1.3B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = "eager"

# Load the multimodal model
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path,
    language_config=language_config,
    trust_remote_code=True
)
# Set appropriate data type and move to the correct device
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
vl_gpt = vl_gpt.to(dtype).to(device).eval()

# Prepare processor and tokenizer
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# CPU-only compatibility: define no-op CUDA cache clear if CPU
def clear_cuda_cache():
    if device.type == "cuda":
        torch.cuda.empty_cache()

@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    clear_cuda_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    conversation = [
        {"role": "User", "content": f"<image_placeholder>\n{question}", "images": [image]},
        {"role": "Assistant", "content": ""},
    ]
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(device, dtype=dtype)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

def generate(input_ids, width, height, temperature, parallel_size, cfg_weight):
    clear_cuda_cache()
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int, device=device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, 576), dtype=torch.int, device=device)

    pkv = None
    for i in range(576):
        outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond, logit_uncond = logits[0::2], logits[1::2]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(-1)
        next_pair = torch.cat([next_token, next_token], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_pair)
        inputs_embeds = img_embeds.unsqueeze(1)

    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(torch.int),
        shape=[parallel_size, 8, width // 16, height // 16]
    )
    return generated_tokens, patches

def unpack(dec, width, height, parallel_size=5):
    arr = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    arr = np.clip((arr + 1)/2 * 255, 0, 255).astype(np.uint8)
    return np.stack([Image.fromarray(arr[i]) for i in range(arr.shape[0])])

@torch.inference_mode()
def generate_image(prompt, seed, cfg_weight):
    clear_cuda_cache()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

    width = height = 384
    parallel_size = 5
    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}],
        sft_format=vl_chat_processor.sft_format,
        system_prompt=""
    ) + vl_chat_processor.image_start_tag
    input_ids = torch.LongTensor(tokenizer.encode(text)).to(device)
    _, patches = generate(input_ids, width, height, temperature=1, parallel_size=parallel_size, cfg_weight=cfg_weight)
    images = unpack(patches, width, height, parallel_size)
    # upscale for display
    return [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(parallel_size)]

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Understanding")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(0, 1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(0, 1, value=0.1, step=0.05, label="temperature")
    gr.Button("Chat").click(multimodal_understanding, inputs=[image_input, question_input, seed_input, top_p, temperature], outputs=gr.Textbox(label="Response"))

    gr.Markdown("# Text-to-Image Generation")
    prompt_input = gr.Textbox(label="Prompt")
    cfg_input = gr.Slider(1, 10, value=5, step=0.5, label="CFG Weight")
    gr.Button("Generate Images").click(generate_image, inputs=[prompt_input, seed_input, cfg_input], outputs=gr.Gallery(label="Generated Images", columns=2, rows=2))
demo.launch(share=True)
