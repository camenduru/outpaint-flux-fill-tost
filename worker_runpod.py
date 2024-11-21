import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import  nodes_flux, nodes_differential_diffusion

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
DifferentialDiffusion = nodes_differential_diffusion.NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()

ImagePadForOutpaint =  NODE_CLASS_MAPPINGS["ImagePadForOutpaint"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
InpaintModelConditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("flux1-fill-dev.safetensors", "default")[0]
    unet = DifferentialDiffusion.apply(unet)[0]
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    vae = VAELoader.load_vae("ae.sft")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    left = values['left']
    right = values['right']
    top = values['top']
    bottom = values['bottom']
    feathering = values['feathering']
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    guidance = values['guidance']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    conditioning_positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    conditioning_negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    image = LoadImage.load_image(input_image)[0]
    pixels, mask = ImagePadForOutpaint.expand_image(image, left, top, right, bottom, feathering=feathering)
    conditioning_positive = FluxGuidance.append(conditioning_positive, guidance)[0]
    positive, negative, latent_image = InpaintModelConditioning.encode(conditioning_positive, conditioning_negative, pixels, vae, mask, noise_mask=False)
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0)[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"/content/outpaint-flux-fill-{seed}-tost.png")

    result = f"/content/outpaint-flux-fill-{seed}-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})