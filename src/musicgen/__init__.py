import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = " Create a modern Latin reggaeton track with a catchy rhythm, deep punchy bass, infectious dembow drums, crisp percussion, and vibrant urban energy. Blend contemporary reggaeton with subtle Latin pop and Caribbean influences. Use a danceable mid-tempo groove, atmospheric synths, melodic hooks, and smooth transitions between sections. The track should feel sensual, energetic, confident, and addictive, with a powerful club-ready beat and a polished, professional studio sound. Include a memorable chorus, rhythmic variations, and a strong beat drop. Make it suitable for parties, nightlife, dancing, and summer vibes. High-quality production, rich low-end, clean percussion, modern sound design, and a radio-ready mix."

print("Cargando Musicgen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model= model.to(DEVICE)
model.eval()

print("Preparando promt ...")

inputs = processor(text=[PROMPT], padding= True,return_tensors="pt").to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio= model.generate(**inputs, do_sample=True, guidance_scale= 3.0, max_new_tokens=1500)

sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")
    
