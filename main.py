import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import gradio as gr
from PIL import Image
import numpy as np
import soundfile as sf
import os
from transformers import SpeechT5Processor

# Load the models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")
speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
speech_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# Load speaker embeddings
speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(speaker_embeddings_dataset[0]["xvector"]).unsqueeze(0)

# Frame extraction function (optimized)
def extract_frames(video_path, frame_rate=1, max_frames=100, target_size=(224, 224)):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    success, image = vidcap.read()
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    while success and len(frames) < max_frames:
        if count % int(fps * frame_rate) == 0:
            # Resize image
            image = cv2.resize(image, target_size)
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames

# Image Captioning (BLIP model)
def generate_captions(frames):
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    inputs = blip_processor(images=images, return_tensors="pt", padding=True)
    outputs = blip_model.generate(**inputs, max_new_tokens=50)  # Limit token length for each caption
    captions = [blip_processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# Summarization (T5 model)
def summarize_text(text):
    input_ids = t5_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def text_to_speech(text, output_file="output.wav"):
    # Explicitly tell the processor that the input is text
    inputs = speech_processor(text=text, return_tensors="pt").input_ids
    
    # Generate speech using the tokenized input
    audio_output = speech_model.generate_speech(inputs, speaker_embeddings=speaker_embeddings, vocoder=speech_vocoder)
    
    # Save the audio to a file
    sf.write(output_file, audio_output, 22050)
    
    return output_file


# Full video processing pipeline
def process_video(video_path):
    frames = extract_frames(video_path, frame_rate=1)  # Extract one frame per second
    captions = generate_captions(frames)  # Generate captions for each frame
    full_text = " ".join(captions)  # Combine captions into a single string
    summary = summarize_text(full_text)  # Summarize the captions
    audio_file = text_to_speech(summary)  # Convert the summary to speech
    return summary, audio_file

# Gradio interface
def interface(video):
    summary, audio_file = process_video(video)
    return summary, audio_file

gr.Interface(
    fn=interface,
    inputs=gr.Video(label="Upload a Video"),
    outputs=[
        gr.Textbox(label="Video Summary"),
        gr.Audio(label="Generated Audio")
    ],
    title="Video Description and Summarization"
).launch()
