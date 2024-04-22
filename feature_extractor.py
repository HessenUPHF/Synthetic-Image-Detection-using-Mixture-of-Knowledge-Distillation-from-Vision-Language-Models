import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration,VisionEncoderDecoderModel, AutoFeatureExtractor,AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.utils.data import Dataset
import PIL.Image as Image
from torch.utils.data import  DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#BLIP2 Model Loading stage with PEFT 
config = PeftConfig.from_pretrained('./BLIP2/ldmFineTune')
model_blip2 = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path,  device_map="auto")  #load_in_8bit=True,
model_blip2 = PeftModel.from_pretrained(model_blip2, './BLIP2/ldmFineTune')
model_blip2 = model_blip2.to(device)
processor_blip2 = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")


# VITGPT2 Model Loading.....
# Model initialization
image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "gpt2"
model = VisionEncoderDecoderModel.from_pretrained('./Vit-GPT2/ldmFineTune')
model = model.to(device)

# image feature extractor,  text tokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id






