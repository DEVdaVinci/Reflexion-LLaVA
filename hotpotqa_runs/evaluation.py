from diffusers import DiffusionPipeline

#similarity metric
import torch
import clip
from PIL import Image
import torch.nn as nn

class StableDiffusionEval_test:
    def __init__(self):
        #self.modelType = modelType
        self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipeline.to("cuda")
        #This is necessary for the similarity metric
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    def evaluatePrompt(self, prompt, originalImage):
        generatedImage = self.generateImage(prompt)
        similarityScore = self.evaluateGeneratedImage(image_og = originalImage, image_generated = generatedImage)
        return similarityScore
    def generateImage(self, prompt: str):
        generatedImage = self.pipeline(prompt).images[0]
        return generatedImage
    def evaluateGeneratedImage(self, image_og, image_generated):
        cos = torch.nn.CosineSimilarity(dim=0)

        image_og_preprocess = self.preprocess(image_og).unsqueeze(0).to(self.device)
        image_og_features = self.model.encode_image( image_og_preprocess)

        image_generated_preprocess = self.preprocess(image_generated).unsqueeze(0).to(self.device)
        image_generated_features = self.model.encode_image( image_generated_preprocess)

        similarity = cos(image_og_features[0],image_generated_features[0]).item()
        similarity = (similarity+1)/2
        '''
            if doPrint:
                print("Image similarity", similarity)
        '''
        return similarity
