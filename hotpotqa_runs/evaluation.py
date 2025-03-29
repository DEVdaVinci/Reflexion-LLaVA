from diffusers import DiffusionPipeline

#similarity metric
import torch
import clip
from PIL import Image #not used
import torch.nn as nn
import torchvision.transforms as transforms
import lpips

class StableDiffusionEval_test:
    def __init__(self):
        #self.modelType = modelType
        #self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        #self.pipeline.to("cuda")
        #This is necessary for the similarity metric
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    def evaluatePrompt(self, prompt, originalImage, doPrint = False):
        generatedImage = self.generateImage(prompt)
        similarityScore = self.evaluateGeneratedImage(image_og = originalImage, image_generated = generatedImage, doPrint = doPrint)
        return similarityScore, generatedImage
    def generateImage(self, prompt: str):
        self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipeline.to("cuda")
        
        generatedImage = self.pipeline(prompt).images[0]
        return generatedImage
    def evaluateGeneratedImage(self, image_og, image_generated, doPrint = False):
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''cos = torch.nn.CosineSimilarity(dim=0)

        image_og_preprocess = self.preprocess(image_og).unsqueeze(0).to(self.device)
        image_og_features = self.model.encode_image( image_og_preprocess)

        image_generated_preprocess = self.preprocess(image_generated).unsqueeze(0).to(self.device)
        image_generated_features = self.model.encode_image( image_generated_preprocess)

        similarity = cos(image_og_features[0],image_generated_features[0]).item()
        similarity = (similarity+1)/2
        '''
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        image_og_features = self.get_clip_embedding(image_og)
        image_generated_features = self.get_clip_embedding(image_generated)



        cosine_similarity_CLIP = torch.nn.functional.cosine_similarity(image_og_features, image_generated_features).item()
        if doPrint:
            print(f"Similarity Score (CLIP): {cosine_similarity_CLIP:.4f}")




        # Load images
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        img1 = transform(image_og).unsqueeze(0)
        img2 = transform(image_generated).unsqueeze(0)



        # Load LPIPS model
        lpips_model_alex = lpips.LPIPS(net='alex')  # Use 'vgg' for VGG features
        # Compute LPIPS distance (lower = more similar)
        lpips_score_alex = lpips_model_alex(img1, img2).item()
        lpips_simScore_alex = 1 - lpips_score_alex  # Convert to similarity score (higher = more similar)
        if doPrint:
            print(f"LPIPS Perceptual Similarity Score (Alex): {lpips_simScore_alex}")

        # Load LPIPS model
        loss_fn_vgg = lpips.LPIPS(net='vgg')  # 'alex', 'vgg', or 'squeeze' network
        # Compute perceptual similarity
        lpips_score_vgg = loss_fn_vgg(img1, img2).item()
        lpips_simScore_vgg = 1 - lpips_score_vgg  # Convert to similarity score (higher = more similar)
        if doPrint:
            print(f"LPIPS Perceptual Similarity (VGG): {lpips_simScore_vgg}")

        # Load LPIPS model
        loss_fn_squeeze = lpips.LPIPS(net='squeeze')  # 'alex', 'vgg', or 'squeeze' network
        # Compute perceptual similarity
        lpips_score_squeeze = loss_fn_squeeze(img1, img2).item()
        lpips_simScore_squeeze = 1 - lpips_score_squeeze  # Convert to similarity score (higher = more similar)
        if doPrint:
            print(f"LPIPS Perceptual Similarity (Squeeze): {lpips_simScore_squeeze}")


        #What are sim scores?
            #These are essentially normalized scores
            #you have to subtract the scores from 1 
            #Initially high numbers = different images & low numbers = similar
            #After you "normalize" high numbers = simialr & llow numbers = different



        #This is the one you should go with for now

        #Go with Alex. Its hard to tell a difinative winner but lets go with this

        simScore_alex = lpips_simScore_alex * cosine_similarity_CLIP
        simScore_vgg = lpips_simScore_vgg * cosine_similarity_CLIP
        simScore_squeeze = lpips_simScore_squeeze * cosine_similarity_CLIP

        
        avgSimScore_alex = (lpips_simScore_alex + cosine_similarity_CLIP)/2
        avgSimScore_vgg = (lpips_simScore_vgg + cosine_similarity_CLIP)/2
        avgSimScore_squeeze = (lpips_simScore_squeeze + cosine_similarity_CLIP)/2
        if doPrint:
            print(f"Average Similarity Score (Alex): {avgSimScore_alex}")
            print(f"Average Similarity Score (VGG): {avgSimScore_vgg}")
            print(f"Average Similarity Score (Squeeze): {avgSimScore_squeeze}")



        avgScore = avgSimScore_alex + avgSimScore_vgg + avgSimScore_squeeze
        avgScore = avgScore/3
        if doPrint:
            print(f"avgScore = {avgSimScore_alex}")








        #MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
        simpleWeightedAvgSimScore2_alex = self.simpleWeightedAvg_modified(lpips_simScore_alex, cosine_similarity_CLIP, avgScore)
        simpleWeightedAvgSimScore2_vgg = self.simpleWeightedAvg_modified(lpips_simScore_vgg, cosine_similarity_CLIP, avgScore)
        simpleWeightedAvgSimScore2_squeeze = self.simpleWeightedAvg_modified(lpips_simScore_squeeze, cosine_similarity_CLIP, avgScore)
        if doPrint:
            print(f"Simple Weighted Average Similarity Score (Alex): {simpleWeightedAvgSimScore2_alex}")
            print(f"Simple Weighted Average Similarity Score (VGG): {simpleWeightedAvgSimScore2_vgg}")
            print(f"Simple Weighted Average Similarity Score (Squeeze): {simpleWeightedAvgSimScore2_squeeze}")
        #WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

        similarity = simpleWeightedAvgSimScore2_vgg
        
        if doPrint:
            print("Image similarity", similarity)

        return similarity

    def get_clip_embedding(self, inImage):
        image = inImage
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
    def simpleWeightedAvg_modified_og(self, inScore_LPIPS, inScore_CLIP, inAvgScore, inMultiplier = 4):
        if inAvgScore > .5:
            #inScore_LPIPS = 0
            inScore_CLIP = inScore_CLIP**(1/inMultiplier)
            return inScore_CLIP
        else:
            inScore_LPIPS = inScore_LPIPS**inMultiplier
            #inScore_CLIP = 0
            return inScore_LPIPS
    def simpleWeightedAvg_modified_og2(self, inScore_LPIPS, inScore_CLIP, inAvgScore):
        #The lowest value of the multiplier is 1
        #The Highest vaue is 4
        #If the multiplier has its lowest vaue no boosting occurs
        inMultiplier = 3*(abs(.5 - inAvgScore)/.5)+1
        if inAvgScore > .5:
            #inScore_LPIPS = 0
            inScore_CLIP = inScore_CLIP**(1/inMultiplier)
            return inScore_CLIP
        else:
            inScore_LPIPS = inScore_LPIPS**inMultiplier
            #inScore_CLIP = 0
            return inScore_LPIPS
    def simpleWeightedAvg_modified(self, inScore_LPIPS, inScore_CLIP, inAvgScore, inMultiplier = None):
        if inAvgScore > .5:
            #inScore_LPIPS = 0
            inScore_CLIP = inScore_CLIP
            return inScore_CLIP
        else:
            inScore_LPIPS = inScore_LPIPS
            #inScore_CLIP = 0
            return inScore_LPIPS