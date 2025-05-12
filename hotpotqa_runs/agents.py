import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain import OpenAI, Wikipedia
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT

import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline

from evaluation import StableDiffusionEval_test

from prompts import i2p_reflect_prompt, i2p_reflect_agent_prompt
import openai
import base64
import io

import hashlib
import pandas
from PIL import Image

class ModelSettings:
    def __init__(self, type, name, temperature = None, maxTokens = None, kwargs = None):
        self.type = type
        self.name = name
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.kwargs = kwargs


class LLaVA_ModelSettings(ModelSettings):
    def __init__(self, load_in_4bit = None, bnb_4bit_compute_dtype = None, model_id = None):
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.model_id = model_id
        


class ActionLLM:
    def __init__(self, modelType):
        self.modelType = modelType

        if modelType == "LLaVA":
            self.model_id = "llava-hf/llava-1.5-7b-hf"
            self.settings = LLaVA_ModelSettings(type = self.modelType, name = "LLaVA", temperature = None, load_in_4bit = True, bnb_4bit_compute_dtype = torch.float16, model_id = "llava-hf/llava-1.5-7b-hf")
            
            #!!!!!
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.settings.load_in_4bit,
                bnb_4bit_compute_dtype=self.settings.bnb_4bit_compute_dtype
            )
            
            self.settings.kwargs = {"quantization_config": self.quantization_config}
            self.model = pipeline("image-to-text", model=self.settings.model_id, model_kwargs=self.settings.kwargs)
            #!!!!!
        elif modelType == "AnyOpenAILLM":
            self.settings = ModelSettings(type = self.modelType, name = "gpt-3.5-turbo", temperature = 0, maxTokens = 250)
            self.settings.kwargs = {"stop": "\n"}
            self.model = AnyOpenAILLM(
                temperature=self.settings.temperature,
                max_tokens=self.settings.maxTokens,
                model_name=self.settings.name,
                model_kwargs=self.settings.kwargs,
                openai_api_key=os.environ['OPENAI_API_KEY'])
        elif modelType == "gpt-vision":
            self.settings = ModelSettings(type = self.modelType, name = "gpt-4o")
            print("gpt-4o was selected")
        else:
            self.settings = ModelSettings(type = self.modelType, name = self.modelType)
            print(f"self.model type is: {self.modelType}")
            print(f"The model type is: {modelType}")
    def run(self, inPrompt, inImages = [], inMaxNewTokens = None):
        if(len(inImages) == 0):
            inImage = None
        elif(len(inImages) == 1):
            inImage = inImages[0]
        else:
            inImage = inImages[1]
        print(f"\nRunning {self.modelType} model with prompt: |{inPrompt}| ...\n")
        if(self.modelType == "LLaVA"):
            return self.run_LLaVA(inPrompt, inImage, inMaxNewTokens)
        elif(self.modelType == "AnyOpenAILLM"):
            return self.run_AnyOpenAILLM(inPrompt)
        elif self.modelType == "gpt-vision" and len(inImages) == 2:
            return self.run_GPT_4o(inPrompt, inImages, inMaxNewTokens)
        else:
            return self.run_OpenAI(inPrompt, inImages, inMaxNewTokens)
    def run_LLaVA(self, prompt, image = None, inMaxNewTokens = 400):
        if(inMaxNewTokens == None):
            max_new_tokens = 200
        else:
            max_new_tokens = inMaxNewTokens
        
        self.settings.maxTokens = max_new_tokens

        if image != None:
            outputs = self.model(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
        else:
            outputs = self.model(prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
        #return outputs
        return outputs[0]["generated_text"]#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def run_AnyOpenAILLM(self, prompt):
        return self.model(prompt)
        print("UNDER CONSTRUCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    def run_GPT_4o(self, inPrompt, inImages = [], inMaxNewTokens = 500):
        if(inMaxNewTokens == None):
            max_new_tokens = 300
        else:
            max_new_tokens = inMaxNewTokens

        self.settings.maxTokens = max_new_tokens

        image_original = inImages[0]
        image_generated = inImages[1]
        originalImage_b64 = self.encode_image(image_original)
        generatedImage_b64 = self.encode_image(image_generated)

        openai.api_key = os.environ['OPENAI_API_KEY']

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": inPrompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{originalImage_b64}"},},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{generatedImage_b64}"},},
                    ],
                }
            ],
            max_tokens=self.settings.maxTokens,
        )

        extractedText = response.choices[0].message.content

        return extractedText
    def run_OpenAI(self, inPrompt, inImages = [], inMaxNewTokens = 300):
        if(inMaxNewTokens == None):
            max_new_tokens = 300
        else:
            max_new_tokens = inMaxNewTokens

        self.settings.maxTokens = max_new_tokens




        promptContent = [{"type": "text", "text": inPrompt}]

        #Add images to content of prompt
        for img in inImages:
            img_b64 = self.encode_image(img)
            promptContent.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"},})



        openai.api_key = os.environ['OPENAI_API_KEY']

        response = openai.ChatCompletion.create(
            model=self.modelType,
            messages=[
                {
                    "role": "user",
                    "content": promptContent,
                }
            ],
            max_tokens=self.settings.maxTokens,
        )

        extractedText = response.choices[0].message.content

        return extractedText
    

    
    def encode_image(self, image):
        """Encodes a PIL Image object to a base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # or "PNG" depending on your image
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
        

#class ReflectLLM():


class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


class CoTAgent:
    def __init__(self,
                    action_task: str,
                    reflect_task: str,
                    context_agent: str,
                    context_reflection: str,
                    agent_prompt: PromptTemplate = i2p_reflect_agent_prompt,
                    reflect_prompt: PromptTemplate = i2p_reflect_prompt,
                    cot_examples: str = COT,#!!!!!!!!!!!!!!!!!!!!!!
                    reflect_examples: str = COT_REFLECT,#!!!!!!!!!!
                    reflectLLM_modelType: str = "AnyOpenAILLM",
                    actionLLM_modelType: str = "LLaVA",
                    threshold: float = 0.90,
                    maxStep: int = 3,
                    simplePromptMode = True,
                    reportFolder_path = "../reports/",
                    runReport_path = None,
                    stepReport_path = None,
                    doPrint = False,
                    
                    ) -> None:
        self.action_task = action_task
        self.reflect_task = reflect_task
        self.context_agent = context_agent
        self.context_reflection = context_reflection
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples 
        self.reflect_examples = reflect_examples
        self.reflectLLM_modelType = reflectLLM_modelType
        self.actionLLM_modelType = actionLLM_modelType
        self.self_reflect_llm = ActionLLM(reflectLLM_modelType)
        self.action_llm = ActionLLM(actionLLM_modelType)
        self.threshold = threshold
        self.maxStep = maxStep
        self.doPrint = doPrint
        self.simplePromptMode = False

        self.reflections: List[str] = []
        self.reflections_str = ''
        self.answer = ''
        self.step_n: int = 1
        self.originalImage = None
        self.generatedImage = None
        self.generatedImages = []
        self.scratchpad: str = ''
        self.previousScratchpad: str = ''
        self.scratchpads = []
        self.modelOutputs = []
        self.reset()
        if runReport_path == None:
            self.runReport_path = reportFolder_path + "Run_Reports.csv"
        else: 
            self.runReport_path = runReport_path
        if stepReport_path == None:
            self.stepReport_path = reportFolder_path + "Step_Reports.csv"
        else:
            self.stepReport_path = stepReport_path
        self.step_reports: list[StepReport] = []

        
        

    def run(self, inInputImageFolder, inInputImageFilename, inOutputImageFolder, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION, inMaxStep: int = None, inThreshold = None) -> None:
        if inMaxStep == None:
            maxSteps = self.maxStep
        else:
            maxSteps = inMaxStep
        
        if inThreshold != None:
            self.threshold = inThreshold
        
        self.runReport = RunReport(runReport_path = self.runReport_path, image_path = inImagePath, threshold = self.threshold, max_steps = self.maxStep, agent_model_type = self.actionLLM_modelType, agent_model_name = self.action_llm.settings.name, agent_model_setting_temperature = self.action_llm.settings.temperature, agent_model_setting_max_tokens = self.action_llm.settings.maxTokens, agent_model_setting_misc = "N/A", reflection_model_type = self.self_reflect_llm.settings.type, reflection_model_name = self.self_reflect_llm.settings.name, reflection_model_setting_temperature = self.self_reflect_llm.settings.temperature, reflection_model_setting_max_tokens = self.self_reflect_llm.settings.maxTokens, reflection_model_setting_misc = "N/A", agent_prompt_template = self.getAgentPromptTemplate(), reflection_prompt_template = self.getReflectionPromptTemplate())
        self.step_reports = []


        inImagePath = inInputImageFolder + inInputImageFilename + ".png"
        self.originalImageFolder = inInputImageFolder
        self.originalImageFilename = inInputImageFilename
        self.inputImagePath = inImagePath
        inImage = Image.open(inImagePath)

        self.generatedImageFolder = inOutputImageFolder

        
        
        
        print("\n\n===============================================================")
        self.originalImage = inImage
        self.reset()
        self.step()
        self.step_n += 1
        #Loop until done
        #If it is correct after then your done and the following code will never be excecuted
        while(self.step_n > 1 and self.step_n <= maxSteps and not self.is_correct(self.answer, inImage) and reflexion_strategy != ReflexionStrategy.NONE):
            self.reflect(reflexion_strategy)
            print("---------------------------------------------------------------")
            self.reset()
            self.step()
            self.step_n += 1
        self.runReport.is_successful = self.is_correct(self.answer, inImage)
        self.runReport.save()
        for step_report in self.step_reports:
            step_report.run_id = self.runReport.run_id
            step_report.save()
        self.reset()
        self.generatedImagePath = None
        print("===============================================================\n\n")
    def step(self, inImage = None) -> None:

        self.stepReport = StepReport(stepReport_path = self.stepReport_path, step = self.step_n)

        if inImage == None:
            inImage = self.originalImage
        
        
        print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
        
        if self.step_n == 1:
            self.scratchpad = "N/A (This is the first trial)"
        else:
            self.scratchpad = self.previousScratchpad

        modelOutput = self.prompt_agent(inImage)
        self.scratchpad: str = ''
        
        self.stepReport.agent_response = modelOutput

        # Think
        print(f"Model output: {modelOutput}")
        self.thought = self.formatAgentResponse(modelOutput, "thought")
        self.scratchpad += f'\nThought:'
        print(f"Adding processed thought to scratchpad: |{self.thought}|...")
        self.scratchpad += ' ' + self.thought
        
        
        
        # Act
        
        
        
        self.action = self.formatAgentResponse(modelOutput, "action")
        self.scratchpad += f'\nAction:'
        print(f"Adding processed action to scratchpad: |{self.action}|...")
        self.scratchpad += ' ' + self.action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #action_type, argument = parse_action(action)
        
        
        self.scratchpad += f'\nObservation: '
        '''if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            return
        else:
            print('Invalid action type, please try again.')
        '''
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.answer = self.action
        '''
        if self.is_correct(self.action, inImage):
            self.scratchpad += 'Answer is CORRECT'
            print('Answer is CORRECT')
        else: 
            self.scratchpad += 'Answer is INCORRECT'
            print('Answer is INCORRECT')
        '''
        self.finished = True
        print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
        print(f"Finished step!\n[Scratch Pad]\n{self.scratchpad}\n\n")
        self.modelOutputs.append(modelOutput)
        self.previousScratchpad = self.scratchpad
        

        

        
        
        self.stepReport.is_successful = self.is_correct(self.answer, inImage)
        #similarity_score is calculated in the process of running is_correct()
        self.stepReport.similarity_score = self.similarityScore
        self.generatedImagePath = self.calcGeneratedImagePath(self.generatedImageFolder, self.originalImageFilename)
        #save image
        self.generatedImage.save(self.generatedImagePath, format='PNG')
        self.stepReport.output_image_path = self.generatedImagePath
        

        self.step_reports.append(self.stepReport)

    
    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Running Reflexion strategy...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.action_task , self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.action_task , self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += '\n'+ format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        self.stepReport.reflection_prompt = self._build_reflection_prompt()
        self.stepReport.reflection_response = format_step(self.self_reflect_llm.run(self.stepReport.reflection_prompt, [self.originalImage, self.generatedImage]))
        return self.stepReport.reflection_response

    def reset(self) -> None:
        self.scratchpads.append(self.scratchpad)
        self.scratchpad: str = ''
        self.finished = False
        if(self.generatedImage != None):
            self.generatedImages.append(self.generatedImage)
        self.generatedImage = None
        self.similarityScore = None


    
    def prompt_agent(self, inImage) -> str:
        if(self.actionLLM_modelType == "LLaVA"):
            #tempPrompt = "Generate a prompt that could be used to generate a similar image."
            self.stepReport.agent_prompt = self._build_agent_prompt()
            modelOutput = self.action_llm.run(self.stepReport.agent_prompt, [inImage])
        else:
            self.stepReport.agent_prompt = self._build_agent_prompt()
            modelOutput = format_step(self.action_llm.run(self.stepReport.agent_prompt, [inImage]))
        return modelOutput
    
    
    
     
    def _build_agent_prompt(self, inPrompt: str = None) -> str:
        if inPrompt == None:
            task = self.action_task
        else:
            task = inPrompt

        if self.actionLLM_modelType == "LLaVA":
            if(self.simplePromptMode == True):
                newPrompt = "USER: <image>\n" + task + "\nASSISTANT:"
            else:
                promptFromTemplate = self.agent_prompt.format(
                                    reflections = self.reflections_str,
                                    context = self.context_agent,
                                    action_agent_task = task,
                                    scratchpad = self.previousScratchpad)
                newPrompt = "USER: <image>\n" + promptFromTemplate + "\nASSISTANT:"
        else:
            newPrompt = self.agent_prompt.format(
                                reflections = self.reflections_str,
                                context = self.context_agent,
                                action_agent_task = task,
                                scratchpad = self.previousScratchpad)
        return newPrompt
    
    def getAgentPromptTemplate(self, inPrompt: str = None) -> str:
        if inPrompt == None:
            task = self.action_task
        else:
            task = inPrompt

        if self.actionLLM_modelType == "LLaVA":
            if(self.simplePromptMode == True):
                newPrompt = "USER: <image>\n" + task + "\nASSISTANT:"
            else:
                promptFromTemplate = self.agent_prompt.format(
                                    reflections = "\n(Reflections)\n",
                                    context = self.context_agent,
                                    action_agent_task = task,
                                    scratchpad = "\n(Scratchpad)\n")
                newPrompt = "USER: <image>\n" + promptFromTemplate + "\nASSISTANT:"
        else:
            newPrompt = self.agent_prompt.format(
                                reflections = "\n(Reflections)\n",
                                context = self.context_agent,
                                action_agent_task = task,
                                scratchpad = "\n(Scratchpad)\n")
        return newPrompt
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def _build_reflection_prompt(self) -> str:
        if(self.simplePromptMode == True):
            #examples = "N/A"
            context = """The model was given the task of analyzing an input image and generating a prompt that can be used to generate a similar image. Image A is the original and Image B was created using the prompt generated from analyzing the original image.

                        Image A: <image_1>
                        Image B: <image_2>
                        """
        else:
            #examples = self.reflect_examples
            context = self.context_reflection
            
        return self.reflect_prompt.format(
                            context = context,
                            reflect_agent_task = self.reflect_task,
                            scratchpad = self.previousScratchpad)
    def getReflectionPromptTemplate(self) -> str:
        if(self.simplePromptMode == True):
            #examples = "N/A"
            context = """The model was given the task of analyzing an input image and generating a prompt that can be used to generate a similar image. Image A is the original and Image B was created using the prompt generated from analyzing the original image.

                        Image A: <image_1>
                        Image B: <image_2>
                        """
        else:
            #examples = self.reflect_examples
            context = self.context_reflection
            
        return self.reflect_prompt.format(
                            context = context,
                            reflect_agent_task = self.reflect_task,
                            scratchpad = "\n(Scratchpad)\n")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def is_finished(self) -> bool:
        return self.finished

    
    def calcSimScoreAndOutImage(self, modelOutput, inImage):
        evaluator = StableDiffusionEval_test()
        similarityScore, outGeneratedImage = evaluator.evaluatePrompt(modelOutput, inImage, self.doPrint)
        self.generatedImage = outGeneratedImage
        self.similarityScore = similarityScore
        
        print("Displaying generated image...")
        print("self.generatedImage.show():")
        self.generatedImage.show()
        print("self.generatedImage:")
        self.generatedImage
        print("display generated image COMPLETE")

    def is_correct(self, modelOutput, inImage) -> bool:   
        self.calcSimScoreAndOutImage(self, modelOutput, inImage)

        if self.similarityScore > self.threshold:
            self.scratchpad += 'Answer is CORRECT'
            print('Answer is CORRECT')
            return True
        else:
            self.scratchpad += 'Answer is INCORRECT!' + ' Similarity Score: ' +  str(similarityScore) + " (Which failed to surpass the goal of " + str(self.threshold) + ")"
            print('Answer is INCORRECT')
            self.previousScratchpad = self.scratchpad
            return False
       
    def formatAgentResponse(self, inResponse: str, responseType: str = None) -> str:
        tempResponse = inResponse
        if(self.actionLLM_modelType == "LLaVA"):
            targetString = "ASSISTANT: "
            startIndex = tempResponse.find(targetString)
            if(startIndex > -1):
                lenTarget = len(targetString)
                targetIndex = startIndex + lenTarget
                inResponse = tempResponse[targetIndex:]
                tempResponse = inResponse

                
        
        if(responseType in ["thought", "action"]):
            if(responseType == "thought"):
                startString = "[Thought Start]"
                endString = "[Thought End]"
            elif(responseType == "action"):
                startString = "[Prompt Start]"
                endString = "[Prompt End]"
            
            startIndex = inResponse.find(startString) + len(startString)
            endIndex = inResponse.find(endString)

            newResponse = inResponse[startIndex:endIndex]
        else:
            targetStrings = ["ASSISTANT: ", "PROMPT: ", "Prompt:", "Thought: ", "Thought:", "Finish[prompt]: "]
            for targetString in targetStrings:
                startIndex = tempResponse.find(targetString)
                if(startIndex > -1):
                    lenTarget = len(targetString)
                    targetIndex = startIndex + lenTarget
                    tempResponse = tempResponse[targetIndex:]
                
            newResponse = tempResponse
            
        return newResponse
    
        
    def formatAgentResponse_og(self, inThought: str) -> str:
        if(self.actionLLM_modelType == "LLaVA"):
            tempThought = inThought
            targetStrings = ["ASSISTANT: ", "PROMPT: ", "Prompt:", "Thought: ", "Thought:", "Finish[prompt]: "]
            for targetString in targetStrings:
                startIndex = tempThought.find(targetString)
                if(startIndex > -1):
                    lenTarget = len(targetString)
                    targetIndex = startIndex + lenTarget
                    tempThought = tempThought[targetIndex:]
                
            newThought = tempThought
             
            return newThought
        else:
            return inThought

    def calcGeneratedImagePath(self, inPathTogeneratedImageFolder,inInputImageFilename, miscText = ""):
        currTimestamp = pandas.Timestamp.now(tz="UTC")
        timestamp_str = currTimestamp.strftime("%Y%m%d%H%M%S%f")
        outputImagePath = inPathTogeneratedImageFolder + inInputImageFilename + "-generatedImage_" + timestamp_str + miscText + ".png"
        


'''
class ReactAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                            temperature=0,
                                            max_tokens=100,
                                            model_name="gpt-3.5-turbo",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6

        self.docstore = DocstoreExplorer(docstore) # Search, Lookup
        self.llm = react_llm
        
        self.enc = tiktoken.encoding_for_model("text-davinci-003")

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        return format_step(self.llm(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                             temperature=0,
                                             max_tokens=100,
                                             model_name="gpt-3.5-turbo",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY']),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=250,
                                               model_name="gpt-3.5-turbo",
                                               openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''
    
    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset)
    
    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION: 
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))


    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = truncate_scratchpad(self.scratchpad, tokenizer=self.enc))
 
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            reflections = self.reflections_str,
                            question = self.question,
                            scratchpad = self.scratchpad)
   
'''


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(task: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Task: {task}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


def getImageID(inFilename):
    imageID = getImageHash(inFilename)
    return imageID
def getImageHash(inFilename):
    with open(inFilename,"rb") as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash



class RunReport:
    def __init__(self, runReport_path, image_path, run_id= None, duration= None, start_timestamp= None, end_timestamp= None, image_id= None, image_path_PLACEHOLDER= None, threshold= None, max_steps= None, agent_model_type= None, agent_model_name= None, agent_model_setting_temperature= None, agent_model_setting_max_tokens= None, agent_model_setting_misc= None, reflection_model_type= None, reflection_model_name= None, reflection_model_setting_temperature= None, reflection_model_setting_max_tokens= None, reflection_model_setting_misc= None, agent_prompt_template= None, reflection_prompt_template= None, is_successful= None, run_feedback= None):
        self.timestamp = pandas.Timestamp.now(tz="UTC")

        self.runReport_path = runReport_path
        self.run_id = run_id
        
        self.duration = duration
        
        if start_timestamp == None:
            self.start_timestamp = self.timestamp
        else:
            self.start_timestamp = start_timestamp
        
        self.end_timestamp = end_timestamp
        
        if image_id == None:
            self.image_id = getImageID(image_path)
        else:
            self.image_id = image_id
        
        self.image_path = image_path
        self.threshold = threshold
        self.max_steps = max_steps
        self.agent_model_type = agent_model_type
        self.agent_model_name = agent_model_name
        self.agent_model_setting_temperature = agent_model_setting_temperature
        self.agent_model_setting_max_tokens = agent_model_setting_max_tokens
        self.agent_model_setting_misc = agent_model_setting_misc
        self.reflection_model_type = reflection_model_type
        self.reflection_model_name = reflection_model_name
        self.reflection_model_setting_temperature = reflection_model_setting_temperature
        self.reflection_model_setting_max_tokens = reflection_model_setting_max_tokens
        self.reflection_model_setting_misc = reflection_model_setting_misc
        self.agent_prompt_template = agent_prompt_template
        self.reflection_prompt_template = reflection_prompt_template
        self.is_successful = is_successful
        self.run_feedback = run_feedback

    def setRunID(self):
        self.run_id = TimestampToStr(self.start_timestamp) + TimestampToStr(self.end_timestamp)

    

        
    def createDictionary(self):
        if self.end_timestamp == None:
            self.end_timestamp = pandas.Timestamp.now(tz="UTC")
        
        if self.duration == None:
            self.duration = self.end_timestamp - self.start_timestamp

        if self.run_id == None:
            self.setRunID()
        
        if self.image_id == None:
            self.image_id = getImageID(self.image_path)

        self.dictionary = {
            'run_id': [self.run_id],
            'duration': [self.duration],
            'start_timestamp': [self.start_timestamp],
            'end_timestamp': [self.end_timestamp],
            'image_id': [self.image_id],
            'image_path': [self.image_path],
            'threshold': [self.threshold],
            'max_steps': [self.max_steps],
            'agent_model_type': [self.agent_model_type],
            'agent_model_name': [self.agent_model_name],
            'agent_model_setting_temperature': [self.agent_model_setting_temperature],
            'agent_model_setting_max_tokens': [self.agent_model_setting_max_tokens],
            'agent_model_setting_misc': [self.agent_model_setting_misc],
            'reflection_model_type': [self.reflection_model_type],
            'reflection_model_name': [self.reflection_model_name],
            'reflection_model_setting_temperature': [self.reflection_model_setting_temperature],
            'reflection_model_setting_max_tokens': [self.reflection_model_setting_max_tokens],
            'reflection_model_setting_misc': [self.reflection_model_setting_misc],
            'agent_prompt_template': [self.agent_prompt_template],
            'reflection_prompt_template': [self.reflection_prompt_template],
            'is_successful': [self.is_successful],
            'run_feedback': [self.run_feedback]
        }
    def createDataFrame(self):
        self.createDictionary()
        self.dataFrame = pandas.DataFrame(self.dictionary)
        self.dataFrame.set_index('run_id', inplace=True)

    def save(self, addIndexCol = True, addHeader = False):
        self.createDataFrame()
        self.dataFrame.to_csv(self.runReport_path, mode='a', index=addIndexCol, header=addHeader)
    def saveTo(self, path, addIndexCol = True, addHeader = False):
        self.createDataFrame()
        self.dataFrame.to_csv(path, mode='a', index=addIndexCol, header=addHeader)


class StepReport:
    def __init__(self, stepReport_path: str, step: int, run_id = None, output_image_path = None, output_image_sha256 = None, start_timestamp = None, end_timestamp = None, duration = None, agent_prompt = None, agent_response = None, reflection_prompt = None, reflection_response = None, similarity_score = None, is_successful = None, step_feedback = None):
        self.timestamp = pandas.Timestamp.now(tz="UTC")

        self.stepReport_path = stepReport_path
        
        self.run_id = run_id
        self.step = step
        
        self.output_image_path = output_image_path
        
        if output_image_sha256 == None:
            self.output_image_sha256 = getImageHash(output_image_path)
        else:
            self.output_image_sha256 = output_image_sha256
        
        
        
        if start_timestamp == None:
            self.start_timestamp = self.timestamp
        else:
            self.start_timestamp = start_timestamp
        
        self.end_timestamp = end_timestamp

        self.duration = duration
        
        
        
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.duration = duration
        self.agent_prompt = agent_prompt
        self.agent_response = agent_response
        self.reflection_prompt = reflection_prompt
        self.reflection_response = reflection_response
        self.similarity_score = similarity_score
        self.is_successful = is_successful
        self.step_feedback = step_feedback
    
    

        
    def createDictionary(self):
        if self.end_timestamp == None:
            self.end_timestamp = pandas.Timestamp.now(tz="UTC")
        
        if self.duration == None:
            self.duration = self.end_timestamp - self.start_timestamp
        
        if self.output_image_sha256 == None:
            self.output_image_sha256 = getImageHash(self.output_image_path)
        
        self.dictionary = {
            'run_id': [self.run_id],
            'step': [self.step],
            'output_image_path': [self.output_image_path],
            'output_image_sha256': [self.output_image_sha256],
            'start_timestamp': [self.start_timestamp],
            'end_timestamp': [self.end_timestamp],
            'duration': [self.duration],
            'agent_prompt': [self.agent_prompt],
            'agent_response': [self.agent_response],
            'reflection_prompt': [self.reflection_prompt],
            'reflection_response': [self.reflection_response],
            'similarity_score': [self.similarity_score],
            'is_successful': [self.is_successful],
            'step_feedback': [self.step_feedback]
        }
    def createDataFrame(self):
        self.createDictionary()
        self.dataFrame = pandas.DataFrame(self.dictionary)
        
    def save(self, addIndexCol = False, addHeader = False):
        self.createDataFrame()
        self.dataFrame.to_csv(self.stepReport_path, mode='a', index=addIndexCol, header=addHeader)
    def saveTo(self, path, addIndexCol = False, addHeader = False):
        self.createDataFrame()
        self.dataFrame.to_csv(path, mode='a', index=addIndexCol, header=addHeader)


def TimestampToStr(inTimestamp):
    year    = inTimestamp.year
    month   = inTimestamp.month
    day     = inTimestamp.day
    hour    = inTimestamp.hour
    minute  = inTimestamp.minute
    second  = inTimestamp.second
    microSecond   = inTimestamp.microsecond
    nanoSecond    = inTimestamp.nanosecond

    
    timestamp_str = f"{year:04}{month:02}{day:02}{hour:02}{minute:02}{second:02}{microSecond:06}{nanoSecond:03}"
    return timestamp_str
