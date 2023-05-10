import openai
from typing import List
import numpy as np


openai.api_key  = ""
def importance_from_gpt(text: str) -> int:
    # needs to feed text to chatgpt and return an integer
    prompt = """On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 
        is extremely poignant (e.g., a break up, college 
        acceptance), rate the likely poignancy of the following piece of memory. Make sure no matter what the circumstances are,
        you provide a number only. Nothing else.
        Memory: {} 
        Rating: <fill in>""".format(text)
    importance = get_completion(prompt)
    return int(importance)

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
            )
    return response.choices[0].message["content"]

def cosine_similarity(vector1, vector2) -> float:
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

def min_max_scale(scores: List[float]) -> List[float]:
    min_score = min(scores)
    max_score = max(scores)
    scaled_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return scaled_scores

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        
        temperature=temperature, # this is the degree of randomness of the model's output
        )
    return response.choices[0].message["content"]

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def passive_voice(text):
    messages = [{'role': 'system','content': '''You are passiveBot. Your task is to turn active voice into passive. Keep it short. An example would be:\n
        Bob said, "Hi, I'm Bob" \n
        Your response: Bob introduced himself.'''},{'role':'user','content': text}]
    return get_completion_from_messages(messages)