import openai
from typing import List
import numpy as np

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    return embedding

def get_importance(text):
    prompt = "On the scale of 1 to 10, where 1 is purely mundane"\
    + " (e.g., brushing teeth, making bed) and 10 is"\
    + " extremely important (e.g., a break up, college"\
    + " acceptance), rate the likely importance of the"\
    + " following piece of memory. Respond with a single integer."\
    + f"\nMemory: {text}"\
    + "\nRating: "
    score = int(get_completion_from_messages(prompt))
    return score

def get_importances(text):
    prompt = "On the scale of 1 to 10, where 1 is purely mundane"\
    +" (e.g., brushing teeth, making bed) and 10 is"\
    + " extremely important (e.g., a break up, college"\
    + " acceptance), rate the likely importances of the"\
    + " following pieces of memories. Always answer with only a list of numbers."\
    + " If just given one memory still respond in a list."\
    + " Memories are separated by semicolons (;)"\
    + f"\Memories: {text}"\
    + "\nRating: "
    score_list = get_completion_from_messages(prompt)
    return [int(x) for x in score_list]

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
            )
    return response.choices[0].message["content"]

def cosine_similarity(vector1, vector2) -> float:
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

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