from transformers import GPT2Tokenizer, GPT2Model
import datetime
import math
import numpy as np
from typing import List, Tuple

from utils import cosine_similarity, get_completion_from_messages, importance_from_gpt


model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2",output_hidden_states = True)
def generate_embedding_vector(text: str) -> List[float]:
  # use language model to generate embedding vector
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model(**inputs, output_hidden_states=True)
  hidden_states = outputs.hidden_states
  last_layer_hidden_states = hidden_states[-1]
  embedding = last_layer_hidden_states.mean(dim=1).flatten()
  return embedding.detach().numpy()

class MemoryObject(object):
  def __init__(self, creation: datetime.datetime, text: str):
    self.creation = creation
    self.text = text
    self.last_access = creation

class MemoryStream:
  def __init__(self):
    self.memory_objects = []

  def add_memory_object(self, creation, text):
    self.memory_objects.append(MemoryObject(creation, text))

  def retrieve_memory_objects(self, query_memory: str, context_window_size: int) -> List[MemoryObject]:
    recency_scores = []
    importance_scores = []
    relevance_scores = []
    a_rec, a_imp, a_rel = 1,1,1
    
    for memory_object in self.memory_objects:
      # recency score
      time_since_last_access = datetime.datetime.now() - memory_object.last_access
      recency_score = math.exp(-0.99 * time_since_last_access.total_seconds() / 3600)
      recency_scores.append(recency_score)

      # importance score using chatgpt
      importance_score = importance_from_gpt(memory_object.text)
      importance_scores.append(importance_score)

      # relevance score
      embedding_vector_memory = generate_embedding_vector(memory_object.text)
      embedding_vector_query = generate_embedding_vector(query_memory)
      relevance_score = cosine_similarity(embedding_vector_memory, embedding_vector_query)
      relevance_scores.append(relevance_score)

      # update last access time
      memory_object.last_access = datetime.datetime.now()

    # print(recency_scores)
    # # normalize scores
    # recency_scores = min_max_scale(recency_scores)
    # importance_scores = min_max_scale(importance_scores)
    # relevance_scores = min_max_scale(relevance_scores)

    # calculate final retrieval score
    scores = [(recency_scores[i], importance_scores[i], relevance_scores[i]) for i in range(len(self.memory_objects))]
    retrieval_scores = [recency_score * a_rec + importance_score * a_imp + relevance_score * a_rel for (recency_score, importance_score, relevance_score) in scores]

    # rank memories and return top ones that fit in context window
    ranked_memory_indices = sorted(range(len(retrieval_scores)), key=lambda k: retrieval_scores[k], reverse=True)
    top_memory_indices = ranked_memory_indices[:context_window_size]
    top_memory_objects = [self.memory_objects[i] for i in top_memory_indices]
    

    return top_memory_objects

class Agent(object):
    def __init__(self, name,context = None, prompt=None, initiate_conversation = False):
        self.name = name
        self.memory_stream = MemoryStream()
        self.model = "gpt-3.5-turbo"
        self.context = context
        self.initiate_conversation = initiate_conversation



    def chat(self,prompt=None):

        # if prompt is not None and self.initiate_conversation == False:
        # prompt = "Adam said, '{}'".format(prompt)
        messages = [{'role': 'system','content': self.context}]
        if self.initiate_conversation and prompt == None:
            response = get_completion_from_messages(messages)
            self.initiate_conversation = False
            self.memory_stream.add_memory_object(datetime.datetime.now(), "You have said: '{}' to Watson".format(response))
        else:
            if len(self.memory_stream.memory_objects)>0:
                relevant_mems = self.memory_stream.retrieve_memory_objects(query_memory = prompt, context_window_size = 2)
                for mem in relevant_mems:
                    messages.append({'role':'system','content': "You have said, '{}' to Watson".format(mem.text)})
                messages.append({'role': 'system','content': "Watson has said, '{}' to you.".format(prompt)})
                response = get_completion_from_messages(messages)

            else:
                messages.append({'role': 'system','content': "Watson has said, '{}' to you.".format(prompt)})
                response = get_completion_from_messages(messages)

        self.memory_stream.add_memory_object(datetime.datetime.now(),  "Watson has said, '{}' to you.".format(prompt))
        self.memory_stream.add_memory_object(datetime.datetime.now(), "You have said: '{}' to Watson".format(response))
        print(response)

        return response