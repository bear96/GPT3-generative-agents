import openai
import datetime
import math
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from .utils import *
import re


class MemoryObject(object):
  def __init__(self,  text, creation = None):
    self.creation = datetime.datetime.now() if creation is None else creation
    self.text = text
    self.last_access = creation
    self.embedding = None
    self.importance = None
    self.relevance = None
    self.recency = None

class MemoryStream:
  def __init__(self,reflection_threshold = 50,sort_by_importance=False, context_window_size = 15):
    self.context_window_size = context_window_size
    self.memory_objects = []
    self.a_imp = 1
    self.a_rec = 1
    self.a_rel = 1
    self.aggregate_importance = 0.0
    self.reflection_threshold = reflection_threshold
    self.sort_by_importance = sort_by_importance

  def add_memory(self, text,creation = None):
    importance = get_importance(text)
    embedding = get_embedding(text)
    mem = MemoryObject(text, creation)
    mem.importance = importance
    self.aggregate_importance += importance
    mem.embedding = embedding
    self.memory_objects.append(mem)
    if (self.reflection_threshold is not None and self.aggregate_importance > self.reflection_threshold and not self.reflecting):
      self.reflecting = True
      self.pause_to_reflect(now=now)
      self.aggregate_importance = 0.0
      self.reflecting = False

  def add_memories(self , text, creation = None):
    importances = get_importances(text)
    self.aggregate_importance += sum(importances)
    mems_list = text.split(";")
    for i in range(len(mems_list)):
      embedding = get_embedding(mems_list[i])
      mem = MemoryObject(mems_list[i], creation)
      mem.embedding = embedding
      mem.importance = importances[i]
      self.memory_objects.append(mem)
    if (self.reflection_threshold is not None and self.aggregate_importance > self.reflection_threshold and not self.reflecting):
      self.reflecting = True
      self.pause_to_reflect()
      self.aggregate_importance = 0.0
      self.reflecting = False


  def retrieve_memory_objects(self, query: str) -> List[MemoryObject]:
    embedding_vector_query = get_embedding(query)
    for memory_object in self.memory_objects:
      # recency score
      time_since_last_access = datetime.datetime.now() - memory_object.last_access
      recency_score = math.exp(-0.99 * time_since_last_access.total_seconds() / 3600)
      memory_object.recency = recency_score
      memory_object.relevance = cosine_similarity(memory_object.embedding, embedding_vector_query)

      # update last access time
      memory_object.last_access = datetime.datetime.now()

    # calculate final retrieval score
    scores = [(mems.recency, mems.importance, mems.relevance) for mems in self.memory_objects]
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    normalized_scores = scaler.fit_transform(scores)

    retrieval_scores = [self.a_rec*recency_score + self.a_imp*importance_score + self.a_rel*relevance_score for (recency_score, importance_score, relevance_score) in normalized_scores]

    # rank memories and return top ones that fit in context window
    ranked_memory_indices = sorted(range(len(retrieval_scores)), key=lambda k: retrieval_scores[k], reverse=True)
    top_memory_indices = ranked_memory_indices[:context_window_size]
    top_memory_objects = [self.memory_objects[i] for i in top_memory_indices]
    
    return top_memory_objects

  def clear_memory(self):
    self.memory_objects.clear()

  def _get_topics_of_reflection(self, k = 50):
    if self.sort_by_importance:
      imp_mem_objects = sorted(self.memory_objects, key=lambda x: x.importance, reverse=True)
      reflection_memories = imp_mem_objects[:k]
    else:
      imp_mem_objects = self.memory_objects[:k]

    observation_str = "\n".join([x.text for x in reflection_memories])
    prompt = f"{observation_str}\n\n"\
    +"Given only the information above, what are the 3 most salient "\
    +"high-level questions we can answer about the subjects in the statements?\n"\
    +"Provide each question on a new line."\

    result = get_completion(prompt)
    topics = re.split(r"\n", result.strip())
    topics = [topic for topic in topics if topic.strip()]
    return topics

  def _get_insights_on_topic(self, topic):
    related_memories = self.retrieve_memory_objects(topic)
    related_statements = "\n".join([m.text for m in related_memories])
    prompt = f"Statements relevant to: '{topic}'\n"\
      +"---\n"\
      +f"{related_statements}\n"\
      +"---\n"\
      +"What 5 high-level novel insights can you infer from the above statements "\
      +"that are relevant for answering the following question?\n"\
      +"Do not include any insights that are not relevant to the question.\n"\
      +"Do not repeat any insights that have already been made.\n\n"\
      +f"Question: {topic}\n\n"\
      +"(example format: insight (because of 1, 5, 3))"
    result = get_completion(prompt)
    insights = re.split(r"\n", result.strip())
    insights = [insight for insight in insights if insight.strip()]
    return insights

  def pause_to_reflect(self):
    new_insights = []
    topics = self._get_topics_of_reflection()
    for topic in topics:
      insights = self._get_insights_on_topic(topic)
      for insight in insights:
        self.add_memory(insight)
