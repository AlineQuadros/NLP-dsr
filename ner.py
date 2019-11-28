# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:56:04 2019

@author: MyLAP
"""
import spacy

nlp = spacy.load("en_core_web_sm")

sentence = str(abstracts.abstract[4])
sentence_nlp = nlp(sentence)

# print named entities in article
print([(word, word.ent_type_) for word in sentence_nlp if word.ent_type_])

# visualize named entities
displacy.render(sentence_nlp, style='ent', jupyter=True)
