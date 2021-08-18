#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xml.etree.cElementTree as ET
import pprint
from pprint import pprint
from collections import defaultdict
import re
import os
import csv
import codecs
import cerberus
import sqlite3
import pandas as pd

#getting unique users (!Ref. "Case Study - Exploring Users"!)

def get_user(element):
    uid = ''
    if element.tag == "node" or element.tag == "way" or element.tag == "relation":
        uid = element.get('uid')
    return uid


def process_map(filename):
    users = set()
    for _, element in ET.iterparse(filename):
        if get_user(element):
            users.add(get_user(element))
            users.discard('')
        pass

    return users

users = process_map(SAMPLE_FILE)  # Using Sample file as input
print ('UNIQUE USERS: ', len(users))


# Counting the element tags in the file (!Ref. Case Study - Iterative Parsing!)

def count_tags(filename):
    tags = {}
    for event,elem in ET.iterparse(filename, events=("start",)):
        if elem.tag in tags.keys():
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags    
    
tags = count_tags(SAMPLE_FILE)  # Using Sample file as input
pprint(tags)


# Finding out formatting scheme of K attribute in tags (!Ref. "Case study OSM - Tag Types"!)

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

keys = defaultdict(int)

def key_count(keys, element):
    if element.tag == 'tag':
        keys[element.attrib['k']] += 1
    return keys

def key_type(element, key_categories):
    if element.tag == "tag":
        if lower.match(element.attrib['k']):
            key_categories["lower"] += 1
        elif lower_colon.search(element.attrib['k']):
            key_categories["lower_colon"] += 1
        elif problemchars.search(element.attrib['k']):
            key_categories["problemchars"] += 1
        else:
            key_categories["other"] += 1
        pass
        
    return key_categories

def process_key_map(filename):
    key_categories = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        key_categories = key_type(element, key_categories)
        key_count(keys, element)

    return key_categories

key_categories = process_key_map(SAMPLE_FILE)
pprint(key_categories)


def filter_count_tags(d, threshold=100):
    l1 = []
    for key in d:
        if d[key] > threshold:
            l1.append((key, d[key]))
    l1.sort(key=lambda tup: tup[1], reverse= True)
    return l1

filter_count_tags(keys)

