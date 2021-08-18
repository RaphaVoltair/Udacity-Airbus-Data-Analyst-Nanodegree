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

#Data Cleaning
#Phone numbers:

# ================================================== #
#               Cleaning Functions                   #
# ================================================== #

def update_phone(phone_num):

  m = phone_re.search(phone_num)
  if not m:
    # remove all dashes
    if '-' in phone_num:
      phone_num = re.sub('-', '', phone_num)
    # remove all brackets
    if '(' in phone_num or ')' in phone_num:
      phone_num = re.sub('[()]', '', phone_num)
    # remove all slashes
    if '/' in phone_num:
      phone_num = re.sub('/','',phone_num)
    # remove all spaces
    if ' ' in phone_num:
      phone_num = re.sub(' ','', phone_num)
    # insert + sign if needed
    if re.match(r'49', phone_num):
      phone_num = '+' + phone_num
    # insert country code if needed
    if re.match(r'\w+', phone_num):
      phone_num = '+49' + phone_num
    # remove 0 from regional code
    if re.search(r'^\+490', phone_num):
      p = re.search(r'^(\+49)(0)', phone_num)
      phone_num = re.sub(r'^\+49(0)', p.group(1), phone_num)
    # insert space after country code and regional code
    if re.match(r'^(\+49)([1-9]\d{2})', phone_num):
      p = re.compile(r'(\+49)([1-9]\d{2})')
      phone_num = p.sub(r'\1 \2 ', phone_num)
    # exclude all numbers that after all the above corrections still do not match the desired pattern
    # this will correct for number lenghth (too long or too short) and for special characters such as ';' or ',' or '#'
    if re.search(r'^\+49\s\d{3}\s(\d{6,8})$', phone_num) is None:
      return
    
  return  phone_num

#Preparing for Database

#(!Ref. "Case Study - Preparing for Database - SQL"!)

import schema
schema_file = open("my_schema.py")
from my_schema import schema

OSM_PATH = SAMPLE_FILE

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
# Regex to match phone numbers with correct format
phone_re = re.compile(r'^\+49\s\d{3}\s\d{6,8}$')
# Regex to match correct postcode format
postcode_re = re.compile(r'^\d{5}$')

SCHEMA = schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

#Cleaning Post codes:

# ================================================== #
#               Cleaning Functions                   #
# ================================================== #

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements
    
    if element.tag == 'node':
        # creating the dictionary for the node key
        for item in node_attr_fields:
            node_attribs[item] = element.attrib[item]
        # create the list of dictionaries for the node_tags key
        for tag in element.findall('tag'):
            entry = {}
            entry['id'] = node_attribs['id']
            if tag.attrib['k'] == 'phone' or tag.attrib['k'] == 'contact:phone':
              result = update_phone(tag.attrib['v'])
              if result:
                entry['value'] = result
              else:
                continue
            elif tag.attrib['k'] == 'addr:postcode':
              m = postcode_re.search(tag.attrib['v'])
              if m:
                postcode = m.group()
                if postcode not in expected:
                  # remove all elements associated with this postcode
                  return
                else:
                  entry['value'] = tag.attrib['v']
              else:
                # remove postcode tags that do not match regex
                continue
                  

            else:
              entry['value'] = tag.attrib['v']

            # get the match objects for problem and colon characters
            m = problem_chars.search(tag.attrib['k'])
            t = LOWER_COLON.search(tag.attrib['k'])
            # check if it has problem characters
            if m:
              return
            # check if there are colons in the value of k
            elif t:
              entry['key'] = tag.attrib['k'].split(':',1)[1]
              entry['type'] = tag.attrib['k'].split(':',1)[0]
            else:
              entry['key'] = tag.attrib['k']
              entry['type'] = default_tag_type
                

            
            tags.append(entry)
            
    if element.tag == 'way':
        # creating the dictionary for the way key
        for item in way_attr_fields:
            way_attribs[item] = element.attrib[item]
        # create the list of dictionaries for the way_tags key
        for tag in element.findall('tag'):
            entry = {}
            entry['id'] = way_attribs['id']
            if tag.attrib['k'] == 'phone' or tag.attrib['k'] == 'contact:phone':
              result = update_phone(tag.attrib['v'])
              if result:
                entry['value'] = result
              else:
                continue
            elif tag.attrib['k'] == 'addr:postcode':
              m = postcode_re.search(tag.attrib['v'])
              if m:
                postcode = m.group()
                if postcode not in expected:
                  # remove all elements associated with this postcode
                  return
                else:
                  entry['value'] = tag.attrib['v']
              else:
                # remove postcode tags that do not match regex
                continue
            else:
              entry['value'] = tag.attrib['v']

            # get the match objects for problem and colon characters
            m = problem_chars.search(tag.attrib['k'])
            t = LOWER_COLON.search(tag.attrib['k'])
            # check if it has problem characters
            if m:
              return
            # check if there are colons in the value of k
            elif t:
              entry['key'] = tag.attrib['k'].split(':',1)[1]
              entry['type'] = tag.attrib['k'].split(':',1)[0]
            else:
              entry['key'] = tag.attrib['k']
              entry['type'] = default_tag_type
            
            tags.append(entry)
        # create the list of dictionaries for way_nodes   
        count = 0
        for nd in element.findall('nd'):
            entry = {}
            entry['id'] = way_attribs['id']
            entry['node_id'] = nd.attrib['ref']
            entry['position'] = count
            count = count + 1
            
            way_nodes.append(entry)

            
    
       
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}

# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors), 'element =', element
        
        raise Exception(message_string.format(field, error_string))
        
#Write CSV-Files

# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    nodes = []
    node_tags = []
    way = []
    way_nodes = []
    way_tags = []

    validator = cerberus.Validator()

    for element in get_element(file_in, tags=('node', 'way')):
        el = shape_element(element)
        if el:
            if validate is True:
                validate_element(el, validator)

            if element.tag == 'node':
                nodes.append(el['node'])
                node_tags += el['node_tags']
            elif element.tag == 'way':
                way.append(el['way'])
                way_nodes += el['way_nodes']
                way_tags += el['way_tags']
                
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(NODES_PATH, index=False)

    node_tags_df = pd.DataFrame(node_tags)
    node_tags_df.to_csv(NODE_TAGS_PATH, index=False)

    way_df = pd.DataFrame(way)
    way_df.to_csv(WAYS_PATH, index=False)

    way_nodes_df = pd.DataFrame(way_nodes)
    way_nodes_df.to_csv(WAY_NODES_PATH, index=False)

    way_tags_df = pd.DataFrame(way_tags)
    way_tags_df.to_csv(WAY_TAGS_PATH, index=False)
    
process_map(OSM_PATH, validate=True)

