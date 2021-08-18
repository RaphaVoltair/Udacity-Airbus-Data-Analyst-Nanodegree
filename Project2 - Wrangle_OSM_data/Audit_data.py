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

#Audit phone numbers

phone_re = re.compile(r'^\+49\s\d{3}\s\d{6,8}$')

def audit_phone(phone_types, number):
    good_format = phone_re.search(number)
    if not good_format:
        phone_types.add(number)


def is_phone_number(elem):
    return (elem.attrib['k'] == 'phone')

#Audit post codes

postcode_re = re.compile(r'^\d{5}$')


# https://www.muenchen.de/int/en/living/postal-codes.html
expected = ['80995', '80997', '80999', '81247', '81249', '80331', '80333', '80335', '80336', '80469', '80538', '80539', 
            '81541', '81543', '81667', '81669', '81671', '81675', '81677', '81243', '81245', '81249', '81671', '81673', 
            '81735', '81825', '81675', '81677', '81679', '81925', '81927', '81929', '80933', '80935', '80995', '80689', 
            '81375', '81377', '80686', '80687', '80689', '80335', '80336', '80337', '80469', '80333', '80335', '80539', 
            '80636', '80797', '80798', '80799', '80801', '80802', '80807', '80809', '80937', '80939', '80637', '80638', 
            '80992', '80993', '80997', '80634', '80636', '80637', '80638', '80639', '81539', '81541', '81547', '81549', 
            '80687', '80689', '81241', '81243', '81245', '81247', '81539', '81549', '81669', '81671', '81735', '81737', 
            '81739', '80538', '80801', '80802', '80803', '80804', '80805', '80807', '80939', '80796', '80797', '80798', 
            '80799', '80801', '80803', '80804', '80809', '80335', '80339', '80336', '80337', '80469', '81369', '81371', 
            '81373', '81379', '80686', '81369', '81373', '81377', '81379', '81379', '81475', '81476', '81477', '81479', 
            '81735', '81825', '81827', '81829', '81543', '81545', '81547']

def audit_postcode(bad_postcodes, postcode):
    m = postcode_re.search(postcode)
    if m:
        postcode = m.group()
        if postcode not in expected:
            bad_postcodes[postcode] +=1
    else:
        bad_postcodes[postcode] +=1


def is_postcode(elem):
    return (elem.attrib['k'] == 'addr:postcode')

#Audit street types

pattern = r'''
^(Am\b|Auf\sdem\b|Auf\sder\b|Im\b|In\sder\b|An\b|Zum\b|Zur\b)|
(straße|\bStraße|-Straße|\bWeg|weg|-Weg|platz|-Platz|\bPlatz|gasse|
\bRing|ring|\bHof|hof|\bAllee|-Allee|allee|wall|\bWall|markt|gürtel|feld)$'''

street_type_re = re.compile(pattern, re.VERBOSE)


# Count occurences for each street type
def count_street_type(street_type_count, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        street_type_count[street_type] +=1
    else:
        street_type_count[street_name] +=1


# Group streets by type
def group_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        street_types[street_type].add(street_name)
    else:
        street_types[street_name].add(street_name)

# Sorts the count of street types in descending order
def sort_values(street_type_count):
    sorted_counts = []
    d_view = [(v,k) for k,v in street_type_count.items()]
    d_view.sort(reverse=True) # natively sort tuples by first element
    for v,k in d_view:
        sorted_counts.append("%s: %d" % (k,v))
    return sorted_counts


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

# perform all of the above audits on the file
def audit(osmfile):

    phone_types = set()
    bad_postcodes = defaultdict(int)
    street_type_count = defaultdict(int)
    street_types = defaultdict(set)
    
    for event, elem in ET.iterparse(osmfile, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    count_street_type(street_type_count, tag.attrib['v'])
                    group_street_type(street_types, tag.attrib['v'])
                if is_phone_number(tag):
                    audit_phone(phone_types,tag.attrib['v'])
                if is_postcode(tag):
                    audit_postcode(bad_postcodes, tag.attrib['v'])

    street_type_count = sort_values(street_type_count)

    return phone_types, street_type_count, street_types, bad_postcodes

pprint(audit(SAMPLE_FILE))

