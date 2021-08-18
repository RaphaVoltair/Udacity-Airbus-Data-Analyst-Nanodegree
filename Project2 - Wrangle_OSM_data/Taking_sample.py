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

#Taking sample from OSM file:
#creating systematic sample elements from original OSM-file (!Ref. "Project Details"!)

#import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSM_FILE = "OSM_munich.osm"  # Replace this with your osm file
SAMPLE_FILE = "sample_file_munich.osm"

k = 6 # Parameter: take every k-th top level element

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag
    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'wb') as output:
    output.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(b'<osm>\n  ')
    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))
    
    output.write(b'</osm>')

#Check sizes of OSM- and Sample-Files:
#resource: http://stackoverflow.com/questions/2104080/how-to-check-file-size-in-python

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

#check size of sample file

OSM_size = file_size(OSM_FILE)
sample_size = file_size(SAMPLE_FILE)

print ('OSM File Size', OSM_size)
print ('Sample File Size', sample_size)

