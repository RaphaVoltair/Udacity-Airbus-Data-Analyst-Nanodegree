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

# Create database
sqlite_file = 'osm.db'

# Connect to the database
conn = sqlite3.connect(sqlite_file)

# Get a cursor object
cur = conn.cursor()


# Create the SQL-table nodes. Drop table if it already exists

cur.execute(''' DROP TABLE IF EXISTS nodes''')
conn.commit()

cur.execute('''
    CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT
)''')

conn.commit()

# Create the SQL-table nodes_tags. Drop table if it already exists

cur.execute(''' DROP TABLE IF EXISTS nodes_tags''')
conn.commit()

cur.execute('''
    CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id)
)''')

conn.commit()

# Create the SQL-table ways. Drop table if it already exists

cur.execute(''' DROP TABLE IF EXISTS ways''')
conn.commit()
cur.execute('''
CREATE TABLE ways (
    id INTEGER PRIMARY KEY NOT NULL,
    user TEXT,
    uid INTEGER,
    version TEXT,
    changeset INTEGER,
    timestamp TEXT
)''')

conn.commit()

# Create the SQL-table ways_tags. Drop table if it already exists

cur.execute(''' DROP TABLE IF EXISTS ways_tags''')
conn.commit()
cur.execute('''
CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id)
)''')

conn.commit()

# Create the SQL-table ways_nodes. Drop table if it already exists

cur.execute(''' DROP TABLE IF EXISTS ways_nodes''')
conn.commit()
cur.execute('''
CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
)''')

conn.commit()

#Importing to database

# Read in the csv file as a dictionary
# Format the data as a list of tubles:

with open('nodes_tags.csv', 'r', encoding="utf8") as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]

# Insert the formatted data

cur.executemany('INSERT INTO nodes_tags(id, key, value,type) VALUES (?, ?, ?, ?);', to_db)
# commit the changes
conn.commit()


with open('ways_tags.csv', 'r', encoding="utf8") as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]

cur.executemany('INSERT INTO ways_tags(id, key, value,type) VALUES (?, ?, ?, ?);', to_db)

conn.commit()


with open('nodes.csv', 'r', encoding="utf8") as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['lat'], i['lon'], i['user'], i['uid'], i['version'], i['changeset'], i['timestamp'] ) for i in dr]

cur.executemany('INSERT INTO nodes(id, lat, lon, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);', to_db)

conn.commit()


with open('ways.csv', 'r', encoding="utf8") as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['user'], i['uid'], i['version'], i['changeset'], i['timestamp'] ) for i in dr]

cur.executemany('INSERT INTO ways(id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?);', to_db)

conn.commit()


with open('ways_nodes.csv', 'r', encoding="utf8") as fin:
    dr = csv.DictReader(fin) # comma is the default delimiter
    to_db = [(i['id'], i['node_id'], i['position']) for i in dr]

cur.executemany('INSERT INTO ways_nodes(id, node_id, position) VALUES (?, ?, ?);', to_db)

conn.commit()

#Database overview

# list file sizes in the data folder.  
# ref http://stackoverflow.com/questions/10565435/iterating-through-directories-and-checking-the-files-size

cwd = os.getcwd()
for root, dirs, files in os.walk(cwd, topdown=False):
    for name in files:
        f = os.path.join(root, name)
        print(name.ljust(20), round(os.path.getsize(f)/1000000, 2), 'MB')

