# Project: Investigate a Dataset (TMDb movie data)

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

Investigation of TMDb Movie data<br>
dependent variable(or target-variable): popularity, revenue, quantity of released movies<br>
independent (or predictor) variables:genre, budget, popularity

Questions to be answered:<br>

1) What kinds of properties are associated with movies that have high revenues?<br>
2) 3) 4) Which are the most sucessful actors, directors, production companies?<br>
5) Which genres are most popular from year to year?<br>


```python
import pandas as pd
import numpy as np
import unicodecsv
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

### Data Cleaning with Python
At first I tried cleaning the data with python. Looking back it would have probably made more sense to directly start working with pandas. But well, in hindsight one is always wiser.<br>
Anyway, I chose to start by newly reading in the csv.-file in to a python list of dictionaries


```python
tmdbmovies = []
f = open('tmdb-movies.csv', 'rb')
reader = unicodecsv.DictReader(f)
for row in reader:
    tmdbmovies.append(row)
f.close()

print(len(tmdbmovies))

tmdbmovies[:1]
```

    10866
    




    [{'id': '135397',
      'imdb_id': 'tt0369610',
      'popularity': '32.985763',
      'budget': '150000000',
      'revenue': '1513528810',
      'original_title': 'Jurassic World',
      'cast': "Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vincent D'Onofrio|Nick Robinson",
      'homepage': 'http://www.jurassicworld.com/',
      'director': 'Colin Trevorrow',
      'tagline': 'The park is open.',
      'keywords': 'monster|dna|tyrannosaurus rex|velociraptor|island',
      'overview': 'Twenty-two years after the events of Jurassic Park, Isla Nublar now features a fully functioning dinosaur theme park, Jurassic World, as originally envisioned by John Hammond.',
      'runtime': '124',
      'genres': 'Action|Adventure|Science Fiction|Thriller',
      'production_companies': 'Universal Studios|Amblin Entertainment|Legendary Pictures|Fuji Television Network|Dentsu',
      'release_date': '6/9/15',
      'vote_count': '5562',
      'vote_average': '6.5',
      'release_year': '2015',
      'budget_adj': '137999939.280026',
      'revenue_adj': '1392445892.5238'}]



I created a 'list' of 'dictionaries'. Took my while to understand what that means. 


```python
type(tmdbmovies)
```




    list




```python
type(tmdbmovies[0])
```




    dict



Since the types of the dictionary entries are all 'string' I decided to firstly parse everything properly


```python
type(tmdbmovies[0]['budget'])
```




    str




```python
type(tmdbmovies[0]['budget_adj'])
```




    str




```python

def parse_budget_and_revenue(i):
    if i == 0:
        return None
    else:
        return i
```


```python
#parsing string to int
for tmdbmovie in tmdbmovies:
    tmdbmovie['budget'] = int(float(tmdbmovie['budget']))
    tmdbmovie['revenue'] = int(float(tmdbmovie['revenue']))
    tmdbmovie['budget_adj'] = int(float(tmdbmovie['budget_adj']))
    tmdbmovie['revenue_adj'] = int(float(tmdbmovie['revenue_adj']))
```


```python
#replace '0' values as 'None'
for tmdbmovie in tmdbmovies:
    tmdbmovie['budget'] = parse_budget_and_revenue(tmdbmovie['budget'])
    tmdbmovie['revenue'] = parse_budget_and_revenue(tmdbmovie['revenue'])
    tmdbmovie['budget_adj'] = parse_budget_and_revenue(tmdbmovie['budget_adj'])
    tmdbmovie['revenue_adj'] = parse_budget_and_revenue(tmdbmovie['revenue_adj'])
```


```python
for i in tmdbmovies:
    print(i['budget'])
    print(type(i['budget']))
```

    150000000
    <class 'int'>
    150000000
    <class 'int'>
    110000000
    <class 'int'>
    200000000
    <class 'int'>
    190000000
    <class 'int'>
    135000000
    <class 'int'>
    155000000
    <class 'int'>
    108000000
    <class 'int'>
    74000000
    <class 'int'>
    175000000
    <class 'int'>
    245000000
    <class 'int'>
    176000003
    <class 'int'>
    15000000
    <class 'int'>
    88000000
    <class 'int'>
    280000000
    <class 'int'>
    44000000
    <class 'int'>
    48000000
    <class 'int'>
    130000000
    <class 'int'>
    95000000
    <class 'int'>
    160000000
    <class 'int'>
    190000000
    <class 'int'>
    30000000
    <class 'int'>
    110000000
    <class 'int'>
    40000000
    <class 'int'>
    28000000
    <class 'int'>
    150000000
    <class 'int'>
    68000000
    <class 'int'>
    81000000
    <class 'int'>
    20000000
    <class 'int'>
    61000000
    <class 'int'>
    None
    <class 'NoneType'>
    49000000
    <class 'int'>
    29000000
    <class 'int'>
    40000000
    <class 'int'>
    58000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    175000000
    <class 'int'>
    50000000
    <class 'int'>
    11000000
    <class 'int'>
    28000000
    <class 'int'>
    90000000
    <class 'int'>
    30000000
    <class 'int'>
    75000000
    <class 'int'>
    25000000
    <class 'int'>
    10000000
    <class 'int'>
    135000000
    <class 'int'>
    12000000
    <class 'int'>
    30000000
    <class 'int'>
    4000000
    <class 'int'>
    11800000
    <class 'int'>
    35000000
    <class 'int'>
    55000000
    <class 'int'>
    60000000
    <class 'int'>
    105000000
    <class 'int'>
    20000000
    <class 'int'>
    26000000
    <class 'int'>
    60000000
    <class 'int'>
    15000000
    <class 'int'>
    70000000
    <class 'int'>
    30000000
    <class 'int'>
    120000001
    <class 'int'>
    3500000
    <class 'int'>
    65000000
    <class 'int'>
    50100000
    <class 'int'>
    35000000
    <class 'int'>
    100000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    35000000
    <class 'int'>
    13000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    150000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    99000000
    <class 'int'>
    35000000
    <class 'int'>
    25000000
    <class 'int'>
    2500000
    <class 'int'>
    34000000
    <class 'int'>
    80000000
    <class 'int'>
    17000000
    <class 'int'>
    35000000
    <class 'int'>
    11000000
    <class 'int'>
    31000000
    <class 'int'>
    35000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    5000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    64000000
    <class 'int'>
    None
    <class 'NoneType'>
    11930000
    <class 'int'>
    8500000
    <class 'int'>
    10000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    23000000
    <class 'int'>
    15000000
    <class 'int'>
    5000000
    <class 'int'>
    53000000
    <class 'int'>
    8900000
    <class 'int'>
    15000000
    <class 'int'>
    20000000
    <class 'int'>
    5000000
    <class 'int'>
    700000
    <class 'int'>
    28000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    14800000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    19500000
    <class 'int'>
    None
    <class 'NoneType'>
    74000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    1000000
    <class 'int'>
    35000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    1800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    11000000
    <class 'int'>
    15000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    630019
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37000000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    9600000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    15000000
    <class 'int'>
    2240000
    <class 'int'>
    1000000
    <class 'int'>
    13000000
    <class 'int'>
    8000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    3300000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5300000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    447524
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1950000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4400000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    650000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    165000000
    <class 'int'>
    170000000
    <class 'int'>
    170000000
    <class 'int'>
    20000000
    <class 'int'>
    125000000
    <class 'int'>
    250000000
    <class 'int'>
    165000000
    <class 'int'>
    14000000
    <class 'int'>
    34000000
    <class 'int'>
    18000000
    <class 'int'>
    61000000
    <class 'int'>
    68000000
    <class 'int'>
    127000000
    <class 'int'>
    85000000
    <class 'int'>
    250000000
    <class 'int'>
    40000000
    <class 'int'>
    125000000
    <class 'int'>
    40000000
    <class 'int'>
    8500000
    <class 'int'>
    210000000
    <class 'int'>
    30000000
    <class 'int'>
    3300000
    <class 'int'>
    170000000
    <class 'int'>
    55000000
    <class 'int'>
    42000000
    <class 'int'>
    200000000
    <class 'int'>
    178000000
    <class 'int'>
    44000000
    <class 'int'>
    58800000
    <class 'int'>
    140000000
    <class 'int'>
    160000000
    <class 'int'>
    180000000
    <class 'int'>
    15000000
    <class 'int'>
    5000000
    <class 'int'>
    145000000
    <class 'int'>
    2000000
    <class 'int'>
    90000000
    <class 'int'>
    60000000
    <class 'int'>
    13000000
    <class 'int'>
    95000000
    <class 'int'>
    15000000
    <class 'int'>
    17000000
    <class 'int'>
    70000000
    <class 'int'>
    132000000
    <class 'int'>
    50000000
    <class 'int'>
    110000000
    <class 'int'>
    25000000
    <class 'int'>
    50000000
    <class 'int'>
    28000000
    <class 'int'>
    55000000
    <class 'int'>
    7000000
    <class 'int'>
    11000000
    <class 'int'>
    9000000
    <class 'int'>
    16000000
    <class 'int'>
    4000000
    <class 'int'>
    20000000
    <class 'int'>
    100000000
    <class 'int'>
    125000000
    <class 'int'>
    50000000
    <class 'int'>
    12000000
    <class 'int'>
    40000000
    <class 'int'>
    18000000
    <class 'int'>
    11000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    60000000
    <class 'int'>
    50000000
    <class 'int'>
    5500000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    12600000
    <class 'int'>
    120000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    145000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    103000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    65000000
    <class 'int'>
    20000000
    <class 'int'>
    4900000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    130000000
    <class 'int'>
    26000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    13000000
    <class 'int'>
    66000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    60000000
    <class 'int'>
    10000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    25000000
    <class 'int'>
    28000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    13200000
    <class 'int'>
    5000000
    <class 'int'>
    5000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    12250000
    <class 'int'>
    24000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    28000000
    <class 'int'>
    2000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22500000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    19800000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    12500000
    <class 'int'>
    25000000
    <class 'int'>
    5500000
    <class 'int'>
    6700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13300000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    3500000
    <class 'int'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    5500000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    200000
    <class 'int'>
    21000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    30000000
    <class 'int'>
    6600000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    36000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    5000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3150000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5500000
    <class 'int'>
    None
    <class 'NoneType'>
    2700000
    <class 'int'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    318000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    10
    <class 'int'>
    None
    <class 'NoneType'>
    12700000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    325927
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    156660
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    950000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    18
    <class 'int'>
    6900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    150000
    <class 'int'>
    None
    <class 'NoneType'>
    1400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    117
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2489400
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    14000000
    <class 'int'>
    1200000
    <class 'int'>
    4000000
    <class 'int'>
    10000000
    <class 'int'>
    20000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    3400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5500000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    22000000
    <class 'int'>
    810000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    230000
    <class 'int'>
    None
    <class 'NoneType'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    530000
    <class 'int'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    237000000
    <class 'int'>
    70000000
    <class 'int'>
    35000000
    <class 'int'>
    250000000
    <class 'int'>
    175000000
    <class 'int'>
    150000000
    <class 'int'>
    200000000
    <class 'int'>
    30000000
    <class 'int'>
    90000000
    <class 'int'>
    7500000
    <class 'int'>
    90000000
    <class 'int'>
    35000000
    <class 'int'>
    130000000
    <class 'int'>
    200000000
    <class 'int'>
    30000000
    <class 'int'>
    150000000
    <class 'int'>
    105000000
    <class 'int'>
    29000000
    <class 'int'>
    23600000
    <class 'int'>
    50000000
    <class 'int'>
    40000000
    <class 'int'>
    175000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    60000000
    <class 'int'>
    150000000
    <class 'int'>
    20000000
    <class 'int'>
    200000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    26000000
    <class 'int'>
    16000000
    <class 'int'>
    53000000
    <class 'int'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    33000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    26000000
    <class 'int'>
    50000000
    <class 'int'>
    5000000
    <class 'int'>
    150000000
    <class 'int'>
    None
    <class 'NoneType'>
    100000000
    <class 'int'>
    18500000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    70000000
    <class 'int'>
    26000000
    <class 'int'>
    32000000
    <class 'int'>
    175000000
    <class 'int'>
    3000000
    <class 'int'>
    40000000
    <class 'int'>
    15000000
    <class 'int'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    20000000
    <class 'int'>
    15000000
    <class 'int'>
    40000000
    <class 'int'>
    75000000
    <class 'int'>
    4500000
    <class 'int'>
    30000000
    <class 'int'>
    10000000
    <class 'int'>
    60000000
    <class 'int'>
    16000000
    <class 'int'>
    14700000
    <class 'int'>
    41000000
    <class 'int'>
    22000000
    <class 'int'>
    50000000
    <class 'int'>
    27000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    39000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    38000000
    <class 'int'>
    17000000
    <class 'int'>
    11000000
    <class 'int'>
    35000000
    <class 'int'>
    7000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    13000000
    <class 'int'>
    7000000
    <class 'int'>
    20000000
    <class 'int'>
    8000000
    <class 'int'>
    9000000
    <class 'int'>
    150000000
    <class 'int'>
    24000000
    <class 'int'>
    15000000
    <class 'int'>
    40000000
    <class 'int'>
    20000000
    <class 'int'>
    100000000
    <class 'int'>
    85000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    18000000
    <class 'int'>
    100000000
    <class 'int'>
    60000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    8000000
    <class 'int'>
    60000000
    <class 'int'>
    4500000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    87000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    8000000
    <class 'int'>
    8000000
    <class 'int'>
    25000000
    <class 'int'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    7500000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    65000000
    <class 'int'>
    50000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    12000000
    <class 'int'>
    25000000
    <class 'int'>
    40000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    8000000
    <class 'int'>
    45000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3400000
    <class 'int'>
    None
    <class 'NoneType'>
    4300000
    <class 'int'>
    3000000
    <class 'int'>
    10000000
    <class 'int'>
    6000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    800000
    <class 'int'>
    78146652
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    5100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    9500000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    6300000
    <class 'int'>
    17000000
    <class 'int'>
    17000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    6000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    7300000
    <class 'int'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    3700000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    15000000
    <class 'int'>
    4500000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    6400000
    <class 'int'>
    5060730
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    150000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2900000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    425000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    25000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    10000000
    <class 'int'>
    2011799
    <class 'int'>
    None
    <class 'NoneType'>
    100000000
    <class 'int'>
    8500000
    <class 'int'>
    20000000
    <class 'int'>
    12500000
    <class 'int'>
    5000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    1700000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    5000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    108
    <class 'int'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31192
    <class 'int'>
    3250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    160000000
    <class 'int'>
    200000000
    <class 'int'>
    200000000
    <class 'int'>
    13000000
    <class 'int'>
    250000000
    <class 'int'>
    69000000
    <class 'int'>
    80000000
    <class 'int'>
    165000000
    <class 'int'>
    80000000
    <class 'int'>
    170000000
    <class 'int'>
    260000000
    <class 'int'>
    200000000
    <class 'int'>
    110000000
    <class 'int'>
    18000000
    <class 'int'>
    8000000
    <class 'int'>
    155000000
    <class 'int'>
    28000000
    <class 'int'>
    100000000
    <class 'int'>
    150000000
    <class 'int'>
    200000000
    <class 'int'>
    38000000
    <class 'int'>
    125000000
    <class 'int'>
    150000000
    <class 'int'>
    30000000
    <class 'int'>
    40000000
    <class 'int'>
    60000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    95000000
    <class 'int'>
    100000000
    <class 'int'>
    165000000
    <class 'int'>
    25000000
    <class 'int'>
    110000000
    <class 'int'>
    58000000
    <class 'int'>
    20000000
    <class 'int'>
    80000000
    <class 'int'>
    80000000
    <class 'int'>
    55000000
    <class 'int'>
    37000000
    <class 'int'>
    20000000
    <class 'int'>
    68000000
    <class 'int'>
    130000000
    <class 'int'>
    150000000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    30000000
    <class 'int'>
    15000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    32000000
    <class 'int'>
    7000000
    <class 'int'>
    117000000
    <class 'int'>
    3000000
    <class 'int'>
    60000000
    <class 'int'>
    1000000
    <class 'int'>
    40000000
    <class 'int'>
    19000000
    <class 'int'>
    100000000
    <class 'int'>
    112000000
    <class 'int'>
    35000000
    <class 'int'>
    44000000
    <class 'int'>
    19000000
    <class 'int'>
    10000000
    <class 'int'>
    52000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    40000000
    <class 'int'>
    65000000
    <class 'int'>
    8000000
    <class 'int'>
    52000000
    <class 'int'>
    45000000
    <class 'int'>
    8000000
    <class 'int'>
    30000000
    <class 'int'>
    35000000
    <class 'int'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    20000000
    <class 'int'>
    2000000
    <class 'int'>
    75000000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    100000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    36000000
    <class 'int'>
    21800000
    <class 'int'>
    20000000
    <class 'int'>
    14000000
    <class 'int'>
    80000000
    <class 'int'>
    32000000
    <class 'int'>
    55000000
    <class 'int'>
    15000000
    <class 'int'>
    32000000
    <class 'int'>
    24000000
    <class 'int'>
    20000000
    <class 'int'>
    3000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    11000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    12000000
    <class 'int'>
    3200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    22000000
    <class 'int'>
    15000000
    <class 'int'>
    2000000
    <class 'int'>
    24000000
    <class 'int'>
    40000000
    <class 'int'>
    12500000
    <class 'int'>
    90000000
    <class 'int'>
    1800000
    <class 'int'>
    48000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    1987650
    <class 'int'>
    69000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    15000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    150000000
    <class 'int'>
    None
    <class 'NoneType'>
    120000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    13000000
    <class 'int'>
    25000000
    <class 'int'>
    50000000
    <class 'int'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    4500000
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    5000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000
    <class 'int'>
    5000000
    <class 'int'>
    8000000
    <class 'int'>
    20000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5773100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    2500000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    5800000
    <class 'int'>
    4466000
    <class 'int'>
    None
    <class 'NoneType'>
    3100000
    <class 'int'>
    None
    <class 'NoneType'>
    1400000
    <class 'int'>
    None
    <class 'NoneType'>
    550000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    104002432
    <class 'int'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4600000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000
    <class 'int'>
    35000000
    <class 'int'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    120000000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    30000
    <class 'int'>
    3000000
    <class 'int'>
    65000
    <class 'int'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    650000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    425000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    967686
    <class 'int'>
    None
    <class 'NoneType'>
    7347125
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    2300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3167000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    1250000
    <class 'int'>
    None
    <class 'NoneType'>
    10
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000
    <class 'int'>
    None
    <class 'NoneType'>
    3
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    63000000
    <class 'int'>
    63000000
    <class 'int'>
    15000000
    <class 'int'>
    115000000
    <class 'int'>
    80000000
    <class 'int'>
    60000000
    <class 'int'>
    40000000
    <class 'int'>
    135000000
    <class 'int'>
    10500000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    66000000
    <class 'int'>
    90000000
    <class 'int'>
    16000000
    <class 'int'>
    42000000
    <class 'int'>
    33000000
    <class 'int'>
    2000000
    <class 'int'>
    90000000
    <class 'int'>
    45000000
    <class 'int'>
    65000000
    <class 'int'>
    13000000
    <class 'int'>
    80000000
    <class 'int'>
    133000000
    <class 'int'>
    73000000
    <class 'int'>
    170000000
    <class 'int'>
    160000000
    <class 'int'>
    100000000
    <class 'int'>
    24000000
    <class 'int'>
    70000000
    <class 'int'>
    60000000
    <class 'int'>
    21000000
    <class 'int'>
    90000000
    <class 'int'>
    70000000
    <class 'int'>
    10000000
    <class 'int'>
    70000000
    <class 'int'>
    6000000
    <class 'int'>
    10000000
    <class 'int'>
    6000000
    <class 'int'>
    100000000
    <class 'int'>
    25000
    <class 'int'>
    38000000
    <class 'int'>
    50000000
    <class 'int'>
    16000000
    <class 'int'>
    34200000
    <class 'int'>
    32000000
    <class 'int'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    75000000
    <class 'int'>
    35000000
    <class 'int'>
    80000000
    <class 'int'>
    80000000
    <class 'int'>
    60000000
    <class 'int'>
    29000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    37000000
    <class 'int'>
    51000000
    <class 'int'>
    75000000
    <class 'int'>
    40000000
    <class 'int'>
    27000000
    <class 'int'>
    34000000
    <class 'int'>
    68000000
    <class 'int'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    6500000
    <class 'int'>
    25000000
    <class 'int'>
    80
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    82000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    48000000
    <class 'int'>
    64000000
    <class 'int'>
    15000000
    <class 'int'>
    23000000
    <class 'int'>
    65000000
    <class 'int'>
    21000000
    <class 'int'>
    21500000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    25000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    33000000
    <class 'int'>
    5000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    23000000
    <class 'int'>
    75000000
    <class 'int'>
    65000000
    <class 'int'>
    11000000
    <class 'int'>
    10000000
    <class 'int'>
    25
    <class 'int'>
    50000000
    <class 'int'>
    8000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    10000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    55000000
    <class 'int'>
    38000000
    <class 'int'>
    15000000
    <class 'int'>
    600000
    <class 'int'>
    7000000
    <class 'int'>
    55000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    6000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    28000000
    <class 'int'>
    50000000
    <class 'int'>
    21000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    12000000
    <class 'int'>
    75000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    60000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    24000000
    <class 'int'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    1700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4250000
    <class 'int'>
    2500000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    450000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3800000
    <class 'int'>
    1
    <class 'int'>
    9210000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    93000000
    <class 'int'>
    125000000
    <class 'int'>
    15000000
    <class 'int'>
    6000000
    <class 'int'>
    115000000
    <class 'int'>
    100000000
    <class 'int'>
    60000000
    <class 'int'>
    25000000
    <class 'int'>
    85000000
    <class 'int'>
    140000000
    <class 'int'>
    98000000
    <class 'int'>
    120000000
    <class 'int'>
    115000000
    <class 'int'>
    60000000
    <class 'int'>
    30000000
    <class 'int'>
    37000000
    <class 'int'>
    52500000
    <class 'int'>
    100000000
    <class 'int'>
    21000000
    <class 'int'>
    53000000
    <class 'int'>
    70000000
    <class 'int'>
    92000000
    <class 'int'>
    18000000
    <class 'int'>
    65000000
    <class 'int'>
    28000000
    <class 'int'>
    16000000
    <class 'int'>
    45000000
    <class 'int'>
    90000000
    <class 'int'>
    45000000
    <class 'int'>
    17000000
    <class 'int'>
    25000000
    <class 'int'>
    22000000
    <class 'int'>
    28000000
    <class 'int'>
    60000000
    <class 'int'>
    68000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    92000000
    <class 'int'>
    102000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    137000000
    <class 'int'>
    None
    <class 'NoneType'>
    42000000
    <class 'int'>
    70000000
    <class 'int'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    72000000
    <class 'int'>
    60000000
    <class 'int'>
    3000000
    <class 'int'>
    35000000
    <class 'int'>
    48000000
    <class 'int'>
    19800000
    <class 'int'>
    11000000
    <class 'int'>
    57000000
    <class 'int'>
    23000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    57000000
    <class 'int'>
    None
    <class 'NoneType'>
    48000000
    <class 'int'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    21150000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    53000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    31000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    80000000
    <class 'int'>
    10000000
    <class 'int'>
    35000000
    <class 'int'>
    107000000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    68000000
    <class 'int'>
    4000000
    <class 'int'>
    17700000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    21000000
    <class 'int'>
    40000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    72000000
    <class 'int'>
    62000000
    <class 'int'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    2079000
    <class 'int'>
    28000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    48000000
    <class 'int'>
    10000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    49000000
    <class 'int'>
    15000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    6000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1700000
    <class 'int'>
    35000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    22000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    94000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    87000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    12000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    93000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    57000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    7000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    15000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    4800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    165000
    <class 'int'>
    185000000
    <class 'int'>
    180000000
    <class 'int'>
    140000000
    <class 'int'>
    25000000
    <class 'int'>
    185000000
    <class 'int'>
    37000000
    <class 'int'>
    200000000
    <class 'int'>
    130000000
    <class 'int'>
    19000000
    <class 'int'>
    52000000
    <class 'int'>
    150000000
    <class 'int'>
    145000000
    <class 'int'>
    150000000
    <class 'int'>
    33000000
    <class 'int'>
    150000000
    <class 'int'>
    85000000
    <class 'int'>
    85000000
    <class 'int'>
    45000000
    <class 'int'>
    75000000
    <class 'int'>
    48000000
    <class 'int'>
    15000000
    <class 'int'>
    30000000
    <class 'int'>
    75000000
    <class 'int'>
    105000000
    <class 'int'>
    150000000
    <class 'int'>
    15000000
    <class 'int'>
    35000000
    <class 'int'>
    225000000
    <class 'int'>
    92000000
    <class 'int'>
    80000000
    <class 'int'>
    12500000
    <class 'int'>
    50000000
    <class 'int'>
    45000000
    <class 'int'>
    23000000
    <class 'int'>
    80000000
    <class 'int'>
    9000000
    <class 'int'>
    55000000
    <class 'int'>
    25000000
    <class 'int'>
    150000000
    <class 'int'>
    30000000
    <class 'int'>
    130000000
    <class 'int'>
    70000000
    <class 'int'>
    60000000
    <class 'int'>
    70000000
    <class 'int'>
    37000000
    <class 'int'>
    30000000
    <class 'int'>
    60000000
    <class 'int'>
    35000000
    <class 'int'>
    20000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    20000000
    <class 'int'>
    27000000
    <class 'int'>
    6000000
    <class 'int'>
    80000000
    <class 'int'>
    20500000
    <class 'int'>
    35000000
    <class 'int'>
    35000000
    <class 'int'>
    40000000
    <class 'int'>
    24000000
    <class 'int'>
    80000000
    <class 'int'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    55000000
    <class 'int'>
    90000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    9000000
    <class 'int'>
    80000000
    <class 'int'>
    5000000
    <class 'int'>
    7000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    10800000
    <class 'int'>
    62000000
    <class 'int'>
    120000000
    <class 'int'>
    15000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    15000000
    <class 'int'>
    22000000
    <class 'int'>
    90000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    7000000
    <class 'int'>
    20000000
    <class 'int'>
    50000000
    <class 'int'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    40000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    18000000
    <class 'int'>
    12000000
    <class 'int'>
    13000000
    <class 'int'>
    45000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    40000000
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    11500000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    18000000
    <class 'int'>
    60000000
    <class 'int'>
    30000000
    <class 'int'>
    12000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    8
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    6100000
    <class 'int'>
    35000000
    <class 'int'>
    11000000
    <class 'int'>
    10000000
    <class 'int'>
    15000000
    <class 'int'>
    20000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    8000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    1
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    12000000
    <class 'int'>
    25000000
    <class 'int'>
    27000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    16000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    37000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    150000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    14100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8376800
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    6000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15400000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    11500000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    20000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    500000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    200000
    <class 'int'>
    6700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    4300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5750000
    <class 'int'>
    9100000
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    200000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3600000
    <class 'int'>
    None
    <class 'NoneType'>
    515788
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4180000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4600000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    170000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    140000000
    <class 'int'>
    15000000
    <class 'int'>
    125000000
    <class 'int'>
    380000000
    <class 'int'>
    93000000
    <class 'int'>
    150000000
    <class 'int'>
    40000000
    <class 'int'>
    50000000
    <class 'int'>
    40000000
    <class 'int'>
    32000000
    <class 'int'>
    145000000
    <class 'int'>
    50000000
    <class 'int'>
    90000000
    <class 'int'>
    27000000
    <class 'int'>
    150000000
    <class 'int'>
    80000000
    <class 'int'>
    35000000
    <class 'int'>
    110000000
    <class 'int'>
    35000000
    <class 'int'>
    20000000
    <class 'int'>
    125000000
    <class 'int'>
    200000000
    <class 'int'>
    200000000
    <class 'int'>
    80000000
    <class 'int'>
    25000000
    <class 'int'>
    130000000
    <class 'int'>
    32500000
    <class 'int'>
    36000000
    <class 'int'>
    50000000
    <class 'int'>
    82000000
    <class 'int'>
    42000000
    <class 'int'>
    110000000
    <class 'int'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    32000000
    <class 'int'>
    50000000
    <class 'int'>
    90000000
    <class 'int'>
    15000000
    <class 'int'>
    80000000
    <class 'int'>
    163000000
    <class 'int'>
    None
    <class 'NoneType'>
    66000000
    <class 'int'>
    8000000
    <class 'int'>
    135000000
    <class 'int'>
    52000000
    <class 'int'>
    170000000
    <class 'int'>
    40000000
    <class 'int'>
    57000000
    <class 'int'>
    15000000
    <class 'int'>
    66000000
    <class 'int'>
    37000000
    <class 'int'>
    30000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    25000000
    <class 'int'>
    35000000
    <class 'int'>
    30000000
    <class 'int'>
    50200000
    <class 'int'>
    75000000
    <class 'int'>
    250000
    <class 'int'>
    30000000
    <class 'int'>
    40000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    70000000
    <class 'int'>
    47000000
    <class 'int'>
    30000000
    <class 'int'>
    79000000
    <class 'int'>
    20000000
    <class 'int'>
    40000000
    <class 'int'>
    6400000
    <class 'int'>
    6000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    50000000
    <class 'int'>
    7000000
    <class 'int'>
    16000000
    <class 'int'>
    130000000
    <class 'int'>
    30000000
    <class 'int'>
    120000000
    <class 'int'>
    35000000
    <class 'int'>
    90000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    41000000
    <class 'int'>
    17000000
    <class 'int'>
    19000000
    <class 'int'>
    35000000
    <class 'int'>
    75000000
    <class 'int'>
    56000000
    <class 'int'>
    49900000
    <class 'int'>
    3500000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    17000000
    <class 'int'>
    130000000
    <class 'int'>
    40000000
    <class 'int'>
    150000000
    <class 'int'>
    14350531
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    21000000
    <class 'int'>
    25000000
    <class 'int'>
    10000000
    <class 'int'>
    110000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12468389
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    3500000
    <class 'int'>
    36000000
    <class 'int'>
    15774948
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    63000000
    <class 'int'>
    45000000
    <class 'int'>
    10000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    10000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    36000000
    <class 'int'>
    25000000
    <class 'int'>
    100000
    <class 'int'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    195000000
    <class 'int'>
    5000000
    <class 'int'>
    13000000
    <class 'int'>
    70000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    45000000
    <class 'int'>
    27000000
    <class 'int'>
    9500000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    750000
    <class 'int'>
    8000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    7700000
    <class 'int'>
    35000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    15000000
    <class 'int'>
    18000000
    <class 'int'>
    28000000
    <class 'int'>
    4798235
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    38000000
    <class 'int'>
    50000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    1
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4317946
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    1000000
    <class 'int'>
    350000
    <class 'int'>
    3500000
    <class 'int'>
    93
    <class 'int'>
    None
    <class 'NoneType'>
    37000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    29000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    125000000
    <class 'int'>
    500000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7300000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    125000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    10831173
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    135000
    <class 'int'>
    None
    <class 'NoneType'>
    930000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    500000
    <class 'int'>
    19100000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    8400000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    1150000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    41000000
    <class 'int'>
    3000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6965576
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    25000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    134005
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8978040
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3
    <class 'int'>
    13000000
    <class 'int'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4200000
    <class 'int'>
    650000
    <class 'int'>
    17000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45202
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    160000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    1340000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    97
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2902660
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    61733
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    79000000
    <class 'int'>
    100000000
    <class 'int'>
    59000000
    <class 'int'>
    60000000
    <class 'int'>
    33000000
    <class 'int'>
    139000000
    <class 'int'>
    120000000
    <class 'int'>
    52000000
    <class 'int'>
    140000000
    <class 'int'>
    35000000
    <class 'int'>
    102000000
    <class 'int'>
    80000000
    <class 'int'>
    21000000
    <class 'int'>
    70000000
    <class 'int'>
    140000000
    <class 'int'>
    140000000
    <class 'int'>
    5000000
    <class 'int'>
    55000000
    <class 'int'>
    54000000
    <class 'int'>
    100000000
    <class 'int'>
    48000000
    <class 'int'>
    120000000
    <class 'int'>
    5000000
    <class 'int'>
    41000000
    <class 'int'>
    50000000
    <class 'int'>
    3500159
    <class 'int'>
    5000000
    <class 'int'>
    60000000
    <class 'int'>
    72000000
    <class 'int'>
    48000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    5000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    85000000
    <class 'int'>
    13000000
    <class 'int'>
    60000000
    <class 'int'>
    20000000
    <class 'int'>
    8000000
    <class 'int'>
    25000000
    <class 'int'>
    80000000
    <class 'int'>
    63000000
    <class 'int'>
    65000000
    <class 'int'>
    46000000
    <class 'int'>
    35000000
    <class 'int'>
    84000000
    <class 'int'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    70000000
    <class 'int'>
    4000000
    <class 'int'>
    15000000
    <class 'int'>
    1300000
    <class 'int'>
    None
    <class 'NoneType'>
    68
    <class 'int'>
    28
    <class 'int'>
    75000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    50000000
    <class 'int'>
    19000000
    <class 'int'>
    30000000
    <class 'int'>
    12000000
    <class 'int'>
    15000000
    <class 'int'>
    80000000
    <class 'int'>
    20000000
    <class 'int'>
    43000000
    <class 'int'>
    None
    <class 'NoneType'>
    115000000
    <class 'int'>
    36000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    45000000
    <class 'int'>
    25000000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    20000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    14000000
    <class 'int'>
    12000000
    <class 'int'>
    7000000
    <class 'int'>
    45000000
    <class 'int'>
    12000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    42000000
    <class 'int'>
    21000000
    <class 'int'>
    70000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    50000000
    <class 'int'>
    47000000
    <class 'int'>
    10000000
    <class 'int'>
    35000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    33000000
    <class 'int'>
    13500000
    <class 'int'>
    20000000
    <class 'int'>
    70000000
    <class 'int'>
    5000000
    <class 'int'>
    30000000
    <class 'int'>
    4000000
    <class 'int'>
    3000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    2000000
    <class 'int'>
    26000000
    <class 'int'>
    13000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    3800000
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    6000000
    <class 'int'>
    27000000
    <class 'int'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    11000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    12000000
    <class 'int'>
    100000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    11000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    78000000
    <class 'int'>
    None
    <class 'NoneType'>
    29000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    200000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    7000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15600000
    <class 'int'>
    3000000
    <class 'int'>
    2000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000
    <class 'int'>
    8000000
    <class 'int'>
    25000000
    <class 'int'>
    55000000
    <class 'int'>
    45000000
    <class 'int'>
    30000000
    <class 'int'>
    17000000
    <class 'int'>
    30000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    115000000
    <class 'int'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    62000000
    <class 'int'>
    34000000
    <class 'int'>
    27000
    <class 'int'>
    50000000
    <class 'int'>
    22000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    15000000
    <class 'int'>
    40000000
    <class 'int'>
    46000000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    60000000
    <class 'int'>
    14000000
    <class 'int'>
    25000000
    <class 'int'>
    5000000
    <class 'int'>
    35000000
    <class 'int'>
    27000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    18000000
    <class 'int'>
    27000000
    <class 'int'>
    63000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    35000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    26000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    45000000
    <class 'int'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    17000000
    <class 'int'>
    15000000
    <class 'int'>
    26000000
    <class 'int'>
    30000000
    <class 'int'>
    11500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    20000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    17080000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    400000
    <class 'int'>
    2000000
    <class 'int'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    7400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    32
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    220000000
    <class 'int'>
    70000000
    <class 'int'>
    250000000
    <class 'int'>
    100000000
    <class 'int'>
    200000000
    <class 'int'>
    130000000
    <class 'int'>
    250000000
    <class 'int'>
    130000000
    <class 'int'>
    21000000
    <class 'int'>
    215000000
    <class 'int'>
    185000000
    <class 'int'>
    170000000
    <class 'int'>
    42000000
    <class 'int'>
    120000000
    <class 'int'>
    95000000
    <class 'int'>
    50000000
    <class 'int'>
    60000000
    <class 'int'>
    42000000
    <class 'int'>
    165000000
    <class 'int'>
    17000000
    <class 'int'>
    225000000
    <class 'int'>
    75000000
    <class 'int'>
    125000000
    <class 'int'>
    102000000
    <class 'int'>
    100000000
    <class 'int'>
    45000000
    <class 'int'>
    150000000
    <class 'int'>
    13000000
    <class 'int'>
    44500000
    <class 'int'>
    50000000
    <class 'int'>
    7000000
    <class 'int'>
    30000000
    <class 'int'>
    145000000
    <class 'int'>
    30000000
    <class 'int'>
    79000000
    <class 'int'>
    12500000
    <class 'int'>
    30000000
    <class 'int'>
    69000000
    <class 'int'>
    60000000
    <class 'int'>
    3500000
    <class 'int'>
    70000000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    209000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    61000000
    <class 'int'>
    85000000
    <class 'int'>
    260000000
    <class 'int'>
    30000000
    <class 'int'>
    120000000
    <class 'int'>
    40000000
    <class 'int'>
    23000000
    <class 'int'>
    150000000
    <class 'int'>
    33000000
    <class 'int'>
    15000000
    <class 'int'>
    12000000
    <class 'int'>
    85000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    40000000
    <class 'int'>
    65000000
    <class 'int'>
    7500000
    <class 'int'>
    25000000
    <class 'int'>
    12000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    145000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    75000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    68000000
    <class 'int'>
    60000000
    <class 'int'>
    32000000
    <class 'int'>
    25000000
    <class 'int'>
    30000000
    <class 'int'>
    35000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    17000000
    <class 'int'>
    39000000
    <class 'int'>
    70000000
    <class 'int'>
    85000000
    <class 'int'>
    45000000
    <class 'int'>
    35000000
    <class 'int'>
    3730500
    <class 'int'>
    None
    <class 'NoneType'>
    31000000
    <class 'int'>
    42000000
    <class 'int'>
    15000000
    <class 'int'>
    40000000
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    10000000
    <class 'int'>
    16000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    20000000
    <class 'int'>
    15000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    14000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    6900000
    <class 'int'>
    45000000
    <class 'int'>
    40000000
    <class 'int'>
    11000000
    <class 'int'>
    26350000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    15000000
    <class 'int'>
    22000000
    <class 'int'>
    19500000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    12000000
    <class 'int'>
    58000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    35000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    61000000
    <class 'int'>
    6000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    16600000
    <class 'int'>
    5000000
    <class 'int'>
    13000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    11500000
    <class 'int'>
    5000000
    <class 'int'>
    5600000
    <class 'int'>
    2400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    850000
    <class 'int'>
    9500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    5000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    18200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    8000000
    <class 'int'>
    150000
    <class 'int'>
    1800000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4275000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    20000000
    <class 'int'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    12500000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    20500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    5300000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    5000000
    <class 'int'>
    1700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    17
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    170000
    <class 'int'>
    1250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5990000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2840000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10400
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    400000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8510000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    110
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    94000000
    <class 'int'>
    22000000
    <class 'int'>
    140000000
    <class 'int'>
    30000000
    <class 'int'>
    150000000
    <class 'int'>
    150000000
    <class 'int'>
    94000000
    <class 'int'>
    80000000
    <class 'int'>
    40000000
    <class 'int'>
    32000000
    <class 'int'>
    200000000
    <class 'int'>
    15000000
    <class 'int'>
    70000000
    <class 'int'>
    78000000
    <class 'int'>
    55000000
    <class 'int'>
    78000000
    <class 'int'>
    60000000
    <class 'int'>
    20000000
    <class 'int'>
    4000000
    <class 'int'>
    23000000
    <class 'int'>
    40000000
    <class 'int'>
    100000000
    <class 'int'>
    100000000
    <class 'int'>
    48000000
    <class 'int'>
    18000000
    <class 'int'>
    36000000
    <class 'int'>
    35000000
    <class 'int'>
    140000000
    <class 'int'>
    130000000
    <class 'int'>
    95000000
    <class 'int'>
    12600000
    <class 'int'>
    30000000
    <class 'int'>
    150000000
    <class 'int'>
    65000000
    <class 'int'>
    25000000
    <class 'int'>
    5000000
    <class 'int'>
    75000000
    <class 'int'>
    60000000
    <class 'int'>
    137000000
    <class 'int'>
    29000000
    <class 'int'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    50000000
    <class 'int'>
    50000000
    <class 'int'>
    80000000
    <class 'int'>
    70000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    60000000
    <class 'int'>
    80000000
    <class 'int'>
    60000000
    <class 'int'>
    120000000
    <class 'int'>
    80000000
    <class 'int'>
    90000000
    <class 'int'>
    40000000
    <class 'int'>
    40000000
    <class 'int'>
    20000000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    9500000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    79000000
    <class 'int'>
    6000000
    <class 'int'>
    20000000
    <class 'int'>
    41000000
    <class 'int'>
    2000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    50000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    76000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    27440000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    87000000
    <class 'int'>
    50000000
    <class 'int'>
    17000000
    <class 'int'>
    17000000
    <class 'int'>
    68000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    52000000
    <class 'int'>
    38000000
    <class 'int'>
    35000000
    <class 'int'>
    35000000
    <class 'int'>
    60000000
    <class 'int'>
    55000000
    <class 'int'>
    16500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    60000000
    <class 'int'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    30000000
    <class 'int'>
    54000000
    <class 'int'>
    56000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    5000000
    <class 'int'>
    35000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    20000000
    <class 'int'>
    14000000
    <class 'int'>
    32000000
    <class 'int'>
    6400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    7800000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    850000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    780000
    <class 'int'>
    80000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    8256269
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    4361898
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6800000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10
    <class 'int'>
    20000000
    <class 'int'>
    15000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    110000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    569217
    <class 'int'>
    5000000
    <class 'int'>
    300
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    16700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    200000000
    <class 'int'>
    85000000
    <class 'int'>
    90000000
    <class 'int'>
    36000000
    <class 'int'>
    90000000
    <class 'int'>
    125000000
    <class 'int'>
    3500000
    <class 'int'>
    110000000
    <class 'int'>
    10000000
    <class 'int'>
    12000000
    <class 'int'>
    70000000
    <class 'int'>
    50000000
    <class 'int'>
    60000000
    <class 'int'>
    53000000
    <class 'int'>
    32000000
    <class 'int'>
    16500000
    <class 'int'>
    105000000
    <class 'int'>
    3500000
    <class 'int'>
    75000000
    <class 'int'>
    80000000
    <class 'int'>
    35000000
    <class 'int'>
    17000000
    <class 'int'>
    57000000
    <class 'int'>
    45000000
    <class 'int'>
    250000
    <class 'int'>
    116000000
    <class 'int'>
    160000000
    <class 'int'>
    35000000
    <class 'int'>
    24000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    90000000
    <class 'int'>
    62000000
    <class 'int'>
    40000000
    <class 'int'>
    90000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    68000000
    <class 'int'>
    75000000
    <class 'int'>
    85000000
    <class 'int'>
    55000000
    <class 'int'>
    15000000
    <class 'int'>
    50000000
    <class 'int'>
    36000000
    <class 'int'>
    70000000
    <class 'int'>
    38000000
    <class 'int'>
    15000000
    <class 'int'>
    15000000
    <class 'int'>
    50000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    90000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    50000000
    <class 'int'>
    25000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    50000000
    <class 'int'>
    250000
    <class 'int'>
    28000000
    <class 'int'>
    80000000
    <class 'int'>
    50000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    85000000
    <class 'int'>
    60000000
    <class 'int'>
    30000000
    <class 'int'>
    38000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    5000000
    <class 'int'>
    15000000
    <class 'int'>
    35000000
    <class 'int'>
    20000000
    <class 'int'>
    25000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    35000000
    <class 'int'>
    48000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    29000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    17000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    53000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    36000000
    <class 'int'>
    15000000
    <class 'int'>
    6000000
    <class 'int'>
    30000000
    <class 'int'>
    73000000
    <class 'int'>
    37000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    1300000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    786675
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    150000000
    <class 'int'>
    105000000
    <class 'int'>
    170000000
    <class 'int'>
    200000000
    <class 'int'>
    75000000
    <class 'int'>
    130000000
    <class 'int'>
    100000000
    <class 'int'>
    180000000
    <class 'int'>
    120000000
    <class 'int'>
    250000000
    <class 'int'>
    225000000
    <class 'int'>
    23000000
    <class 'int'>
    76000000
    <class 'int'>
    190000000
    <class 'int'>
    20000000
    <class 'int'>
    38000000
    <class 'int'>
    13000000
    <class 'int'>
    105000000
    <class 'int'>
    None
    <class 'NoneType'>
    115000000
    <class 'int'>
    92000000
    <class 'int'>
    3000000
    <class 'int'>
    200000000
    <class 'int'>
    200000000
    <class 'int'>
    90000000
    <class 'int'>
    110000000
    <class 'int'>
    37000000
    <class 'int'>
    61000000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    130000000
    <class 'int'>
    11000000
    <class 'int'>
    70000000
    <class 'int'>
    90000000
    <class 'int'>
    46000000
    <class 'int'>
    20000000
    <class 'int'>
    40000000
    <class 'int'>
    35000000
    <class 'int'>
    6000000
    <class 'int'>
    50000000
    <class 'int'>
    175000000
    <class 'int'>
    3500000
    <class 'int'>
    60000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    130000000
    <class 'int'>
    84000000
    <class 'int'>
    50000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    17000000
    <class 'int'>
    28000000
    <class 'int'>
    30000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    80000000
    <class 'int'>
    103000000
    <class 'int'>
    135000000
    <class 'int'>
    22000000
    <class 'int'>
    50000000
    <class 'int'>
    30000000
    <class 'int'>
    55000000
    <class 'int'>
    25000000
    <class 'int'>
    150000000
    <class 'int'>
    8000000
    <class 'int'>
    35000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    195000000
    <class 'int'>
    30000000
    <class 'int'>
    55000000
    <class 'int'>
    11000000
    <class 'int'>
    26000000
    <class 'int'>
    32000000
    <class 'int'>
    15000000
    <class 'int'>
    35000000
    <class 'int'>
    130000000
    <class 'int'>
    7000000
    <class 'int'>
    43000000
    <class 'int'>
    35000000
    <class 'int'>
    28000000
    <class 'int'>
    5000000
    <class 'int'>
    255000000
    <class 'int'>
    60000000
    <class 'int'>
    35000000
    <class 'int'>
    30000000
    <class 'int'>
    12000000
    <class 'int'>
    1030064
    <class 'int'>
    9000000
    <class 'int'>
    20000000
    <class 'int'>
    2500000
    <class 'int'>
    20000000
    <class 'int'>
    50000000
    <class 'int'>
    7500000
    <class 'int'>
    200000000
    <class 'int'>
    135000000
    <class 'int'>
    20000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    44000000
    <class 'int'>
    30000000
    <class 'int'>
    18000000
    <class 'int'>
    15000000
    <class 'int'>
    16000000
    <class 'int'>
    32000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    6000000
    <class 'int'>
    13000000
    <class 'int'>
    40000000
    <class 'int'>
    40000000
    <class 'int'>
    10900000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    28000000
    <class 'int'>
    125687
    <class 'int'>
    None
    <class 'NoneType'>
    9500000
    <class 'int'>
    15000000
    <class 'int'>
    11000000
    <class 'int'>
    23000000
    <class 'int'>
    78000000
    <class 'int'>
    16000000
    <class 'int'>
    18000000
    <class 'int'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    4800000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13500000
    <class 'int'>
    30000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    16000000
    <class 'int'>
    18000000
    <class 'int'>
    56000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    2900000
    <class 'int'>
    8500000
    <class 'int'>
    60000000
    <class 'int'>
    27220000
    <class 'int'>
    105000000
    <class 'int'>
    16000000
    <class 'int'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    4600000
    <class 'int'>
    28000000
    <class 'int'>
    25000000
    <class 'int'>
    25500000
    <class 'int'>
    120000000
    <class 'int'>
    35000000
    <class 'int'>
    55000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15600000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    3500000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    1500000
    <class 'int'>
    34000000
    <class 'int'>
    None
    <class 'NoneType'>
    8200000
    <class 'int'>
    None
    <class 'NoneType'>
    160000000
    <class 'int'>
    None
    <class 'NoneType'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    17500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6218100
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    5200000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    18000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    4900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    1104000
    <class 'int'>
    None
    <class 'NoneType'>
    950000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    33000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    83000000
    <class 'int'>
    None
    <class 'NoneType'>
    15
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    567000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    10000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    725000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    1800000
    <class 'int'>
    135000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1052753
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    996519
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    89
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    800000
    <class 'int'>
    5112027
    <class 'int'>
    800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7720000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    232000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    750
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1400000
    <class 'int'>
    650
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500
    <class 'int'>
    None
    <class 'NoneType'>
    348164
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    1000000
    <class 'int'>
    20000000
    <class 'int'>
    12305523
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    19000000
    <class 'int'>
    3000000
    <class 'int'>
    44000000
    <class 'int'>
    31000000
    <class 'int'>
    31000000
    <class 'int'>
    15000000
    <class 'int'>
    25000000
    <class 'int'>
    15000000
    <class 'int'>
    10000000
    <class 'int'>
    12500000
    <class 'int'>
    4000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17900000
    <class 'int'>
    3500000
    <class 'int'>
    15000000
    <class 'int'>
    2200000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    3800000
    <class 'int'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    900000
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    945000
    <class 'int'>
    18000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    1700000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    19000000
    <class 'int'>
    114
    <class 'int'>
    11500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    860000
    <class 'int'>
    None
    <class 'NoneType'>
    2410000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    150000000
    <class 'int'>
    150000000
    <class 'int'>
    180000000
    <class 'int'>
    54000000
    <class 'int'>
    40000000
    <class 'int'>
    150000000
    <class 'int'>
    113000000
    <class 'int'>
    50000000
    <class 'int'>
    126000000
    <class 'int'>
    75000000
    <class 'int'>
    100000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    62000000
    <class 'int'>
    56000000
    <class 'int'>
    132000000
    <class 'int'>
    85000000
    <class 'int'>
    26000000
    <class 'int'>
    150000000
    <class 'int'>
    70000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    130000000
    <class 'int'>
    43000000
    <class 'int'>
    207000000
    <class 'int'>
    53000000
    <class 'int'>
    50000000
    <class 'int'>
    28000000
    <class 'int'>
    14000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    87000000
    <class 'int'>
    25000000
    <class 'int'>
    110000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    42000000
    <class 'int'>
    65000000
    <class 'int'>
    43000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    30000000
    <class 'int'>
    70000000
    <class 'int'>
    2200000
    <class 'int'>
    29000000
    <class 'int'>
    72000000
    <class 'int'>
    None
    <class 'NoneType'>
    39000000
    <class 'int'>
    4000000
    <class 'int'>
    58000000
    <class 'int'>
    50000000
    <class 'int'>
    57000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    4800000
    <class 'int'>
    15000000
    <class 'int'>
    88000000
    <class 'int'>
    15000000
    <class 'int'>
    15000000
    <class 'int'>
    25000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    80000000
    <class 'int'>
    88000000
    <class 'int'>
    30000000
    <class 'int'>
    52000000
    <class 'int'>
    50000000
    <class 'int'>
    70000000
    <class 'int'>
    19000000
    <class 'int'>
    6500000
    <class 'int'>
    22000000
    <class 'int'>
    50000000
    <class 'int'>
    84000000
    <class 'int'>
    22000000
    <class 'int'>
    50000000
    <class 'int'>
    43000000
    <class 'int'>
    12000000
    <class 'int'>
    30000000
    <class 'int'>
    7000000
    <class 'int'>
    30000000
    <class 'int'>
    32000000
    <class 'int'>
    19000000
    <class 'int'>
    100000000
    <class 'int'>
    45000000
    <class 'int'>
    45000000
    <class 'int'>
    35000000
    <class 'int'>
    169000
    <class 'int'>
    35000000
    <class 'int'>
    2000000
    <class 'int'>
    55000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    40000000
    <class 'int'>
    45000000
    <class 'int'>
    4000000
    <class 'int'>
    35000000
    <class 'int'>
    50000000
    <class 'int'>
    25000000
    <class 'int'>
    82000000
    <class 'int'>
    3000000
    <class 'int'>
    32000000
    <class 'int'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    18000000
    <class 'int'>
    135000000
    <class 'int'>
    15000000
    <class 'int'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    130000000
    <class 'int'>
    10000000
    <class 'int'>
    7000000
    <class 'int'>
    35000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    950000
    <class 'int'>
    40000000
    <class 'int'>
    80000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    15000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    30000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    475000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    14200000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    900000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    818418
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    20000000
    <class 'int'>
    20000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    9000000
    <class 'int'>
    11000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    30000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    1900000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    1549000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    4750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10284523
    <class 'int'>
    25000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    450000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16398
    <class 'int'>
    10
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9600000
    <class 'int'>
    9000000
    <class 'int'>
    2000000
    <class 'int'>
    6000000
    <class 'int'>
    3500000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    72000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    200000000
    <class 'int'>
    120000000
    <class 'int'>
    150000000
    <class 'int'>
    125000000
    <class 'int'>
    12000000
    <class 'int'>
    80000000
    <class 'int'>
    150000000
    <class 'int'>
    35000000
    <class 'int'>
    90000000
    <class 'int'>
    85000000
    <class 'int'>
    40000000
    <class 'int'>
    110000000
    <class 'int'>
    25000000
    <class 'int'>
    76000000
    <class 'int'>
    None
    <class 'NoneType'>
    270000000
    <class 'int'>
    18000000
    <class 'int'>
    82500000
    <class 'int'>
    None
    <class 'NoneType'>
    52000000
    <class 'int'>
    45000000
    <class 'int'>
    80000000
    <class 'int'>
    50000000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    27000000
    <class 'int'>
    40000000
    <class 'int'>
    25000000
    <class 'int'>
    50000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    100000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    149000000
    <class 'int'>
    80000000
    <class 'int'>
    75000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    45000000
    <class 'int'>
    75000000
    <class 'int'>
    4200000
    <class 'int'>
    50000000
    <class 'int'>
    40000000
    <class 'int'>
    50000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    85000000
    <class 'int'>
    30000000
    <class 'int'>
    15000000
    <class 'int'>
    135000000
    <class 'int'>
    18000000
    <class 'int'>
    31000000
    <class 'int'>
    90000000
    <class 'int'>
    2000000
    <class 'int'>
    21000000
    <class 'int'>
    50000000
    <class 'int'>
    40000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    17500000
    <class 'int'>
    25000000
    <class 'int'>
    55000000
    <class 'int'>
    15500000
    <class 'int'>
    40000000
    <class 'int'>
    70000000
    <class 'int'>
    5000000
    <class 'int'>
    72500000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    40000000
    <class 'int'>
    37665000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    10000000
    <class 'int'>
    30000000
    <class 'int'>
    32000000
    <class 'int'>
    50000000
    <class 'int'>
    9000000
    <class 'int'>
    160000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16800000
    <class 'int'>
    54000000
    <class 'int'>
    8000000
    <class 'int'>
    64000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    12000000
    <class 'int'>
    25000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    63000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    1200000
    <class 'int'>
    19400000
    <class 'int'>
    12000000
    <class 'int'>
    15500000
    <class 'int'>
    20000000
    <class 'int'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    51000000
    <class 'int'>
    3700000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    33000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    11000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    25000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    50000000
    <class 'int'>
    40000000
    <class 'int'>
    6000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    85000000
    <class 'int'>
    30000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    4000000
    <class 'int'>
    5000000
    <class 'int'>
    45000000
    <class 'int'>
    20000000
    <class 'int'>
    15000000
    <class 'int'>
    15000000
    <class 'int'>
    994000
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    20000000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    11000000
    <class 'int'>
    45000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    24000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    150000
    <class 'int'>
    15000000
    <class 'int'>
    13000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    9000000
    <class 'int'>
    13000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    700000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    900000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    850000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    9364800
    <class 'int'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2380000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    14500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    2300000
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    210000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    500000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    8300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    79000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2700000
    <class 'int'>
    None
    <class 'NoneType'>
    130000000
    <class 'int'>
    30000000
    <class 'int'>
    50000000
    <class 'int'>
    20000000
    <class 'int'>
    29000000
    <class 'int'>
    92000000
    <class 'int'>
    75000000
    <class 'int'>
    175000000
    <class 'int'>
    100000000
    <class 'int'>
    120000000
    <class 'int'>
    75000000
    <class 'int'>
    17000000
    <class 'int'>
    160000000
    <class 'int'>
    66000000
    <class 'int'>
    165000000
    <class 'int'>
    150000000
    <class 'int'>
    13000000
    <class 'int'>
    45000000
    <class 'int'>
    110000000
    <class 'int'>
    80000000
    <class 'int'>
    125000000
    <class 'int'>
    116000000
    <class 'int'>
    65000000
    <class 'int'>
    4000000
    <class 'int'>
    65000000
    <class 'int'>
    7000000
    <class 'int'>
    60000000
    <class 'int'>
    140000000
    <class 'int'>
    30000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    26000000
    <class 'int'>
    37000000
    <class 'int'>
    31000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    19000000
    <class 'int'>
    70000000
    <class 'int'>
    20000000
    <class 'int'>
    12000000
    <class 'int'>
    155000000
    <class 'int'>
    9000000
    <class 'int'>
    1200000
    <class 'int'>
    200000000
    <class 'int'>
    25000000
    <class 'int'>
    70000000
    <class 'int'>
    42000000
    <class 'int'>
    27000000
    <class 'int'>
    33000000
    <class 'int'>
    100000000
    <class 'int'>
    60000000
    <class 'int'>
    1000000
    <class 'int'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    120000000
    <class 'int'>
    105000000
    <class 'int'>
    80000000
    <class 'int'>
    40000000
    <class 'int'>
    6500000
    <class 'int'>
    60000000
    <class 'int'>
    57000000
    <class 'int'>
    40000000
    <class 'int'>
    145000000
    <class 'int'>
    2700000
    <class 'int'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    80000000
    <class 'int'>
    5000000
    <class 'int'>
    12000000
    <class 'int'>
    25000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    35000000
    <class 'int'>
    90000000
    <class 'int'>
    7000
    <class 'int'>
    10000000
    <class 'int'>
    110000000
    <class 'int'>
    100000000
    <class 'int'>
    22000000
    <class 'int'>
    6000000
    <class 'int'>
    17000000
    <class 'int'>
    37000000
    <class 'int'>
    17500000
    <class 'int'>
    2800000
    <class 'int'>
    56000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    130000
    <class 'int'>
    47000000
    <class 'int'>
    32000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24665810
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    60000000
    <class 'int'>
    40000000
    <class 'int'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    40000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    20000000
    <class 'int'>
    31000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    825000
    <class 'int'>
    2500000
    <class 'int'>
    400000
    <class 'int'>
    30000000
    <class 'int'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    23000000
    <class 'int'>
    50000000
    <class 'int'>
    110000000
    <class 'int'>
    18000000
    <class 'int'>
    53000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    5400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7200000
    <class 'int'>
    45000000
    <class 'int'>
    23000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    74500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    6800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    74050
    <class 'int'>
    30000000
    <class 'int'>
    200000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19250000
    <class 'int'>
    26000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000
    <class 'int'>
    6500000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    14
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1400000
    <class 'int'>
    18000000
    <class 'int'>
    2600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    90
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2600000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    823258
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    350000
    <class 'int'>
    6000000
    <class 'int'>
    2000000
    <class 'int'>
    1800000
    <class 'int'>
    6000000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    3200000
    <class 'int'>
    12000
    <class 'int'>
    2000000
    <class 'int'>
    300000
    <class 'int'>
    90000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3352254
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    850000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    19000000
    <class 'int'>
    3500000
    <class 'int'>
    27000000
    <class 'int'>
    4500000
    <class 'int'>
    18000000
    <class 'int'>
    54000000
    <class 'int'>
    550000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    6500000
    <class 'int'>
    8000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    44000000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5100000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4800000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    7200000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    10000000
    <class 'int'>
    350000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    300000000
    <class 'int'>
    150000000
    <class 'int'>
    150000000
    <class 'int'>
    70000000
    <class 'int'>
    25000000
    <class 'int'>
    150000000
    <class 'int'>
    130000000
    <class 'int'>
    258000000
    <class 'int'>
    18000000
    <class 'int'>
    70000000
    <class 'int'>
    25000000
    <class 'int'>
    110000000
    <class 'int'>
    140000000
    <class 'int'>
    15000000
    <class 'int'>
    65000000
    <class 'int'>
    110000000
    <class 'int'>
    7500000
    <class 'int'>
    60000000
    <class 'int'>
    61000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    3800000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    160000000
    <class 'int'>
    50000000
    <class 'int'>
    20000000
    <class 'int'>
    60000000
    <class 'int'>
    150000000
    <class 'int'>
    150000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    130000000
    <class 'int'>
    85000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    85000000
    <class 'int'>
    34000000
    <class 'int'>
    75000000
    <class 'int'>
    75000000
    <class 'int'>
    15000000
    <class 'int'>
    70000000
    <class 'int'>
    40000000
    <class 'int'>
    50000000
    <class 'int'>
    25000000
    <class 'int'>
    20000000
    <class 'int'>
    30000000
    <class 'int'>
    180000000
    <class 'int'>
    160000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    70000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    55000000
    <class 'int'>
    25000000
    <class 'int'>
    86000000
    <class 'int'>
    25000000
    <class 'int'>
    15000
    <class 'int'>
    175000000
    <class 'int'>
    85000000
    <class 'int'>
    70000000
    <class 'int'>
    8000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    39000000
    <class 'int'>
    51500000
    <class 'int'>
    70000000
    <class 'int'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    85000000
    <class 'int'>
    25000000
    <class 'int'>
    24000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    67000000
    <class 'int'>
    10000000
    <class 'int'>
    16000000
    <class 'int'>
    150000000
    <class 'int'>
    45000000
    <class 'int'>
    10200000
    <class 'int'>
    None
    <class 'NoneType'>
    53000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    12000000
    <class 'int'>
    15000000
    <class 'int'>
    16500000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    60000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    45000000
    <class 'int'>
    2000000
    <class 'int'>
    30000000
    <class 'int'>
    16000000
    <class 'int'>
    9000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    60795000
    <class 'int'>
    5200000
    <class 'int'>
    20000000
    <class 'int'>
    60000000
    <class 'int'>
    40000000
    <class 'int'>
    27500000
    <class 'int'>
    20000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    21000000
    <class 'int'>
    20000000
    <class 'int'>
    19000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    16500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    230400
    <class 'int'>
    35000000
    <class 'int'>
    24000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    20000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2600000
    <class 'int'>
    9000000
    <class 'int'>
    45000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    23000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    20000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    67000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    4000000
    <class 'int'>
    16000000
    <class 'int'>
    2000000
    <class 'int'>
    13000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4600000
    <class 'int'>
    None
    <class 'NoneType'>
    50000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    825000
    <class 'int'>
    12
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    4000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    120000
    <class 'int'>
    700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    441892
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    31500000
    <class 'int'>
    400000
    <class 'int'>
    35000000
    <class 'int'>
    34000000
    <class 'int'>
    4000000
    <class 'int'>
    239000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    17500000
    <class 'int'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    14000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    2700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    420000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1355000
    <class 'int'>
    800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6400000
    <class 'int'>
    28000000
    <class 'int'>
    30000000
    <class 'int'>
    11000000
    <class 'int'>
    27000000
    <class 'int'>
    15000000
    <class 'int'>
    30000000
    <class 'int'>
    18000000
    <class 'int'>
    8000000
    <class 'int'>
    1800000
    <class 'int'>
    18000000
    <class 'int'>
    28000000
    <class 'int'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    8200000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    14400000
    <class 'int'>
    3000000
    <class 'int'>
    25300000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    1500000
    <class 'int'>
    8000000
    <class 'int'>
    1746964
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    8600000
    <class 'int'>
    58000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7200000
    <class 'int'>
    None
    <class 'NoneType'>
    2550000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    8000000
    <class 'int'>
    14500000
    <class 'int'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1065000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1250000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    5500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    32350000
    <class 'int'>
    25000000
    <class 'int'>
    36000000
    <class 'int'>
    27500000
    <class 'int'>
    None
    <class 'NoneType'>
    39000000
    <class 'int'>
    40600000
    <class 'int'>
    12000000
    <class 'int'>
    15000000
    <class 'int'>
    20500000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    6200000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    10100000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    12000000
    <class 'int'>
    22000000
    <class 'int'>
    10000000
    <class 'int'>
    350000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27
    <class 'int'>
    10000000
    <class 'int'>
    22000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5952000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    980385
    <class 'int'>
    19000000
    <class 'int'>
    350000
    <class 'int'>
    None
    <class 'NoneType'>
    425000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    33000000
    <class 'int'>
    30000000
    <class 'int'>
    6000000
    <class 'int'>
    72000000
    <class 'int'>
    58000000
    <class 'int'>
    65000000
    <class 'int'>
    90000000
    <class 'int'>
    35000000
    <class 'int'>
    29500000
    <class 'int'>
    52000000
    <class 'int'>
    19000000
    <class 'int'>
    55000000
    <class 'int'>
    60000000
    <class 'int'>
    2500000
    <class 'int'>
    90000000
    <class 'int'>
    100000000
    <class 'int'>
    52000000
    <class 'int'>
    12000000
    <class 'int'>
    30000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    175000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    24000000
    <class 'int'>
    3600000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    7000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    60000000
    <class 'int'>
    98000000
    <class 'int'>
    18000000
    <class 'int'>
    16500000
    <class 'int'>
    10000000
    <class 'int'>
    15000000
    <class 'int'>
    30250000
    <class 'int'>
    50000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    3500000
    <class 'int'>
    11000000
    <class 'int'>
    53000000
    <class 'int'>
    45000000
    <class 'int'>
    15000000
    <class 'int'>
    9000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    8
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    44000000
    <class 'int'>
    None
    <class 'NoneType'>
    55000000
    <class 'int'>
    21000000
    <class 'int'>
    5000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    62000000
    <class 'int'>
    17000000
    <class 'int'>
    11000000
    <class 'int'>
    58000000
    <class 'int'>
    42000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8169363
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    6100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    27000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    28000000
    <class 'int'>
    6400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    6
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13365000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    1200000
    <class 'int'>
    28000000
    <class 'int'>
    18000000
    <class 'int'>
    80000000
    <class 'int'>
    50000000
    <class 'int'>
    35000000
    <class 'int'>
    40000000
    <class 'int'>
    11000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    49000000
    <class 'int'>
    23000000
    <class 'int'>
    12500000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    14000000
    <class 'int'>
    31000000
    <class 'int'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    8000000
    <class 'int'>
    31000000
    <class 'int'>
    15000000
    <class 'int'>
    3000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    10000000
    <class 'int'>
    10000000
    <class 'int'>
    8000000
    <class 'int'>
    5000000
    <class 'int'>
    34000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    220000
    <class 'int'>
    11700000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    42000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    3705538
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    10000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    24000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    35000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4439832
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    2800000
    <class 'int'>
    28000000
    <class 'int'>
    12000000
    <class 'int'>
    2000000
    <class 'int'>
    11000000
    <class 'int'>
    10000000
    <class 'int'>
    9300000
    <class 'int'>
    350000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5500000
    <class 'int'>
    1250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    12000000
    <class 'int'>
    7000000
    <class 'int'>
    1750000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    32000000
    <class 'int'>
    18000000
    <class 'int'>
    12000000
    <class 'int'>
    5000000
    <class 'int'>
    4100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    145786
    <class 'int'>
    10000000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    62000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2200000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    80000000
    <class 'int'>
    4000000
    <class 'int'>
    7000000
    <class 'int'>
    75000000
    <class 'int'>
    70000000
    <class 'int'>
    14500000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    100000000
    <class 'int'>
    54000000
    <class 'int'>
    32000000
    <class 'int'>
    19000000
    <class 'int'>
    40000000
    <class 'int'>
    46000000
    <class 'int'>
    12000000
    <class 'int'>
    60000000
    <class 'int'>
    14000000
    <class 'int'>
    65000000
    <class 'int'>
    50000000
    <class 'int'>
    36000000
    <class 'int'>
    57000000
    <class 'int'>
    54000000
    <class 'int'>
    80000000
    <class 'int'>
    30000000
    <class 'int'>
    50000000
    <class 'int'>
    27000000
    <class 'int'>
    55000000
    <class 'int'>
    92000000
    <class 'int'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    44000000
    <class 'int'>
    30000000
    <class 'int'>
    50000000
    <class 'int'>
    10000000
    <class 'int'>
    50000000
    <class 'int'>
    36000000
    <class 'int'>
    25000000
    <class 'int'>
    4500000
    <class 'int'>
    200000
    <class 'int'>
    50000000
    <class 'int'>
    55000000
    <class 'int'>
    45000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    46000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    45000000
    <class 'int'>
    2962051
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    67000000
    <class 'int'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    12000000
    <class 'int'>
    13000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    1000000
    <class 'int'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    47000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    44000000
    <class 'int'>
    5000000
    <class 'int'>
    25530000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    20000000
    <class 'int'>
    50000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    45000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    42000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    40000000
    <class 'int'>
    17500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    8000000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    350000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    7000000
    <class 'int'>
    25000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    103000000
    <class 'int'>
    9000000
    <class 'int'>
    123000000
    <class 'int'>
    90000000
    <class 'int'>
    10000000
    <class 'int'>
    125000000
    <class 'int'>
    48000000
    <class 'int'>
    19000000
    <class 'int'>
    100000000
    <class 'int'>
    70000000
    <class 'int'>
    127500000
    <class 'int'>
    92000000
    <class 'int'>
    45000000
    <class 'int'>
    110000000
    <class 'int'>
    23000000
    <class 'int'>
    45000000
    <class 'int'>
    25000000
    <class 'int'>
    62000000
    <class 'int'>
    5000000
    <class 'int'>
    90000000
    <class 'int'>
    52000000
    <class 'int'>
    55000000
    <class 'int'>
    26000000
    <class 'int'>
    23000000
    <class 'int'>
    26000000
    <class 'int'>
    95000000
    <class 'int'>
    45000000
    <class 'int'>
    31000000
    <class 'int'>
    4500000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    95000000
    <class 'int'>
    30000000
    <class 'int'>
    7000000
    <class 'int'>
    12800000
    <class 'int'>
    16000000
    <class 'int'>
    30000000
    <class 'int'>
    10000000
    <class 'int'>
    80000000
    <class 'int'>
    48000000
    <class 'int'>
    13000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    82000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    51000000
    <class 'int'>
    55000000
    <class 'int'>
    75000000
    <class 'int'>
    65000000
    <class 'int'>
    41300000
    <class 'int'>
    60000000
    <class 'int'>
    28000000
    <class 'int'>
    25000000
    <class 'int'>
    30000000
    <class 'int'>
    80000000
    <class 'int'>
    90000000
    <class 'int'>
    33000000
    <class 'int'>
    46000000
    <class 'int'>
    None
    <class 'NoneType'>
    63600000
    <class 'int'>
    None
    <class 'NoneType'>
    33000000
    <class 'int'>
    83000000
    <class 'int'>
    26000000
    <class 'int'>
    80000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    120000000
    <class 'int'>
    None
    <class 'NoneType'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    43000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    42000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    11000000
    <class 'int'>
    28000000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    14000000
    <class 'int'>
    32000000
    <class 'int'>
    100000000
    <class 'int'>
    60000000
    <class 'int'>
    43000000
    <class 'int'>
    10000000
    <class 'int'>
    20000000
    <class 'int'>
    5500000
    <class 'int'>
    16000000
    <class 'int'>
    5000000
    <class 'int'>
    85000000
    <class 'int'>
    84000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    90000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    65000000
    <class 'int'>
    40000000
    <class 'int'>
    15000000
    <class 'int'>
    35000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    5000000
    <class 'int'>
    40000000
    <class 'int'>
    14000000
    <class 'int'>
    13500000
    <class 'int'>
    8000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    10000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    25000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    10000000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    15000000
    <class 'int'>
    10000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    1500000
    <class 'int'>
    34000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    75000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    65000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    76000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    57000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    10500000
    <class 'int'>
    10000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    17000000
    <class 'int'>
    10700000
    <class 'int'>
    20000000
    <class 'int'>
    1000000
    <class 'int'>
    12000000
    <class 'int'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    13200000
    <class 'int'>
    8000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    4300000
    <class 'int'>
    2500000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    22000000
    <class 'int'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    140000000
    <class 'int'>
    90000000
    <class 'int'>
    45000000
    <class 'int'>
    15000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    33000000
    <class 'int'>
    120000000
    <class 'int'>
    70000000
    <class 'int'>
    60000000
    <class 'int'>
    35000000
    <class 'int'>
    55000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    23000000
    <class 'int'>
    66000000
    <class 'int'>
    20000000
    <class 'int'>
    85000000
    <class 'int'>
    90000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    80000000
    <class 'int'>
    50000000
    <class 'int'>
    90000000
    <class 'int'>
    140000000
    <class 'int'>
    65000000
    <class 'int'>
    130000000
    <class 'int'>
    75000000
    <class 'int'>
    75000000
    <class 'int'>
    70000000
    <class 'int'>
    18500000
    <class 'int'>
    25000000
    <class 'int'>
    95000000
    <class 'int'>
    23000000
    <class 'int'>
    60000000
    <class 'int'>
    60000
    <class 'int'>
    55000000
    <class 'int'>
    None
    <class 'NoneType'>
    52000000
    <class 'int'>
    27000000
    <class 'int'>
    40000000
    <class 'int'>
    71000000
    <class 'int'>
    70000000
    <class 'int'>
    65000000
    <class 'int'>
    60000000
    <class 'int'>
    30000000
    <class 'int'>
    48000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    90000000
    <class 'int'>
    None
    <class 'NoneType'>
    1350000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    60000000
    <class 'int'>
    90000000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    20000000
    <class 'int'>
    13000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    73000000
    <class 'int'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    20000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    25000000
    <class 'int'>
    70000000
    <class 'int'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    30000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    650000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    17000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    45000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    75000000
    <class 'int'>
    25000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    33000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40000000
    <class 'int'>
    48000000
    <class 'int'>
    16000000
    <class 'int'>
    35000000
    <class 'int'>
    40000000
    <class 'int'>
    16400000
    <class 'int'>
    37000000
    <class 'int'>
    32000000
    <class 'int'>
    27000000
    <class 'int'>
    18000000
    <class 'int'>
    70000000
    <class 'int'>
    25000000
    <class 'int'>
    55000000
    <class 'int'>
    10000000
    <class 'int'>
    7500000
    <class 'int'>
    32000000
    <class 'int'>
    30000000
    <class 'int'>
    1500000
    <class 'int'>
    8000000
    <class 'int'>
    500000
    <class 'int'>
    18000000
    <class 'int'>
    11500000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    14000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    11000000
    <class 'int'>
    20000000
    <class 'int'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    18598420
    <class 'int'>
    22500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    31000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13800000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    7000000
    <class 'int'>
    787000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    645180
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    8045760
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    33000000
    <class 'int'>
    2500000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    130000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    100000000
    <class 'int'>
    70000000
    <class 'int'>
    48000000
    <class 'int'>
    30000000
    <class 'int'>
    35000000
    <class 'int'>
    27000000
    <class 'int'>
    23000000
    <class 'int'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    75000000
    <class 'int'>
    30000000
    <class 'int'>
    65000000
    <class 'int'>
    42000000
    <class 'int'>
    16000000
    <class 'int'>
    38000000
    <class 'int'>
    25000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    9000000
    <class 'int'>
    16500000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    2500000
    <class 'int'>
    29000000
    <class 'int'>
    40000000
    <class 'int'>
    6500000
    <class 'int'>
    23000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    6000000
    <class 'int'>
    26000000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    14000000
    <class 'int'>
    30000000
    <class 'int'>
    6000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    48000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    33000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    18000000
    <class 'int'>
    12000000
    <class 'int'>
    8000000
    <class 'int'>
    15000000
    <class 'int'>
    25000000
    <class 'int'>
    1500000
    <class 'int'>
    39000000
    <class 'int'>
    63000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    70000000
    <class 'int'>
    12300000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    29000000
    <class 'int'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    17500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    4000000
    <class 'int'>
    3000000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    7000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    2800000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    20000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    24000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23000000
    <class 'int'>
    15000000
    <class 'int'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    23000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    2500000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    425000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    17000000
    <class 'int'>
    6000000
    <class 'int'>
    22700000
    <class 'int'>
    40000000
    <class 'int'>
    13000000
    <class 'int'>
    3600000
    <class 'int'>
    15000000
    <class 'int'>
    27000000
    <class 'int'>
    23000000
    <class 'int'>
    16000000
    <class 'int'>
    5000000
    <class 'int'>
    15000000
    <class 'int'>
    20000000
    <class 'int'>
    17000000
    <class 'int'>
    6000000
    <class 'int'>
    25000000
    <class 'int'>
    13000000
    <class 'int'>
    23000000
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    25710
    <class 'int'>
    1880006
    <class 'int'>
    3000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    2500000
    <class 'int'>
    10000000
    <class 'int'>
    17000000
    <class 'int'>
    16000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    7200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1800000
    <class 'int'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    95
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    5800000
    <class 'int'>
    114000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3200000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    1800000
    <class 'int'>
    7700000
    <class 'int'>
    1500000
    <class 'int'>
    4300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    275000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    950000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    6000000
    <class 'int'>
    7000000
    <class 'int'>
    2800000
    <class 'int'>
    85000
    <class 'int'>
    60000
    <class 'int'>
    1600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2600000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    500000
    <class 'int'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    7000000
    <class 'int'>
    400000
    <class 'int'>
    1200000
    <class 'int'>
    11000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2700000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    2200000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    1800000
    <class 'int'>
    None
    <class 'NoneType'>
    179000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    15000000
    <class 'int'>
    2000000
    <class 'int'>
    2000000
    <class 'int'>
    2200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    3200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    980000
    <class 'int'>
    None
    <class 'NoneType'>
    2650000
    <class 'int'>
    126
    <class 'int'>
    10000000
    <class 'int'>
    62000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    1800000
    <class 'int'>
    6000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    2135161
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    783000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    1200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    65000
    <class 'int'>
    2200000
    <class 'int'>
    3000000
    <class 'int'>
    7200000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    2500000
    <class 'int'>
    2200000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    777000
    <class 'int'>
    1200000
    <class 'int'>
    1800000
    <class 'int'>
    725000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1645000
    <class 'int'>
    450000
    <class 'int'>
    750000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    850000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    1300000
    <class 'int'>
    1300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    65000000
    <class 'int'>
    25000000
    <class 'int'>
    70000000
    <class 'int'>
    40000000
    <class 'int'>
    20000000
    <class 'int'>
    14000000
    <class 'int'>
    54000000
    <class 'int'>
    22000000
    <class 'int'>
    22000000
    <class 'int'>
    13500000
    <class 'int'>
    20000000
    <class 'int'>
    42000000
    <class 'int'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    30
    <class 'int'>
    35000000
    <class 'int'>
    60000000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    50000000
    <class 'int'>
    11000000
    <class 'int'>
    35000000
    <class 'int'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    10000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    16000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    10500000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37931000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    19000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    49000000
    <class 'int'>
    23000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    22000000
    <class 'int'>
    225000
    <class 'int'>
    None
    <class 'NoneType'>
    9000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22997992
    <class 'int'>
    500000
    <class 'int'>
    18000000
    <class 'int'>
    31000000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2068041
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    2500000
    <class 'int'>
    6000000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    5037000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    806948
    <class 'int'>
    2000000
    <class 'int'>
    12000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    270000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1300000
    <class 'int'>
    1100000
    <class 'int'>
    1800000
    <class 'int'>
    2800000
    <class 'int'>
    9000000
    <class 'int'>
    24000000
    <class 'int'>
    8500000
    <class 'int'>
    9000000
    <class 'int'>
    2000000
    <class 'int'>
    3700000
    <class 'int'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    3800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    150000
    <class 'int'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    14600000
    <class 'int'>
    18000000
    <class 'int'>
    22000000
    <class 'int'>
    63000000
    <class 'int'>
    57000000
    <class 'int'>
    25000000
    <class 'int'>
    44000000
    <class 'int'>
    40000000
    <class 'int'>
    26000000
    <class 'int'>
    6900000
    <class 'int'>
    85000000
    <class 'int'>
    30000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    40000000
    <class 'int'>
    21
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    21000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    12500000
    <class 'int'>
    42000000
    <class 'int'>
    38000000
    <class 'int'>
    None
    <class 'NoneType'>
    45000000
    <class 'int'>
    30000000
    <class 'int'>
    22000000
    <class 'int'>
    11000000
    <class 'int'>
    35000000
    <class 'int'>
    9000000
    <class 'int'>
    22000000
    <class 'int'>
    13500000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    42000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    13000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30
    <class 'int'>
    7000000
    <class 'int'>
    35000000
    <class 'int'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    11500000
    <class 'int'>
    6000000
    <class 'int'>
    34000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    19000000
    <class 'int'>
    10000000
    <class 'int'>
    3500000
    <class 'int'>
    12
    <class 'int'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    33000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    19885552
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    125000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30
    <class 'int'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    68000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    3000000
    <class 'int'>
    12000000
    <class 'int'>
    9500000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    947000
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    640000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    65000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    200
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    31115000
    <class 'int'>
    9400000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    750000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    66500
    <class 'int'>
    18500000
    <class 'int'>
    16000000
    <class 'int'>
    6000000
    <class 'int'>
    15000000
    <class 'int'>
    6000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    17000000
    <class 'int'>
    14000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    24000000
    <class 'int'>
    37000000
    <class 'int'>
    25000000
    <class 'int'>
    13800000
    <class 'int'>
    20000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    113
    <class 'int'>
    None
    <class 'NoneType'>
    25000000
    <class 'int'>
    8500000
    <class 'int'>
    5000000
    <class 'int'>
    15000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    19000000
    <class 'int'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    24500000
    <class 'int'>
    9000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    6400000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    4500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    111000
    <class 'int'>
    22500
    <class 'int'>
    2700000
    <class 'int'>
    None
    <class 'NoneType'>
    4700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    6500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    800000
    <class 'int'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1900000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    8000000
    <class 'int'>
    7000000
    <class 'int'>
    500000
    <class 'int'>
    850000
    <class 'int'>
    None
    <class 'NoneType'>
    1700000
    <class 'int'>
    1500000
    <class 'int'>
    5500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    777000
    <class 'int'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4638783
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    450000
    <class 'int'>
    275000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    4000000
    <class 'int'>
    3000000
    <class 'int'>
    3500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    2200000
    <class 'int'>
    None
    <class 'NoneType'>
    25485000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000
    <class 'int'>
    10000000
    <class 'int'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000
    <class 'int'>
    None
    <class 'NoneType'>
    3716946
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    8200000
    <class 'int'>
    14000000
    <class 'int'>
    5600000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5760000
    <class 'int'>
    115
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    6000000
    <class 'int'>
    360000
    <class 'int'>
    3600000
    <class 'int'>
    6244087
    <class 'int'>
    1500000
    <class 'int'>
    12000000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1250000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    175000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6000000
    <class 'int'>
    20000000
    <class 'int'>
    650000
    <class 'int'>
    55000000
    <class 'int'>
    300000
    <class 'int'>
    2700000
    <class 'int'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2300000
    <class 'int'>
    4000000
    <class 'int'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7920000
    <class 'int'>
    None
    <class 'NoneType'>
    11
    <class 'int'>
    5000000
    <class 'int'>
    12000000
    <class 'int'>
    3500000
    <class 'int'>
    660000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    650000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24000000
    <class 'int'>
    None
    <class 'NoneType'>
    2700000
    <class 'int'>
    6800000
    <class 'int'>
    None
    <class 'NoneType'>
    1000000
    <class 'int'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11000000
    <class 'int'>
    None
    <class 'NoneType'>
    6727000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    90000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    315000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    None
    <class 'NoneType'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1377800
    <class 'int'>
    3000000
    <class 'int'>
    4653000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4800000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5115000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    700000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19000
    <class 'int'>
    


```python
for i in tmdbmovies:
    print(i['budget_adj'])
    print(type(i['budget_adj']))
```

    137999939
    <class 'int'>
    137999939
    <class 'int'>
    101199955
    <class 'int'>
    183999919
    <class 'int'>
    174799923
    <class 'int'>
    124199945
    <class 'int'>
    142599937
    <class 'int'>
    99359956
    <class 'int'>
    68079970
    <class 'int'>
    160999929
    <class 'int'>
    225399900
    <class 'int'>
    161919931
    <class 'int'>
    13799993
    <class 'int'>
    80959964
    <class 'int'>
    257599886
    <class 'int'>
    40479982
    <class 'int'>
    44159980
    <class 'int'>
    119599947
    <class 'int'>
    87399961
    <class 'int'>
    147199935
    <class 'int'>
    174799923
    <class 'int'>
    27599987
    <class 'int'>
    101199955
    <class 'int'>
    36799983
    <class 'int'>
    25759988
    <class 'int'>
    137999939
    <class 'int'>
    62559972
    <class 'int'>
    74519967
    <class 'int'>
    18399991
    <class 'int'>
    56119975
    <class 'int'>
    None
    <class 'NoneType'>
    45079980
    <class 'int'>
    26679988
    <class 'int'>
    36799983
    <class 'int'>
    53359976
    <class 'int'>
    5519997
    <class 'int'>
    None
    <class 'NoneType'>
    160999929
    <class 'int'>
    45999979
    <class 'int'>
    10119995
    <class 'int'>
    25759988
    <class 'int'>
    82799963
    <class 'int'>
    27599987
    <class 'int'>
    68999969
    <class 'int'>
    22999989
    <class 'int'>
    9199995
    <class 'int'>
    124199945
    <class 'int'>
    11039995
    <class 'int'>
    27599987
    <class 'int'>
    3679998
    <class 'int'>
    10855995
    <class 'int'>
    32199985
    <class 'int'>
    50599977
    <class 'int'>
    55199975
    <class 'int'>
    96599957
    <class 'int'>
    18399991
    <class 'int'>
    23919989
    <class 'int'>
    55199975
    <class 'int'>
    13799993
    <class 'int'>
    64399971
    <class 'int'>
    27599987
    <class 'int'>
    110399952
    <class 'int'>
    3219998
    <class 'int'>
    59799973
    <class 'int'>
    46091979
    <class 'int'>
    32199985
    <class 'int'>
    91999959
    <class 'int'>
    18399991
    <class 'int'>
    32199985
    <class 'int'>
    32199985
    <class 'int'>
    11959994
    <class 'int'>
    22999989
    <class 'int'>
    None
    <class 'NoneType'>
    137999939
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    91079959
    <class 'int'>
    32199985
    <class 'int'>
    22999989
    <class 'int'>
    2299998
    <class 'int'>
    31279986
    <class 'int'>
    73599967
    <class 'int'>
    15639993
    <class 'int'>
    32199985
    <class 'int'>
    10119995
    <class 'int'>
    28519987
    <class 'int'>
    32199985
    <class 'int'>
    3679998
    <class 'int'>
    None
    <class 'NoneType'>
    45999979
    <class 'int'>
    4599997
    <class 'int'>
    45999979
    <class 'int'>
    None
    <class 'NoneType'>
    9199995
    <class 'int'>
    58879974
    <class 'int'>
    None
    <class 'NoneType'>
    10975595
    <class 'int'>
    7819996
    <class 'int'>
    9199995
    <class 'int'>
    27599987
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3679998
    <class 'int'>
    None
    <class 'NoneType'>
    27599987
    <class 'int'>
    21159990
    <class 'int'>
    13799993
    <class 'int'>
    4599997
    <class 'int'>
    48759978
    <class 'int'>
    8187996
    <class 'int'>
    13799993
    <class 'int'>
    18399991
    <class 'int'>
    4599997
    <class 'int'>
    643999
    <class 'int'>
    25759988
    <class 'int'>
    10119995
    <class 'int'>
    None
    <class 'NoneType'>
    27599987
    <class 'int'>
    11039995
    <class 'int'>
    None
    <class 'NoneType'>
    22999989
    <class 'int'>
    50599977
    <class 'int'>
    None
    <class 'NoneType'>
    13615994
    <class 'int'>
    9199995
    <class 'int'>
    None
    <class 'NoneType'>
    13799993
    <class 'int'>
    17939992
    <class 'int'>
    None
    <class 'NoneType'>
    68079970
    <class 'int'>
    None
    <class 'NoneType'>
    18399991
    <class 'int'>
    None
    <class 'NoneType'>
    10119995
    <class 'int'>
    None
    <class 'NoneType'>
    36799983
    <class 'int'>
    919999
    <class 'int'>
    32199985
    <class 'int'>
    9199995
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5519997
    <class 'int'>
    36799983
    <class 'int'>
    None
    <class 'NoneType'>
    7359996
    <class 'int'>
    1655999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18399991
    <class 'int'>
    1839999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11039995
    <class 'int'>
    10119995
    <class 'int'>
    13799993
    <class 'int'>
    16559992
    <class 'int'>
    None
    <class 'NoneType'>
    579617
    <class 'int'>
    11039995
    <class 'int'>
    None
    <class 'NoneType'>
    9199995
    <class 'int'>
    36799983
    <class 'int'>
    None
    <class 'NoneType'>
    55199975
    <class 'int'>
    None
    <class 'NoneType'>
    22999989
    <class 'int'>
    12879994
    <class 'int'>
    None
    <class 'NoneType'>
    5519997
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9199995
    <class 'int'>
    18399991
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34039985
    <class 'int'>
    23919989
    <class 'int'>
    None
    <class 'NoneType'>
    25759988
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10119995
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6439997
    <class 'int'>
    None
    <class 'NoneType'>
    2759998
    <class 'int'>
    8831996
    <class 'int'>
    None
    <class 'NoneType'>
    9199995
    <class 'int'>
    13799993
    <class 'int'>
    2060799
    <class 'int'>
    919999
    <class 'int'>
    11959994
    <class 'int'>
    7359996
    <class 'int'>
    12879994
    <class 'int'>
    None
    <class 'NoneType'>
    3679998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7359996
    <class 'int'>
    None
    <class 'NoneType'>
    3035998
    <class 'int'>
    None
    <class 'NoneType'>
    7359996
    <class 'int'>
    None
    <class 'NoneType'>
    32199985
    <class 'int'>
    1839999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6439997
    <class 'int'>
    11039995
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2759998
    <class 'int'>
    5519997
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    91999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9199995
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19319991
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18399991
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4875997
    <class 'int'>
    None
    <class 'NoneType'>
    8279996
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11959994
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4599997
    <class 'int'>
    None
    <class 'NoneType'>
    9199995
    <class 'int'>
    None
    <class 'NoneType'>
    5519997
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2759998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    919999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    411721
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1793999
    <class 'int'>
    None
    <class 'NoneType'>
    919999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4599997
    <class 'int'>
    None
    <class 'NoneType'>
    5519997
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4047998
    <class 'int'>
    9199995
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1839999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24839989
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4599997
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3219998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    643999
    <class 'int'>
    459999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1839999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    919999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18399991
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    551999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6899996
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1379999
    <class 'int'>
    1379999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    597999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2299998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8739996
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2759998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3127998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2759998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    919999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1195999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2759998
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1747999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12419994
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    919999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1839999
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    151980023
    <class 'int'>
    156585478
    <class 'int'>
    156585478
    <class 'int'>
    18421821
    <class 'int'>
    115136381
    <class 'int'>
    230272762
    <class 'int'>
    151980023
    <class 'int'>
    12895274
    <class 'int'>
    31317095
    <class 'int'>
    16579638
    <class 'int'>
    56186554
    <class 'int'>
    62634191
    <class 'int'>
    116978563
    <class 'int'>
    78292739
    <class 'int'>
    230272762
    <class 'int'>
    36843642
    <class 'int'>
    115136381
    <class 'int'>
    36843642
    <class 'int'>
    7829273
    <class 'int'>
    193429120
    <class 'int'>
    27632731
    <class 'int'>
    3039600
    <class 'int'>
    156585478
    <class 'int'>
    50660007
    <class 'int'>
    38685824
    <class 'int'>
    184218210
    <class 'int'>
    163954207
    <class 'int'>
    40528006
    <class 'int'>
    54160153
    <class 'int'>
    128952747
    <class 'int'>
    147374568
    <class 'int'>
    165796389
    <class 'int'>
    13816365
    <class 'int'>
    4605455
    <class 'int'>
    133558202
    <class 'int'>
    1842182
    <class 'int'>
    82898194
    <class 'int'>
    55265463
    <class 'int'>
    11974183
    <class 'int'>
    87503649
    <class 'int'>
    13816365
    <class 'int'>
    15658547
    <class 'int'>
    64476373
    <class 'int'>
    121584018
    <class 'int'>
    46054552
    <class 'int'>
    101320015
    <class 'int'>
    23027276
    <class 'int'>
    46054552
    <class 'int'>
    25790549
    <class 'int'>
    50660007
    <class 'int'>
    6447637
    <class 'int'>
    10132001
    <class 'int'>
    8289819
    <class 'int'>
    14737456
    <class 'int'>
    3684364
    <class 'int'>
    18421821
    <class 'int'>
    92109105
    <class 'int'>
    115136381
    <class 'int'>
    46054552
    <class 'int'>
    11053092
    <class 'int'>
    36843642
    <class 'int'>
    16579638
    <class 'int'>
    10132001
    <class 'int'>
    41449097
    <class 'int'>
    None
    <class 'NoneType'>
    11053092
    <class 'int'>
    None
    <class 'NoneType'>
    23027276
    <class 'int'>
    None
    <class 'NoneType'>
    18421821
    <class 'int'>
    55265463
    <class 'int'>
    46054552
    <class 'int'>
    5066000
    <class 'int'>
    6447637
    <class 'int'>
    None
    <class 'NoneType'>
    11605747
    <class 'int'>
    110530926
    <class 'int'>
    92109105
    <class 'int'>
    36843642
    <class 'int'>
    133558202
    <class 'int'>
    None
    <class 'NoneType'>
    59870918
    <class 'int'>
    None
    <class 'NoneType'>
    94872378
    <class 'int'>
    36843642
    <class 'int'>
    None
    <class 'NoneType'>
    64476373
    <class 'int'>
    59870918
    <class 'int'>
    18421821
    <class 'int'>
    4513346
    <class 'int'>
    15658547
    <class 'int'>
    None
    <class 'NoneType'>
    13816365
    <class 'int'>
    119741836
    <class 'int'>
    23948367
    <class 'int'>
    59870918
    <class 'int'>
    None
    <class 'NoneType'>
    46054552
    <class 'int'>
    11974183
    <class 'int'>
    60792009
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20264003
    <class 'int'>
    55265463
    <class 'int'>
    9210910
    <class 'int'>
    16579638
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3684364
    <class 'int'>
    None
    <class 'NoneType'>
    46054552
    <class 'int'>
    23027276
    <class 'int'>
    25790549
    <class 'int'>
    46054552
    <class 'int'>
    None
    <class 'NoneType'>
    32238186
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    921091
    <class 'int'>
    12158401
    <class 'int'>
    4605455
    <class 'int'>
    4605455
    <class 'int'>
    22106185
    <class 'int'>
    None
    <class 'NoneType'>
    18421821
    <class 'int'>
    11283365
    <class 'int'>
    22106185
    <class 'int'>
    13816365
    <class 'int'>
    None
    <class 'NoneType'>
    13816365
    <class 'int'>
    None
    <class 'NoneType'>
    5987091
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    59870918
    <class 'int'>
    7368728
    <class 'int'>
    None
    <class 'NoneType'>
    24869458
    <class 'int'>
    25790549
    <class 'int'>
    1842182
    <class 'int'>
    18421821
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9210910
    <class 'int'>
    2763273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55265463
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20724548
    <class 'int'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25790549
    <class 'int'>
    18237602
    <class 'int'>
    14737456
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8289819
    <class 'int'>
    None
    <class 'NoneType'>
    7829273
    <class 'int'>
    None
    <class 'NoneType'>
    36843642
    <class 'int'>
    11513638
    <class 'int'>
    23027276
    <class 'int'>
    5066000
    <class 'int'>
    6171310
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12250510
    <class 'int'>
    5526546
    <class 'int'>
    None
    <class 'NoneType'>
    20264003
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5526546
    <class 'int'>
    3223818
    <class 'int'>
    7829273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13816365
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27632731
    <class 'int'>
    1842182
    <class 'int'>
    None
    <class 'NoneType'>
    921091
    <class 'int'>
    5066000
    <class 'int'>
    None
    <class 'NoneType'>
    13816365
    <class 'int'>
    184218
    <class 'int'>
    19342912
    <class 'int'>
    13816365
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5987091
    <class 'int'>
    27632731
    <class 'int'>
    6079200
    <class 'int'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    921091
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1842182
    <class 'int'>
    None
    <class 'NoneType'>
    33159277
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    921091
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    46054552
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18421821
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2579054
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    552654
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5526546
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11053092
    <class 'int'>
    4605455
    <class 'int'>
    9210910
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    64476373
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    27632731
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16579638
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2210618
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2901436
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5066000
    <class 'int'>
    None
    <class 'NoneType'>
    2486945
    <class 'int'>
    None
    <class 'NoneType'>
    1105309
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7368728
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5526546
    <class 'int'>
    None
    <class 'NoneType'>
    921091
    <class 'int'>
    None
    <class 'NoneType'>
    292906
    <class 'int'>
    27632731
    <class 'int'>
    None
    <class 'NoneType'>
    6447637
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    11974183
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3223818
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2302727
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23027276
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23027276
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1842182
    <class 'int'>
    9
    <class 'int'>
    None
    <class 'NoneType'>
    11697856
    <class 'int'>
    None
    <class 'NoneType'>
    3684364
    <class 'int'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1381636
    <class 'int'>
    None
    <class 'NoneType'>
    300208
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    460545
    <class 'int'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9210910
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15658547
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6447637
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    921091
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2763273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4605455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6447637
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9210910
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5526546
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11053092
    <class 'int'>
    1842182
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3684364
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    144298
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7368728
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    875036
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5526546
    <class 'int'>
    None
    <class 'NoneType'>
    16
    <class 'int'>
    6355528
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5434437
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    460545
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7368728
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    138163
    <class 'int'>
    None
    <class 'NoneType'>
    1289527
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    107
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16579638
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14737456
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2763273
    <class 'int'>
    None
    <class 'NoneType'>
    6447637
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1842182
    <class 'int'>
    921091
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    690818
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2763273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    184218
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2993545
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7829273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2292964
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39575591
    <class 'int'>
    50368934
    <class 'int'>
    4317337
    <class 'int'>
    14391124
    <class 'int'>
    35977810
    <class 'int'>
    71955620
    <class 'int'>
    79151182
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12592233
    <class 'int'>
    None
    <class 'NoneType'>
    12592233
    <class 'int'>
    12232455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19787795
    <class 'int'>
    21586686
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3238002
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50368934
    <class 'int'>
    21586686
    <class 'int'>
    None
    <class 'NoneType'>
    35977
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10793343
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21586686
    <class 'int'>
    79151182
    <class 'int'>
    2914202
    <class 'int'>
    None
    <class 'NoneType'>
    5396671
    <class 'int'>
    1798890
    <class 'int'>
    None
    <class 'NoneType'>
    4317337
    <class 'int'>
    None
    <class 'NoneType'>
    827489
    <class 'int'>
    None
    <class 'NoneType'>
    2158668
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1906823
    <class 'int'>
    719556
    <class 'int'>
    None
    <class 'NoneType'>
    240886902
    <class 'int'>
    71148030
    <class 'int'>
    35574015
    <class 'int'>
    254100108
    <class 'int'>
    177870075
    <class 'int'>
    152460065
    <class 'int'>
    203280086
    <class 'int'>
    30492013
    <class 'int'>
    91476039
    <class 'int'>
    7623003
    <class 'int'>
    91476039
    <class 'int'>
    35574015
    <class 'int'>
    132132056
    <class 'int'>
    203280086
    <class 'int'>
    30492013
    <class 'int'>
    152460065
    <class 'int'>
    106722045
    <class 'int'>
    29475612
    <class 'int'>
    23987050
    <class 'int'>
    50820021
    <class 'int'>
    40656017
    <class 'int'>
    177870075
    <class 'int'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    38623216
    <class 'int'>
    101640043
    <class 'int'>
    40656017
    <class 'int'>
    60984026
    <class 'int'>
    152460065
    <class 'int'>
    20328008
    <class 'int'>
    203280086
    <class 'int'>
    None
    <class 'NoneType'>
    66066028
    <class 'int'>
    26426411
    <class 'int'>
    16262406
    <class 'int'>
    53869223
    <class 'int'>
    30492013
    <class 'int'>
    20328008
    <class 'int'>
    33541214
    <class 'int'>
    None
    <class 'NoneType'>
    30492013
    <class 'int'>
    26426411
    <class 'int'>
    50820021
    <class 'int'>
    5082002
    <class 'int'>
    152460065
    <class 'int'>
    None
    <class 'NoneType'>
    101640043
    <class 'int'>
    18803408
    <class 'int'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    50820021
    <class 'int'>
    None
    <class 'NoneType'>
    47770820
    <class 'int'>
    71148030
    <class 'int'>
    26426411
    <class 'int'>
    32524813
    <class 'int'>
    177870075
    <class 'int'>
    3049201
    <class 'int'>
    40656017
    <class 'int'>
    15246006
    <class 'int'>
    81312034
    <class 'int'>
    None
    <class 'NoneType'>
    16262406
    <class 'int'>
    20328008
    <class 'int'>
    15246006
    <class 'int'>
    40656017
    <class 'int'>
    76230032
    <class 'int'>
    4573801
    <class 'int'>
    30492013
    <class 'int'>
    10164004
    <class 'int'>
    60984026
    <class 'int'>
    16262406
    <class 'int'>
    14941086
    <class 'int'>
    41672417
    <class 'int'>
    22360809
    <class 'int'>
    50820021
    <class 'int'>
    27442811
    <class 'int'>
    50820021
    <class 'int'>
    None
    <class 'NoneType'>
    20328008
    <class 'int'>
    39639616
    <class 'int'>
    None
    <class 'NoneType'>
    11180404
    <class 'int'>
    None
    <class 'NoneType'>
    18295207
    <class 'int'>
    38623216
    <class 'int'>
    17278807
    <class 'int'>
    11180404
    <class 'int'>
    35574015
    <class 'int'>
    7114803
    <class 'int'>
    14229606
    <class 'int'>
    None
    <class 'NoneType'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13213205
    <class 'int'>
    13213205
    <class 'int'>
    7114803
    <class 'int'>
    20328008
    <class 'int'>
    8131203
    <class 'int'>
    9147603
    <class 'int'>
    152460065
    <class 'int'>
    24393610
    <class 'int'>
    15246006
    <class 'int'>
    40656017
    <class 'int'>
    20328008
    <class 'int'>
    101640043
    <class 'int'>
    86394036
    <class 'int'>
    508200
    <class 'int'>
    None
    <class 'NoneType'>
    18295207
    <class 'int'>
    18295207
    <class 'int'>
    101640043
    <class 'int'>
    60984026
    <class 'int'>
    71148030
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3557401
    <class 'int'>
    8131203
    <class 'int'>
    60984026
    <class 'int'>
    4573801
    <class 'int'>
    12196805
    <class 'int'>
    None
    <class 'NoneType'>
    15246006
    <class 'int'>
    88426837
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35574015
    <class 'int'>
    None
    <class 'NoneType'>
    71148030
    <class 'int'>
    8131203
    <class 'int'>
    8131203
    <class 'int'>
    25410010
    <class 'int'>
    30492013
    <class 'int'>
    20328008
    <class 'int'>
    20328008
    <class 'int'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    9147603
    <class 'int'>
    7623003
    <class 'int'>
    76230032
    <class 'int'>
    None
    <class 'NoneType'>
    25410010
    <class 'int'>
    66066028
    <class 'int'>
    50820021
    <class 'int'>
    12196805
    <class 'int'>
    None
    <class 'NoneType'>
    14229606
    <class 'int'>
    86394036
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20328008
    <class 'int'>
    12196805
    <class 'int'>
    25410010
    <class 'int'>
    40656017
    <class 'int'>
    15246006
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22360809
    <class 'int'>
    35574015
    <class 'int'>
    None
    <class 'NoneType'>
    15246006
    <class 'int'>
    8131203
    <class 'int'>
    45738019
    <class 'int'>
    12196805
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3455761
    <class 'int'>
    None
    <class 'NoneType'>
    4370521
    <class 'int'>
    3049201
    <class 'int'>
    10164004
    <class 'int'>
    6098402
    <class 'int'>
    15246006
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    813120
    <class 'int'>
    79428291
    <class 'int'>
    3557401
    <class 'int'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    5183642
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8639403
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    9655804
    <class 'int'>
    8131203
    <class 'int'>
    None
    <class 'NoneType'>
    3557401
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    81312034
    <class 'int'>
    None
    <class 'NoneType'>
    6606602
    <class 'int'>
    None
    <class 'NoneType'>
    6403322
    <class 'int'>
    17278807
    <class 'int'>
    17278807
    <class 'int'>
    30492013
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60984026
    <class 'int'>
    None
    <class 'NoneType'>
    19311608
    <class 'int'>
    6098402
    <class 'int'>
    25410010
    <class 'int'>
    None
    <class 'NoneType'>
    7114803
    <class 'int'>
    None
    <class 'NoneType'>
    7419723
    <class 'int'>
    81312034
    <class 'int'>
    None
    <class 'NoneType'>
    3760681
    <class 'int'>
    None
    <class 'NoneType'>
    50820021
    <class 'int'>
    35574015
    <class 'int'>
    None
    <class 'NoneType'>
    9147603
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20328008
    <class 'int'>
    15246006
    <class 'int'>
    4573801
    <class 'int'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15246006
    <class 'int'>
    6504962
    <class 'int'>
    5143728
    <class 'int'>
    None
    <class 'NoneType'>
    60984026
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45738019
    <class 'int'>
    None
    <class 'NoneType'>
    4065601
    <class 'int'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    152460065
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    4065601
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2947561
    <class 'int'>
    None
    <class 'NoneType'>
    6098402
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25410010
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35574015
    <class 'int'>
    None
    <class 'NoneType'>
    431970
    <class 'int'>
    None
    <class 'NoneType'>
    8131203
    <class 'int'>
    25410010
    <class 'int'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7114803
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55902023
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    10164004
    <class 'int'>
    2044793
    <class 'int'>
    None
    <class 'NoneType'>
    101640043
    <class 'int'>
    8639403
    <class 'int'>
    20328008
    <class 'int'>
    12705005
    <class 'int'>
    5082002
    <class 'int'>
    20328008
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14229606
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4573801
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6606602
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23377209
    <class 'int'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20328008
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    76
    <class 'int'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    1727880
    <class 'int'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6606602
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15246006
    <class 'int'>
    5082002
    <class 'int'>
    1016400
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40656017
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21344409
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7114803
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2032800
    <class 'int'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    15246006
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2541001
    <class 'int'>
    None
    <class 'NoneType'>
    1524600
    <class 'int'>
    7114803
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12705005
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42688818
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12196805
    <class 'int'>
    None
    <class 'NoneType'>
    5082
    <class 'int'>
    1524600
    <class 'int'>
    None
    <class 'NoneType'>
    7623003
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3252481
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58951225
    <class 'int'>
    None
    <class 'NoneType'>
    5082002
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3557401
    <class 'int'>
    3557401
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9147603
    <class 'int'>
    109
    <class 'int'>
    None
    <class 'NoneType'>
    254100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25410010
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31703
    <class 'int'>
    3303301
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10164004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6098402
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3557
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    160000000
    <class 'int'>
    200000000
    <class 'int'>
    200000000
    <class 'int'>
    13000000
    <class 'int'>
    250000000
    <class 'int'>
    69000000
    <class 'int'>
    80000000
    <class 'int'>
    165000000
    <class 'int'>
    80000000
    <class 'int'>
    170000000
    <class 'int'>
    260000000
    <class 'int'>
    200000000
    <class 'int'>
    110000000
    <class 'int'>
    18000000
    <class 'int'>
    8000000
    <class 'int'>
    155000000
    <class 'int'>
    28000000
    <class 'int'>
    100000000
    <class 'int'>
    150000000
    <class 'int'>
    200000000
    <class 'int'>
    38000000
    <class 'int'>
    125000000
    <class 'int'>
    150000000
    <class 'int'>
    30000000
    <class 'int'>
    40000000
    <class 'int'>
    60000000
    <class 'int'>
    100000000
    <class 'int'>
    40000000
    <class 'int'>
    95000000
    <class 'int'>
    100000000
    <class 'int'>
    165000000
    <class 'int'>
    25000000
    <class 'int'>
    110000000
    <class 'int'>
    58000000
    <class 'int'>
    20000000
    <class 'int'>
    80000000
    <class 'int'>
    80000000
    <class 'int'>
    55000000
    <class 'int'>
    37000000
    <class 'int'>
    20000000
    <class 'int'>
    68000000
    <class 'int'>
    130000000
    <class 'int'>
    150000000
    <class 'int'>
    100000000
    <class 'int'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    30000000
    <class 'int'>
    15000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    32000000
    <class 'int'>
    7000000
    <class 'int'>
    117000000
    <class 'int'>
    3000000
    <class 'int'>
    60000000
    <class 'int'>
    1000000
    <class 'int'>
    40000000
    <class 'int'>
    19000000
    <class 'int'>
    100000000
    <class 'int'>
    112000000
    <class 'int'>
    35000000
    <class 'int'>
    44000000
    <class 'int'>
    19000000
    <class 'int'>
    10000000
    <class 'int'>
    52000000
    <class 'int'>
    25000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    40000000
    <class 'int'>
    65000000
    <class 'int'>
    8000000
    <class 'int'>
    52000000
    <class 'int'>
    45000000
    <class 'int'>
    8000000
    <class 'int'>
    30000000
    <class 'int'>
    35000000
    <class 'int'>
    25000000
    <class 'int'>
    25000000
    <class 'int'>
    40000000
    <class 'int'>
    None
    <class 'NoneType'>
    47000000
    <class 'int'>
    65000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    20000000
    <class 'int'>
    2000000
    <class 'int'>
    75000000
    <class 'int'>
    26000000
    <class 'int'>
    None
    <class 'NoneType'>
    100000000
    <class 'int'>
    20000000
    <class 'int'>
    35000000
    <class 'int'>
    36000000
    <class 'int'>
    21800000
    <class 'int'>
    20000000
    <class 'int'>
    14000000
    <class 'int'>
    80000000
    <class 'int'>
    32000000
    <class 'int'>
    55000000
    <class 'int'>
    15000000
    <class 'int'>
    32000000
    <class 'int'>
    24000000
    <class 'int'>
    20000000
    <class 'int'>
    3000000
    <class 'int'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    11000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    38000000
    <class 'int'>
    12000000
    <class 'int'>
    3200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    22000000
    <class 'int'>
    15000000
    <class 'int'>
    2000000
    <class 'int'>
    24000000
    <class 'int'>
    40000000
    <class 'int'>
    12500000
    <class 'int'>
    90000000
    <class 'int'>
    1800000
    <class 'int'>
    48000000
    <class 'int'>
    None
    <class 'NoneType'>
    22000000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    1987650
    <class 'int'>
    69000000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    7500000
    <class 'int'>
    15000000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    85000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    150000000
    <class 'int'>
    None
    <class 'NoneType'>
    120000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    28000000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    13000000
    <class 'int'>
    25000000
    <class 'int'>
    50000000
    <class 'int'>
    10000000
    <class 'int'>
    25000000
    <class 'int'>
    30000000
    <class 'int'>
    30000000
    <class 'int'>
    25000000
    <class 'int'>
    4500000
    <class 'int'>
    21000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    5000000
    <class 'int'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    18000000
    <class 'int'>
    None
    <class 'NoneType'>
    50000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000
    <class 'int'>
    5000000
    <class 'int'>
    8000000
    <class 'int'>
    20000000
    <class 'int'>
    22000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5773100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3500000
    <class 'int'>
    2500000
    <class 'int'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    5800000
    <class 'int'>
    4466000
    <class 'int'>
    None
    <class 'NoneType'>
    3100000
    <class 'int'>
    None
    <class 'NoneType'>
    1400000
    <class 'int'>
    None
    <class 'NoneType'>
    550000
    <class 'int'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    27000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    8000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    104002432
    <class 'int'>
    100000
    <class 'int'>
    None
    <class 'NoneType'>
    30000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4600000
    <class 'int'>
    4000000
    <class 'int'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30000
    <class 'int'>
    35000000
    <class 'int'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    120000000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    500000
    <class 'int'>
    30000
    <class 'int'>
    3000000
    <class 'int'>
    65000
    <class 'int'>
    12500000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    2500000
    <class 'int'>
    None
    <class 'NoneType'>
    650000
    <class 'int'>
    None
    <class 'NoneType'>
    13000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    425000000
    <class 'int'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    967686
    <class 'int'>
    None
    <class 'NoneType'>
    7347125
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1500000
    <class 'int'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4500000
    <class 'int'>
    2000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000000
    <class 'int'>
    None
    <class 'NoneType'>
    2300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7200000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3167000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60000000
    <class 'int'>
    300000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1100000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2400000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10000000
    <class 'int'>
    1250000
    <class 'int'>
    None
    <class 'NoneType'>
    10
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35000000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5000
    <class 'int'>
    None
    <class 'NoneType'>
    3
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    900000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    82470329
    <class 'int'>
    82470329
    <class 'int'>
    19635792
    <class 'int'>
    150541077
    <class 'int'>
    104724227
    <class 'int'>
    78543170
    <class 'int'>
    52362113
    <class 'int'>
    176722134
    <class 'int'>
    13745054
    <class 'int'>
    130905284
    <class 'int'>
    None
    <class 'NoneType'>
    14399581
    <class 'int'>
    86397487
    <class 'int'>
    117814756
    <class 'int'>
    20944845
    <class 'int'>
    54980219
    <class 'int'>
    43198743
    <class 'int'>
    2618105
    <class 'int'>
    117814756
    <class 'int'>
    58907378
    <class 'int'>
    85088435
    <class 'int'>
    17017687
    <class 'int'>
    104724227
    <class 'int'>
    174104028
    <class 'int'>
    95560857
    <class 'int'>
    222538983
    <class 'int'>
    209448455
    <class 'int'>
    130905284
    <class 'int'>
    31417268
    <class 'int'>
    91633699
    <class 'int'>
    78543170
    <class 'int'>
    27490109
    <class 'int'>
    117814756
    <class 'int'>
    91633699
    <class 'int'>
    13090528
    <class 'int'>
    91633699
    <class 'int'>
    7854317
    <class 'int'>
    13090528
    <class 'int'>
    7854317
    <class 'int'>
    130905284
    <class 'int'>
    32726
    <class 'int'>
    49744008
    <class 'int'>
    65452642
    <class 'int'>
    20944845
    <class 'int'>
    44769607
    <class 'int'>
    41889691
    <class 'int'>
    52362113
    <class 'int'>
    13090528
    <class 'int'>
    52362113
    <class 'int'>
    None
    <class 'NoneType'>
    104724227
    <class 'int'>
    98178963
    <class 'int'>
    45816849
    <class 'int'>
    104724227
    <class 'int'>
    104724227
    <class 'int'>
    78543170
    <class 'int'>
    37962532
    <class 'int'>
    15708634
    <class 'int'>
    None
    <class 'NoneType'>
    98178963
    <class 'int'>
    48434955
    <class 'int'>
    66761695
    <class 'int'>
    98178963
    <class 'int'>
    52362113
    <class 'int'>
    35344426
    <class 'int'>
    44507796
    <class 'int'>
    89015593
    <class 'int'>
    None
    <class 'NoneType'>
    104724227
    <class 'int'>
    8508843
    <class 'int'>
    32726321
    <class 'int'>
    104
    <class 'int'>
    None
    <class 'NoneType'>
    98178963
    <class 'int'>
    107342333
    <class 'int'>
    32726321
    <class 'int'>
    None
    <class 'NoneType'>
    24872004
    <class 'int'>
    6545264
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    62834536
    <class 'int'>
    83779382
    <class 'int'>
    19635792
    <class 'int'>
    30108215
    <class 'int'>
    85088435
    <class 'int'>
    27490109
    <class 'int'>
    28144636
    <class 'int'>
    13090528
    <class 'int'>
    None
    <class 'NoneType'>
    26181056
    <class 'int'>
    32726321
    <class 'int'>
    98178963
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58907378
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26181056
    <class 'int'>
    43198743
    <class 'int'>
    6545264
    <class 'int'>
    91633699
    <class 'int'>
    None
    <class 'NoneType'>
    104724227
    <class 'int'>
    30108215
    <class 'int'>
    98178963
    <class 'int'>
    85088435
    <class 'int'>
    14399581
    <class 'int'>
    13090528
    <class 'int'>
    32
    <class 'int'>
    65452642
    <class 'int'>
    10472422
    <class 'int'>
    14399581
    <class 'int'>
    None
    <class 'NoneType'>
    31417268
    <class 'int'>
    13090528
    <class 'int'>
    18326739
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22253898
    <class 'int'>
    None
    <class 'NoneType'>
    32726321
    <class 'int'>
    71997906
    <class 'int'>
    49744008
    <class 'int'>
    19635792
    <class 'int'>
    785431
    <class 'int'>
    9163369
    <class 'int'>
    71997906
    <class 'int'>
    17017687
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39271585
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13090528
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11781475
    <class 'int'>
    7854317
    <class 'int'>
    35344426
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    65452642
    <class 'int'>
    36653479
    <class 'int'>
    65452642
    <class 'int'>
    27490109
    <class 'int'>
    7854317
    <class 'int'>
    None
    <class 'NoneType'>
    7854317
    <class 'int'>
    15708634
    <class 'int'>
    98178963
    <class 'int'>
    13090528
    <class 'int'>
    None
    <class 'NoneType'>
    19635792
    <class 'int'>
    78543170
    <class 'int'>
    45816849
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26181056
    <class 'int'>
    31417268
    <class 'int'>
    30108215
    <class 'int'>
    None
    <class 'NoneType'>
    2225389
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    49744008
    <class 'int'>
    None
    <class 'NoneType'>
    3927158
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17017687
    <class 'int'>
    None
    <class 'NoneType'>
    32726321
    <class 'int'>
    None
    <class 'NoneType'>
    18326739
    <class 'int'>
    13090528
    <class 'int'>
    None
    <class 'NoneType'>
    9163369
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15708634
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5563474
    <class 'int'>
    3272632
    <class 'int'>
    91633699
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    589073
    <class 'int'>
    15708634
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4974400
    <class 'int'>
    1
    <class 'int'>
    12056376
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15708634
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    114528394
    <class 'int'>
    153936014
    <class 'int'>
    18472321
    <class 'int'>
    7388928
    <class 'int'>
    141621133
    <class 'int'>
    123148811
    <class 'int'>
    73889287
    <class 'int'>
    30787202
    <class 'int'>
    104676489
    <class 'int'>
    172408336
    <class 'int'>
    120685835
    <class 'int'>
    147778574
    <class 'int'>
    141621133
    <class 'int'>
    73889287
    <class 'int'>
    36944643
    <class 'int'>
    45565060
    <class 'int'>
    64653126
    <class 'int'>
    123148811
    <class 'int'>
    25861250
    <class 'int'>
    65268870
    <class 'int'>
    86204168
    <class 'int'>
    113296906
    <class 'int'>
    22166786
    <class 'int'>
    80046727
    <class 'int'>
    34481667
    <class 'int'>
    19703809
    <class 'int'>
    55416965
    <class 'int'>
    110833930
    <class 'int'>
    55416965
    <class 'int'>
    20935297
    <class 'int'>
    30787202
    <class 'int'>
    27092738
    <class 'int'>
    34481667
    <class 'int'>
    73889287
    <class 'int'>
    83741191
    <class 'int'>
    None
    <class 'NoneType'>
    30787202
    <class 'int'>
    113296906
    <class 'int'>
    125611787
    <class 'int'>
    49259524
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36944643
    <class 'int'>
    168713871
    <class 'int'>
    None
    <class 'NoneType'>
    51722500
    <class 'int'>
    86204168
    <class 'int'>
    49259524
    <class 'int'>
    12314881
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43102084
    <class 'int'>
    88667144
    <class 'int'>
    73889287
    <class 'int'>
    3694464
    <class 'int'>
    43102084
    <class 'int'>
    59111429
    <class 'int'>
    24383464
    <class 'int'>
    13546369
    <class 'int'>
    70194822
    <class 'int'>
    28324226
    <class 'int'>
    27092738
    <class 'int'>
    None
    <class 'NoneType'>
    20935297
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70194822
    <class 'int'>
    None
    <class 'NoneType'>
    59111429
    <class 'int'>
    28324226
    <class 'int'>
    None
    <class 'NoneType'>
    61574405
    <class 'int'>
    None
    <class 'NoneType'>
    26045973
    <class 'int'>
    13546369
    <class 'int'>
    None
    <class 'NoneType'>
    65268870
    <class 'int'>
    22166786
    <class 'int'>
    None
    <class 'NoneType'>
    38176131
    <class 'int'>
    None
    <class 'NoneType'>
    36944643
    <class 'int'>
    73889287
    <class 'int'>
    None
    <class 'NoneType'>
    39407619
    <class 'int'>
    98519049
    <class 'int'>
    12314881
    <class 'int'>
    43102084
    <class 'int'>
    131769228
    <class 'int'>
    None
    <class 'NoneType'>
    1847232
    <class 'int'>
    83741191
    <class 'int'>
    4925952
    <class 'int'>
    21797339
    <class 'int'>
    None
    <class 'NoneType'>
    61574405
    <class 'int'>
    25861250
    <class 'int'>
    49259524
    <class 'int'>
    14777857
    <class 'int'>
    None
    <class 'NoneType'>
    13546369
    <class 'int'>
    None
    <class 'NoneType'>
    88667144
    <class 'int'>
    76352263
    <class 'int'>
    None
    <class 'NoneType'>
    123148
    <class 'int'>
    None
    <class 'NoneType'>
    55416965
    <class 'int'>
    2560263
    <class 'int'>
    34481667
    <class 'int'>
    27092738
    <class 'int'>
    None
    <class 'NoneType'>
    59111429
    <class 'int'>
    12314881
    <class 'int'>
    92361608
    <class 'int'>
    None
    <class 'NoneType'>
    60342917
    <class 'int'>
    18472321
    <class 'int'>
    17240833
    <class 'int'>
    None
    <class 'NoneType'>
    46796548
    <class 'int'>
    7388928
    <class 'int'>
    16009345
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2093529
    <class 'int'>
    43102084
    <class 'int'>
    43102084
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    92361608
    <class 'int'>
    27092738
    <class 'int'>
    18472321
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16009345
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    115759882
    <class 'int'>
    29555714
    <class 'int'>
    None
    <class 'NoneType'>
    615744
    <class 'int'>
    None
    <class 'NoneType'>
    43102084
    <class 'int'>
    107139466
    <class 'int'>
    92361608
    <class 'int'>
    None
    <class 'NoneType'>
    16009345
    <class 'int'>
    12314881
    <class 'int'>
    None
    <class 'NoneType'>
    49259524
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1847232
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    41870595
    <class 'int'>
    6157440
    <class 'int'>
    None
    <class 'NoneType'>
    8620416
    <class 'int'>
    43102084
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1847232
    <class 'int'>
    14777857
    <class 'int'>
    92361608
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    307872
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13546369
    <class 'int'>
    46796548
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    114528394
    <class 'int'>
    None
    <class 'NoneType'>
    27092738
    <class 'int'>
    70194822
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1231488
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    49259524
    <class 'int'>
    8620416
    <class 'int'>
    6157440
    <class 'int'>
    None
    <class 'NoneType'>
    862041
    <class 'int'>
    30787202
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7388928
    <class 'int'>
    13546369
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12314881
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8620416
    <class 'int'>
    18472321
    <class 'int'>
    6157440
    <class 'int'>
    None
    <class 'NoneType'>
    5911142
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19703809
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    203195
    <class 'int'>
    187365527
    <class 'int'>
    182301594
    <class 'int'>
    141790128
    <class 'int'>
    25319665
    <class 'int'>
    187365527
    <class 'int'>
    37473105
    <class 'int'>
    202557326
    <class 'int'>
    131662262
    <class 'int'>
    19242946
    <class 'int'>
    52664904
    <class 'int'>
    151917995
    <class 'int'>
    146854061
    <class 'int'>
    151917995
    <class 'int'>
    33421958
    <class 'int'>
    151917995
    <class 'int'>
    86086863
    <class 'int'>
    86086863
    <class 'int'>
    45575398
    <class 'int'>
    75958997
    <class 'int'>
    48613758
    <class 'int'>
    15191799
    <class 'int'>
    30383599
    <class 'int'>
    75958997
    <class 'int'>
    106342596
    <class 'int'>
    151917995
    <class 'int'>
    15191799
    <class 'int'>
    35447532
    <class 'int'>
    227876992
    <class 'int'>
    93176370
    <class 'int'>
    81022930
    <class 'int'>
    12659832
    <class 'int'>
    50639331
    <class 'int'>
    45575398
    <class 'int'>
    23294092
    <class 'int'>
    81022930
    <class 'int'>
    9115079
    <class 'int'>
    55703264
    <class 'int'>
    25319665
    <class 'int'>
    151917995
    <class 'int'>
    30383599
    <class 'int'>
    131662262
    <class 'int'>
    70895064
    <class 'int'>
    60767198
    <class 'int'>
    70895064
    <class 'int'>
    37473105
    <class 'int'>
    30383599
    <class 'int'>
    60767198
    <class 'int'>
    35447532
    <class 'int'>
    20255732
    <class 'int'>
    65831131
    <class 'int'>
    None
    <class 'NoneType'>
    70895064
    <class 'int'>
    20255732
    <class 'int'>
    27345239
    <class 'int'>
    6076719
    <class 'int'>
    81022930
    <class 'int'>
    20762125
    <class 'int'>
    35447532
    <class 'int'>
    35447532
    <class 'int'>
    40511465
    <class 'int'>
    24306879
    <class 'int'>
    81022930
    <class 'int'>
    28358025
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32409172
    <class 'int'>
    70895064
    <class 'int'>
    None
    <class 'NoneType'>
    15191799
    <class 'int'>
    55703264
    <class 'int'>
    91150797
    <class 'int'>
    12153439
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2937081
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20255732
    <class 'int'>
    9115079
    <class 'int'>
    81022930
    <class 'int'>
    5063933
    <class 'int'>
    7089506
    <class 'int'>
    25319665
    <class 'int'>
    None
    <class 'NoneType'>
    8608686
    <class 'int'>
    10938095
    <class 'int'>
    62792771
    <class 'int'>
    121534396
    <class 'int'>
    15191799
    <class 'int'>
    86086863
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55703264
    <class 'int'>
    30383599
    <class 'int'>
    None
    <class 'NoneType'>
    60767198
    <class 'int'>
    15191799
    <class 'int'>
    22281305
    <class 'int'>
    91150797
    <class 'int'>
    32409172
    <class 'int'>
    None
    <class 'NoneType'>
    35447532
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9115079
    <class 'int'>
    7089506
    <class 'int'>
    20255732
    <class 'int'>
    50639331
    <class 'int'>
    30383599
    <class 'int'>
    20255732
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55703264
    <class 'int'>
    40511465
    <class 'int'>
    35447532
    <class 'int'>
    None
    <class 'NoneType'>
    25319665
    <class 'int'>
    18230159
    <class 'int'>
    12153439
    <class 'int'>
    13166226
    <class 'int'>
    45575398
    <class 'int'>
    30383599
    <class 'int'>
    25319665
    <class 'int'>
    40511465
    <class 'int'>
    21268519
    <class 'int'>
    None
    <class 'NoneType'>
    25319665
    <class 'int'>
    60767198
    <class 'int'>
    None
    <class 'NoneType'>
    11647046
    <class 'int'>
    None
    <class 'NoneType'>
    20255732
    <class 'int'>
    18230159
    <class 'int'>
    60767198
    <class 'int'>
    30383599
    <class 'int'>
    12153439
    <class 'int'>
    5063933
    <class 'int'>
    None
    <class 'NoneType'>
    2025573
    <class 'int'>
    8
    <class 'int'>
    None
    <class 'NoneType'>
    65831131
    <class 'int'>
    15191799
    <class 'int'>
    None
    <class 'NoneType'>
    60767198
    <class 'int'>
    None
    <class 'NoneType'>
    6177998
    <class 'int'>
    35447532
    <class 'int'>
    11140652
    <class 'int'>
    10127866
    <class 'int'>
    15191799
    <class 'int'>
    20255732
    <class 'int'>
    20255732
    <class 'int'>
    35447532
    <class 'int'>
    None
    <class 'NoneType'>
    5063933
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8102293
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8608686
    <class 'int'>
    None
    <class 'NoneType'>
    30383599
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6076719
    <class 'int'>
    30383599
    <class 'int'>
    None
    <class 'NoneType'>
    7089506
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20255732
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20255732
    <class 'int'>
    60767198
    <class 'int'>
    None
    <class 'NoneType'>
    12153439
    <class 'int'>
    None
    <class 'NoneType'>
    3544753
    <class 'int'>
    8102293
    <class 'int'>
    20255732
    <class 'int'>
    None
    <class 'NoneType'>
    1
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6076719
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3544753
    <class 'int'>
    12153439
    <class 'int'>
    25319665
    <class 'int'>
    27345239
    <class 'int'>
    40511465
    <class 'int'>
    None
    <class 'NoneType'>
    3544753
    <class 'int'>
    16204586
    <class 'int'>
    10127866
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2025573
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10127866
    <class 'int'>
    37473105
    <class 'int'>
    None
    <class 'NoneType'>
    20255732
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    151917
    <class 'int'>
    18230159
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1012786
    <class 'int'>
    14280291
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8483911
    <class 'int'>
    None
    <class 'NoneType'>
    30383599
    <class 'int'>
    None
    <class 'NoneType'>
    3038359
    <class 'int'>
    None
    <class 'NoneType'>
    5063933
    <class 'int'>
    6076719
    <class 'int'>
    25319665
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10127866
    <class 'int'>
    12153439
    <class 'int'>
    None
    <class 'NoneType'>
    10127866
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15596914
    <class 'int'>
    12153439
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    253196
    <class 'int'>
    11647046
    <class 'int'>
    25319665
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45575398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6076719
    <class 'int'>
    20255732
    <class 'int'>
    19242946
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15191799
    <class 'int'>
    None
    <class 'NoneType'>
    8102293
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2025573
    <class 'int'>
    None
    <class 'NoneType'>
    21268519
    <class 'int'>
    None
    <class 'NoneType'>
    759589
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1519179
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4760097
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25420944
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    506393
    <class 'int'>
    506393
    <class 'int'>
    14179012
    <class 'int'>
    None
    <class 'NoneType'>
    7089506
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12153439
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7089506
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37473105
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25319665
    <class 'int'>
    None
    <class 'NoneType'>
    14179012
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27345239
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8102293
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4051146
    <class 'int'>
    None
    <class 'NoneType'>
    6076719
    <class 'int'>
    202557
    <class 'int'>
    6785670
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35447532
    <class 'int'>
    1519179
    <class 'int'>
    None
    <class 'NoneType'>
    4354982
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5823523
    <class 'int'>
    9216358
    <class 'int'>
    21268519
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    202557
    <class 'int'>
    2531966
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2329409
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20255
    <class 'int'>
    8102293
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2531966
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15191799
    <class 'int'>
    5063933
    <class 'int'>
    None
    <class 'NoneType'>
    10127866
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3646031
    <class 'int'>
    None
    <class 'NoneType'>
    522383
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6076719
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4233448
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2531966
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4658818
    <class 'int'>
    15191799
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2025573
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11140652
    <class 'int'>
    3038359
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5063933
    <class 'int'>
    None
    <class 'NoneType'>
    22281305
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    172173
    <class 'int'>
    10127866
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7089506
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3544753
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8102293
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    506393
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    135715725
    <class 'int'>
    14540970
    <class 'int'>
    121174755
    <class 'int'>
    368371256
    <class 'int'>
    90154017
    <class 'int'>
    145409706
    <class 'int'>
    38775921
    <class 'int'>
    48469902
    <class 'int'>
    38775921
    <class 'int'>
    31020737
    <class 'int'>
    140562716
    <class 'int'>
    48469902
    <class 'int'>
    87245823
    <class 'int'>
    26173747
    <class 'int'>
    145409706
    <class 'int'>
    77551843
    <class 'int'>
    33928931
    <class 'int'>
    106633784
    <class 'int'>
    33928931
    <class 'int'>
    19387960
    <class 'int'>
    121174755
    <class 'int'>
    193879608
    <class 'int'>
    193879608
    <class 'int'>
    77551843
    <class 'int'>
    24234951
    <class 'int'>
    126021745
    <class 'int'>
    31505436
    <class 'int'>
    34898329
    <class 'int'>
    48469902
    <class 'int'>
    79490639
    <class 'int'>
    40714717
    <class 'int'>
    106633784
    <class 'int'>
    6301087
    <class 'int'>
    None
    <class 'NoneType'>
    53316892
    <class 'int'>
    31020737
    <class 'int'>
    48469902
    <class 'int'>
    87245823
    <class 'int'>
    14540970
    <class 'int'>
    77551843
    <class 'int'>
    158011880
    <class 'int'>
    None
    <class 'NoneType'>
    63980270
    <class 'int'>
    7755184
    <class 'int'>
    130868735
    <class 'int'>
    50408698
    <class 'int'>
    164797667
    <class 'int'>
    38775921
    <class 'int'>
    55255688
    <class 'int'>
    14540970
    <class 'int'>
    63980270
    <class 'int'>
    35867727
    <class 'int'>
    29081941
    <class 'int'>
    72704853
    <class 'int'>
    None
    <class 'NoneType'>
    7755184
    <class 'int'>
    None
    <class 'NoneType'>
    29081941
    <class 'int'>
    58163882
    <class 'int'>
    None
    <class 'NoneType'>
    11632776
    <class 'int'>
    24234951
    <class 'int'>
    33928931
    <class 'int'>
    29081941
    <class 'int'>
    48663781
    <class 'int'>
    72704853
    <class 'int'>
    242349
    <class 'int'>
    29081941
    <class 'int'>
    38775921
    <class 'int'>
    6785786
    <class 'int'>
    None
    <class 'NoneType'>
    38775921
    <class 'int'>
    67857862
    <class 'int'>
    45561708
    <class 'int'>
    29081941
    <class 'int'>
    76582445
    <class 'int'>
    19387960
    <class 'int'>
    38775921
    <class 'int'>
    6204147
    <class 'int'>
    5816388
    <class 'int'>
    48469902
    <class 'int'>
    None
    <class 'NoneType'>
    29081941
    <class 'int'>
    24234951
    <class 'int'>
    24234951
    <class 'int'>
    48469902
    <class 'int'>
    6785786
    <class 'int'>
    15510368
    <class 'int'>
    126021745
    <class 'int'>
    29081941
    <class 'int'>
    116327765
    <class 'int'>
    33928931
    <class 'int'>
    87245823
    <class 'int'>
    None
    <class 'NoneType'>
    43622911
    <class 'int'>
    58163882
    <class 'int'>
    None
    <class 'NoneType'>
    39745319
    <class 'int'>
    16479766
    <class 'int'>
    18418562
    <class 'int'>
    33928931
    <class 'int'>
    72704853
    <class 'int'>
    54286290
    <class 'int'>
    48372962
    <class 'int'>
    3392893
    <class 'int'>
    3877592
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4846990
    <class 'int'>
    16479766
    <class 'int'>
    126021745
    <class 'int'>
    38775921
    <class 'int'>
    145409706
    <class 'int'>
    13911376
    <class 'int'>
    20357358
    <class 'int'>
    None
    <class 'NoneType'>
    38775921
    <class 'int'>
    None
    <class 'NoneType'>
    72704853
    <class 'int'>
    20357358
    <class 'int'>
    24234951
    <class 'int'>
    9693980
    <class 'int'>
    106633784
    <class 'int'>
    None
    <class 'NoneType'>
    29081941
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12086831
    <class 'int'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    4846990
    <class 'int'>
    3392893
    <class 'int'>
    34898329
    <class 'int'>
    15292203
    <class 'int'>
    None
    <class 'NoneType'>
    19387960
    <class 'int'>
    61072076
    <class 'int'>
    43622911
    <class 'int'>
    9693980
    <class 'int'>
    5816388
    <class 'int'>
    None
    <class 'NoneType'>
    9693980
    <class 'int'>
    9693980
    <class 'int'>
    23265553
    <class 'int'>
    None
    <class 'NoneType'>
    34898329
    <class 'int'>
    24234951
    <class 'int'>
    96939
    <class 'int'>
    12117475
    <class 'int'>
    None
    <class 'NoneType'>
    189032618
    <class 'int'>
    4846990
    <class 'int'>
    12602174
    <class 'int'>
    67857862
    <class 'int'>
    18418562
    <class 'int'>
    None
    <class 'NoneType'>
    14540970
    <class 'int'>
    None
    <class 'NoneType'>
    3877592
    <class 'int'>
    43622911
    <class 'int'>
    26173747
    <class 'int'>
    9209281
    <class 'int'>
    14540970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16479766
    <class 'int'>
    None
    <class 'NoneType'>
    38775921
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24234951
    <class 'int'>
    24234951
    <class 'int'>
    1938796
    <class 'int'>
    None
    <class 'NoneType'>
    14540970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5816388
    <class 'int'>
    727048
    <class 'int'>
    7755184
    <class 'int'>
    4846990
    <class 'int'>
    None
    <class 'NoneType'>
    7464364
    <class 'int'>
    33928931
    <class 'int'>
    29081941
    <class 'int'>
    None
    <class 'NoneType'>
    23265553
    <class 'int'>
    14540970
    <class 'int'>
    17449164
    <class 'int'>
    27143145
    <class 'int'>
    4651399
    <class 'int'>
    29081941
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6785786
    <class 'int'>
    None
    <class 'NoneType'>
    19387960
    <class 'int'>
    None
    <class 'NoneType'>
    7755184
    <class 'int'>
    36837125
    <class 'int'>
    48469902
    <class 'int'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    3877592
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11632776
    <class 'int'>
    None
    <class 'NoneType'>
    4846990
    <class 'int'>
    7755184
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4185808
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38775921
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31020737
    <class 'int'>
    969398
    <class 'int'>
    339289
    <class 'int'>
    3392893
    <class 'int'>
    90
    <class 'int'>
    None
    <class 'NoneType'>
    35867727
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28112543
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4362291
    <class 'int'>
    121174755
    <class 'int'>
    484699
    <class 'int'>
    2423495
    <class 'int'>
    None
    <class 'NoneType'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    3877592
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6398027
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1938796
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7076605
    <class 'int'>
    7755184
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    121174
    <class 'int'>
    4846990
    <class 'int'>
    None
    <class 'NoneType'>
    19387960
    <class 'int'>
    10499717
    <class 'int'>
    None
    <class 'NoneType'>
    19387960
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19387960
    <class 'int'>
    13571572
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16479766
    <class 'int'>
    None
    <class 'NoneType'>
    484699
    <class 'int'>
    130868
    <class 'int'>
    None
    <class 'NoneType'>
    901540
    <class 'int'>
    None
    <class 'NoneType'>
    12602174
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3877592
    <class 'int'>
    484699
    <class 'int'>
    18515502
    <class 'int'>
    6785786
    <class 'int'>
    None
    <class 'NoneType'>
    1066337
    <class 'int'>
    None
    <class 'NoneType'>
    8142943
    <class 'int'>
    None
    <class 'NoneType'>
    24234951
    <class 'int'>
    3877592
    <class 'int'>
    None
    <class 'NoneType'>
    1114807
    <class 'int'>
    None
    <class 'NoneType'>
    2908194
    <class 'int'>
    None
    <class 'NoneType'>
    27143145
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39745319
    <class 'int'>
    2908194
    <class 'int'>
    24234951
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14540970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5816388
    <class 'int'>
    14540970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24234951
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7755184
    <class 'int'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6752415
    <class 'int'>
    None
    <class 'NoneType'>
    9693980
    <class 'int'>
    7755184
    <class 'int'>
    None
    <class 'NoneType'>
    16479766
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    969398
    <class 'int'>
    24234951
    <class 'int'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    96939
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    129904
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    67857
    <class 'int'>
    None
    <class 'NoneType'>
    484699
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8703294
    <class 'int'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5816388
    <class 'int'>
    7755184
    <class 'int'>
    None
    <class 'NoneType'>
    7755
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    484699
    <class 'int'>
    None
    <class 'NoneType'>
    24234951
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2
    <class 'int'>
    12602174
    <class 'int'>
    581638
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24234951
    <class 'int'>
    14540970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4071471
    <class 'int'>
    630108
    <class 'int'>
    16479
    <class 'int'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    2908194
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43818
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    155103686
    <class 'int'>
    1938796
    <class 'int'>
    None
    <class 'NoneType'>
    969398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4846990
    <class 'int'>
    1454097
    <class 'int'>
    None
    <class 'NoneType'>
    3877592
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    242349
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1938796
    <class 'int'>
    29081941
    <class 'int'>
    None
    <class 'NoneType'>
    1298993
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    94
    <class 'int'>
    8724582
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9693980
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6591906
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2423495
    <class 'int'>
    None
    <class 'NoneType'>
    727048
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    678578
    <class 'int'>
    None
    <class 'NoneType'>
    290819
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2813832
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19387960
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    242349
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    59843
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    95768650
    <class 'int'>
    121226139
    <class 'int'>
    71523422
    <class 'int'>
    72735683
    <class 'int'>
    40004625
    <class 'int'>
    168504333
    <class 'int'>
    145471367
    <class 'int'>
    63037592
    <class 'int'>
    169716595
    <class 'int'>
    42429148
    <class 'int'>
    123650662
    <class 'int'>
    96980911
    <class 'int'>
    25457489
    <class 'int'>
    84858297
    <class 'int'>
    169716595
    <class 'int'>
    169716595
    <class 'int'>
    6061306
    <class 'int'>
    66674376
    <class 'int'>
    65462115
    <class 'int'>
    121226139
    <class 'int'>
    58188546
    <class 'int'>
    145471367
    <class 'int'>
    6061306
    <class 'int'>
    49702717
    <class 'int'>
    60613069
    <class 'int'>
    4243107
    <class 'int'>
    6061306
    <class 'int'>
    72735683
    <class 'int'>
    87282820
    <class 'int'>
    58188546
    <class 'int'>
    None
    <class 'NoneType'>
    13334875
    <class 'int'>
    6061306
    <class 'int'>
    32731057
    <class 'int'>
    None
    <class 'NoneType'>
    103042218
    <class 'int'>
    15759398
    <class 'int'>
    72735683
    <class 'int'>
    24245227
    <class 'int'>
    9698091
    <class 'int'>
    30306534
    <class 'int'>
    96980911
    <class 'int'>
    76372467
    <class 'int'>
    78796990
    <class 'int'>
    55764024
    <class 'int'>
    42429148
    <class 'int'>
    101829957
    <class 'int'>
    None
    <class 'NoneType'>
    96980911
    <class 'int'>
    66674376
    <class 'int'>
    None
    <class 'NoneType'>
    72735683
    <class 'int'>
    84858297
    <class 'int'>
    4849045
    <class 'int'>
    18183920
    <class 'int'>
    1575939
    <class 'int'>
    None
    <class 'NoneType'>
    82
    <class 'int'>
    33
    <class 'int'>
    90919604
    <class 'int'>
    6061306
    <class 'int'>
    None
    <class 'NoneType'>
    14547136
    <class 'int'>
    60613069
    <class 'int'>
    23032966
    <class 'int'>
    36367841
    <class 'int'>
    14547136
    <class 'int'>
    18183920
    <class 'int'>
    96980911
    <class 'int'>
    24245227
    <class 'int'>
    52127239
    <class 'int'>
    None
    <class 'NoneType'>
    139410060
    <class 'int'>
    43641410
    <class 'int'>
    None
    <class 'NoneType'>
    20608443
    <class 'int'>
    None
    <class 'NoneType'>
    60613069
    <class 'int'>
    54551762
    <class 'int'>
    30306534
    <class 'int'>
    121226139
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42429148
    <class 'int'>
    46065932
    <class 'int'>
    None
    <class 'NoneType'>
    46065932
    <class 'int'>
    24245227
    <class 'int'>
    24245227
    <class 'int'>
    None
    <class 'NoneType'>
    48490455
    <class 'int'>
    16971659
    <class 'int'>
    14547136
    <class 'int'>
    8485829
    <class 'int'>
    54551762
    <class 'int'>
    14547136
    <class 'int'>
    42429148
    <class 'int'>
    None
    <class 'NoneType'>
    48490455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    72735683
    <class 'int'>
    103042218
    <class 'int'>
    None
    <class 'NoneType'>
    50914978
    <class 'int'>
    25457489
    <class 'int'>
    84858297
    <class 'int'>
    36367841
    <class 'int'>
    30306534
    <class 'int'>
    60613069
    <class 'int'>
    56976285
    <class 'int'>
    12122613
    <class 'int'>
    42429148
    <class 'int'>
    36367841
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15759398
    <class 'int'>
    38792364
    <class 'int'>
    None
    <class 'NoneType'>
    40004625
    <class 'int'>
    16365528
    <class 'int'>
    24245227
    <class 'int'>
    84858297
    <class 'int'>
    6061306
    <class 'int'>
    36367841
    <class 'int'>
    4849045
    <class 'int'>
    3636784
    <class 'int'>
    3636784
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    84858297
    <class 'int'>
    None
    <class 'NoneType'>
    36367841
    <class 'int'>
    None
    <class 'NoneType'>
    19396182
    <class 'int'>
    2424522
    <class 'int'>
    31518796
    <class 'int'>
    15759398
    <class 'int'>
    21820705
    <class 'int'>
    None
    <class 'NoneType'>
    14547136
    <class 'int'>
    18183920
    <class 'int'>
    None
    <class 'NoneType'>
    7879699
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20608443
    <class 'int'>
    4606593
    <class 'int'>
    10910352
    <class 'int'>
    None
    <class 'NoneType'>
    9698091
    <class 'int'>
    7273568
    <class 'int'>
    32731057
    <class 'int'>
    15153267
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10910352
    <class 'int'>
    13334875
    <class 'int'>
    12122613
    <class 'int'>
    None
    <class 'NoneType'>
    24245227
    <class 'int'>
    None
    <class 'NoneType'>
    606130
    <class 'int'>
    14547136
    <class 'int'>
    121226139
    <class 'int'>
    36367841
    <class 'int'>
    30306534
    <class 'int'>
    13334875
    <class 'int'>
    12122613
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    94556388
    <class 'int'>
    None
    <class 'NoneType'>
    35155580
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    72735683
    <class 'int'>
    242452
    <class 'int'>
    24245227
    <class 'int'>
    None
    <class 'NoneType'>
    36367841
    <class 'int'>
    4242914
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15759398
    <class 'int'>
    1454713
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16971659
    <class 'int'>
    None
    <class 'NoneType'>
    48490455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42429148
    <class 'int'>
    8485829
    <class 'int'>
    12122613
    <class 'int'>
    None
    <class 'NoneType'>
    1212261
    <class 'int'>
    30306534
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18911277
    <class 'int'>
    3636784
    <class 'int'>
    2424522
    <class 'int'>
    4849045
    <class 'int'>
    None
    <class 'NoneType'>
    66674376
    <class 'int'>
    48490455
    <class 'int'>
    12122613
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16971659
    <class 'int'>
    10304221
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30306534
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24245227
    <class 'int'>
    13334875
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4849045
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    48490
    <class 'int'>
    11768892
    <class 'int'>
    36777789
    <class 'int'>
    80911135
    <class 'int'>
    66200020
    <class 'int'>
    44133346
    <class 'int'>
    25008896
    <class 'int'>
    44133346
    <class 'int'>
    88266693
    <class 'int'>
    None
    <class 'NoneType'>
    169177829
    <class 'int'>
    33835565
    <class 'int'>
    None
    <class 'NoneType'>
    80911135
    <class 'int'>
    91208916
    <class 'int'>
    50017793
    <class 'int'>
    39720
    <class 'int'>
    73555578
    <class 'int'>
    32364454
    <class 'int'>
    44133346
    <class 'int'>
    None
    <class 'NoneType'>
    8826669
    <class 'int'>
    41191123
    <class 'int'>
    None
    <class 'NoneType'>
    66200020
    <class 'int'>
    22066673
    <class 'int'>
    58844462
    <class 'int'>
    67671131
    <class 'int'>
    None
    <class 'NoneType'>
    55902239
    <class 'int'>
    22066673
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80911135
    <class 'int'>
    88266693
    <class 'int'>
    20595561
    <class 'int'>
    36777789
    <class 'int'>
    7355557
    <class 'int'>
    51488904
    <class 'int'>
    39720012
    <class 'int'>
    17653338
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58844462
    <class 'int'>
    26480008
    <class 'int'>
    39720012
    <class 'int'>
    92680028
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    73555578
    <class 'int'>
    None
    <class 'NoneType'>
    4413334
    <class 'int'>
    51488904
    <class 'int'>
    66200020
    <class 'int'>
    None
    <class 'NoneType'>
    41191123
    <class 'int'>
    38248900
    <class 'int'>
    73555578
    <class 'int'>
    None
    <class 'NoneType'>
    2206667
    <class 'int'>
    None
    <class 'NoneType'>
    44133346
    <class 'int'>
    66200020
    <class 'int'>
    58844462
    <class 'int'>
    14711115
    <class 'int'>
    25008896
    <class 'int'>
    22066673
    <class 'int'>
    38248900
    <class 'int'>
    44133346
    <class 'int'>
    16917782
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    33835565
    <class 'int'>
    29422231
    <class 'int'>
    26480008
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    66200020
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22066673
    <class 'int'>
    None
    <class 'NoneType'>
    36777789
    <class 'int'>
    None
    <class 'NoneType'>
    29422231
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3677778
    <class 'int'>
    None
    <class 'NoneType'>
    19124450
    <class 'int'>
    1029778
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5148890
    <class 'int'>
    8826669
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16182227
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58844462
    <class 'int'>
    None
    <class 'NoneType'>
    25126585
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1471111
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36777789
    <class 'int'>
    None
    <class 'NoneType'>
    66200020
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    588444
    <class 'int'>
    2942223
    <class 'int'>
    294222
    <class 'int'>
    None
    <class 'NoneType'>
    73555578
    <class 'int'>
    882666
    <class 'int'>
    None
    <class 'NoneType'>
    10886225
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    66200020
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19124450
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3677778
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15152449
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    29422231
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13240004
    <class 'int'>
    None
    <class 'NoneType'>
    47
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    208943741
    <class 'int'>
    66482099
    <class 'int'>
    237436070
    <class 'int'>
    94974428
    <class 'int'>
    189948856
    <class 'int'>
    123466756
    <class 'int'>
    237436070
    <class 'int'>
    123466756
    <class 'int'>
    19944629
    <class 'int'>
    204195020
    <class 'int'>
    175702692
    <class 'int'>
    161456527
    <class 'int'>
    39889259
    <class 'int'>
    113969313
    <class 'int'>
    90225706
    <class 'int'>
    47487214
    <class 'int'>
    56984656
    <class 'int'>
    39889259
    <class 'int'>
    156707806
    <class 'int'>
    16145652
    <class 'int'>
    213692463
    <class 'int'>
    71230821
    <class 'int'>
    118718035
    <class 'int'>
    96873916
    <class 'int'>
    94974428
    <class 'int'>
    42738492
    <class 'int'>
    142461642
    <class 'int'>
    12346675
    <class 'int'>
    42263620
    <class 'int'>
    47487214
    <class 'int'>
    6648209
    <class 'int'>
    28492328
    <class 'int'>
    137712920
    <class 'int'>
    28492328
    <class 'int'>
    75029798
    <class 'int'>
    11871803
    <class 'int'>
    28492328
    <class 'int'>
    65532355
    <class 'int'>
    56984656
    <class 'int'>
    3324104
    <class 'int'>
    66482099
    <class 'int'>
    15195908
    <class 'int'>
    None
    <class 'NoneType'>
    47487214
    <class 'int'>
    198496554
    <class 'int'>
    10447187
    <class 'int'>
    None
    <class 'NoneType'>
    61733378
    <class 'int'>
    57934401
    <class 'int'>
    80728263
    <class 'int'>
    246933513
    <class 'int'>
    28492328
    <class 'int'>
    113969313
    <class 'int'>
    37989771
    <class 'int'>
    21844118
    <class 'int'>
    142461642
    <class 'int'>
    31341561
    <class 'int'>
    14246164
    <class 'int'>
    11396931
    <class 'int'>
    80728263
    <class 'int'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    6648209
    <class 'int'>
    None
    <class 'NoneType'>
    61733378
    <class 'int'>
    37989771
    <class 'int'>
    61733378
    <class 'int'>
    7123082
    <class 'int'>
    23743607
    <class 'int'>
    11396931
    <class 'int'>
    23743607
    <class 'int'>
    None
    <class 'NoneType'>
    137712920
    <class 'int'>
    None
    <class 'NoneType'>
    24693351
    <class 'int'>
    None
    <class 'NoneType'>
    24693351
    <class 'int'>
    71230821
    <class 'int'>
    33241049
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    64582611
    <class 'int'>
    56984656
    <class 'int'>
    30391817
    <class 'int'>
    23743607
    <class 'int'>
    28492328
    <class 'int'>
    33241049
    <class 'int'>
    18994885
    <class 'int'>
    33241049
    <class 'int'>
    16145652
    <class 'int'>
    37040026
    <class 'int'>
    66482099
    <class 'int'>
    80728263
    <class 'int'>
    42738492
    <class 'int'>
    33241049
    <class 'int'>
    3543021
    <class 'int'>
    None
    <class 'NoneType'>
    29442072
    <class 'int'>
    39889259
    <class 'int'>
    14246164
    <class 'int'>
    37989771
    <class 'int'>
    8547698
    <class 'int'>
    None
    <class 'NoneType'>
    5698465
    <class 'int'>
    9497442
    <class 'int'>
    15195908
    <class 'int'>
    4748721
    <class 'int'>
    None
    <class 'NoneType'>
    13296419
    <class 'int'>
    18994885
    <class 'int'>
    14246164
    <class 'int'>
    18994885
    <class 'int'>
    None
    <class 'NoneType'>
    11396931
    <class 'int'>
    18994885
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    61733378
    <class 'int'>
    13296419
    <class 'int'>
    28492328
    <class 'int'>
    28492328
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14246164
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37989771
    <class 'int'>
    None
    <class 'NoneType'>
    28492328
    <class 'int'>
    3798977
    <class 'int'>
    None
    <class 'NoneType'>
    16145652
    <class 'int'>
    None
    <class 'NoneType'>
    2374360
    <class 'int'>
    None
    <class 'NoneType'>
    6553235
    <class 'int'>
    42738492
    <class 'int'>
    37989771
    <class 'int'>
    10447187
    <class 'int'>
    25025761
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    56984656
    <class 'int'>
    14246164
    <class 'int'>
    20894374
    <class 'int'>
    18520013
    <class 'int'>
    None
    <class 'NoneType'>
    4748721
    <class 'int'>
    11396931
    <class 'int'>
    55085168
    <class 'int'>
    9497442
    <class 'int'>
    None
    <class 'NoneType'>
    13296419
    <class 'int'>
    33241049
    <class 'int'>
    1424616
    <class 'int'>
    None
    <class 'NoneType'>
    33241049
    <class 'int'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    37989771
    <class 'int'>
    None
    <class 'NoneType'>
    949744
    <class 'int'>
    None
    <class 'NoneType'>
    9497442
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    57934401
    <class 'int'>
    5698465
    <class 'int'>
    11396931
    <class 'int'>
    None
    <class 'NoneType'>
    15765755
    <class 'int'>
    4748721
    <class 'int'>
    12346675
    <class 'int'>
    23743607
    <class 'int'>
    None
    <class 'NoneType'>
    10922059
    <class 'int'>
    4748721
    <class 'int'>
    5318567
    <class 'int'>
    2279386
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    807282
    <class 'int'>
    9022570
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4748721
    <class 'int'>
    4748721
    <class 'int'>
    23743607
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    949744
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23743607
    <class 'int'>
    17285345
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11396931
    <class 'int'>
    None
    <class 'NoneType'>
    1899488
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11396931
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2849232
    <class 'int'>
    7597954
    <class 'int'>
    142461
    <class 'int'>
    1709539
    <class 'int'>
    None
    <class 'NoneType'>
    37989771
    <class 'int'>
    5698465
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4060156
    <class 'int'>
    None
    <class 'NoneType'>
    18994885
    <class 'int'>
    18994885
    <class 'int'>
    712308
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11396931
    <class 'int'>
    1899488
    <class 'int'>
    None
    <class 'NoneType'>
    16145652
    <class 'int'>
    None
    <class 'NoneType'>
    4748721
    <class 'int'>
    11871803
    <class 'int'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    949744
    <class 'int'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7597954
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2849232
    <class 'int'>
    2374360
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28492328
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2849232
    <class 'int'>
    19469757
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11396931
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1899488
    <class 'int'>
    None
    <class 'NoneType'>
    10447187
    <class 'int'>
    None
    <class 'NoneType'>
    5033644
    <class 'int'>
    3324104
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4748721
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1424616
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5983388
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    949744
    <class 'int'>
    4748721
    <class 'int'>
    1614565
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5698465
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    189948
    <class 'int'>
    None
    <class 'NoneType'>
    16
    <class 'int'>
    None
    <class 'NoneType'>
    474872
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    61733378
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16145652
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    284923
    <class 'int'>
    None
    <class 'NoneType'>
    1899488
    <class 'int'>
    161456
    <class 'int'>
    1187180
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5688968
    <class 'int'>
    1899488
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1899488
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9497442
    <class 'int'>
    4748721
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3324104
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2374360
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13296419
    <class 'int'>
    None
    <class 'NoneType'>
    4748721
    <class 'int'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1424616
    <class 'int'>
    474872
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2697273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3798977
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1044718
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7597
    <class 'int'>
    None
    <class 'NoneType'>
    474872
    <class 'int'>
    None
    <class 'NoneType'>
    18994885
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    284923
    <class 'int'>
    None
    <class 'NoneType'>
    1424616
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    474872
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9877
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    94974
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4748721
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7123082
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    379897
    <class 'int'>
    None
    <class 'NoneType'>
    61733378
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15195908
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8082323
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2374360
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11396931
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2849232
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1899488
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    104
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    111423148
    <class 'int'>
    26077758
    <class 'int'>
    165949370
    <class 'int'>
    35560579
    <class 'int'>
    177802896
    <class 'int'>
    177802896
    <class 'int'>
    111423148
    <class 'int'>
    94828211
    <class 'int'>
    47414105
    <class 'int'>
    37931284
    <class 'int'>
    237070528
    <class 'int'>
    17780289
    <class 'int'>
    82974685
    <class 'int'>
    92457506
    <class 'int'>
    65194395
    <class 'int'>
    92457506
    <class 'int'>
    71121158
    <class 'int'>
    23707052
    <class 'int'>
    4741410
    <class 'int'>
    27263110
    <class 'int'>
    47414105
    <class 'int'>
    118535264
    <class 'int'>
    118535264
    <class 'int'>
    56896926
    <class 'int'>
    21336347
    <class 'int'>
    42672695
    <class 'int'>
    41487342
    <class 'int'>
    165949370
    <class 'int'>
    154095843
    <class 'int'>
    112608501
    <class 'int'>
    14935443
    <class 'int'>
    35560579
    <class 'int'>
    177802896
    <class 'int'>
    77047921
    <class 'int'>
    29633816
    <class 'int'>
    5926763
    <class 'int'>
    88901448
    <class 'int'>
    71121158
    <class 'int'>
    162393312
    <class 'int'>
    34375226
    <class 'int'>
    None
    <class 'NoneType'>
    28448463
    <class 'int'>
    59267632
    <class 'int'>
    59267632
    <class 'int'>
    94828211
    <class 'int'>
    82974685
    <class 'int'>
    53340869
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30819168
    <class 'int'>
    71121158
    <class 'int'>
    94828211
    <class 'int'>
    71121158
    <class 'int'>
    142242317
    <class 'int'>
    94828211
    <class 'int'>
    106681738
    <class 'int'>
    47414105
    <class 'int'>
    47414105
    <class 'int'>
    23707052
    <class 'int'>
    30819168
    <class 'int'>
    None
    <class 'NoneType'>
    11260850
    <class 'int'>
    None
    <class 'NoneType'>
    47414105
    <class 'int'>
    93642858
    <class 'int'>
    7112115
    <class 'int'>
    23707052
    <class 'int'>
    48599458
    <class 'int'>
    2370705
    <class 'int'>
    35560579
    <class 'int'>
    29633816
    <class 'int'>
    None
    <class 'NoneType'>
    94828211
    <class 'int'>
    59267632
    <class 'int'>
    100754974
    <class 'int'>
    None
    <class 'NoneType'>
    90086801
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26077758
    <class 'int'>
    21336347
    <class 'int'>
    None
    <class 'NoneType'>
    32526076
    <class 'int'>
    71121158
    <class 'int'>
    None
    <class 'NoneType'>
    103125680
    <class 'int'>
    59267632
    <class 'int'>
    20150994
    <class 'int'>
    20150994
    <class 'int'>
    80603979
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    61638337
    <class 'int'>
    45043400
    <class 'int'>
    41487342
    <class 'int'>
    41487342
    <class 'int'>
    71121158
    <class 'int'>
    65194395
    <class 'int'>
    19558318
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1778028
    <class 'int'>
    71121158
    <class 'int'>
    8890144
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9482821
    <class 'int'>
    14224231
    <class 'int'>
    None
    <class 'NoneType'>
    23707052
    <class 'int'>
    35560579
    <class 'int'>
    64009042
    <class 'int'>
    66379748
    <class 'int'>
    None
    <class 'NoneType'>
    7112115
    <class 'int'>
    22521700
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23707052
    <class 'int'>
    5926763
    <class 'int'>
    41487342
    <class 'int'>
    59267632
    <class 'int'>
    None
    <class 'NoneType'>
    3556057
    <class 'int'>
    None
    <class 'NoneType'>
    8297468
    <class 'int'>
    9482821
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8297468
    <class 'int'>
    23707052
    <class 'int'>
    16594937
    <class 'int'>
    37931284
    <class 'int'>
    7586256
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    355605
    <class 'int'>
    None
    <class 'NoneType'>
    88901448
    <class 'int'>
    9245750
    <class 'int'>
    None
    <class 'NoneType'>
    8890144
    <class 'int'>
    7112115
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1778028
    <class 'int'>
    None
    <class 'NoneType'>
    1007549
    <class 'int'>
    592676
    <class 'int'>
    None
    <class 'NoneType'>
    17780289
    <class 'int'>
    None
    <class 'NoneType'>
    5926763
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    71121158
    <class 'int'>
    924575
    <class 'int'>
    94828211
    <class 'int'>
    26077758
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11853526
    <class 'int'>
    None
    <class 'NoneType'>
    9786590
    <class 'int'>
    None
    <class 'NoneType'>
    592676
    <class 'int'>
    5170387
    <class 'int'>
    None
    <class 'NoneType'>
    33189874
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28448463
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15409584
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8060397
    <class 'int'>
    None
    <class 'NoneType'>
    71121158
    <class 'int'>
    65194395
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16594937
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11
    <class 'int'>
    23707052
    <class 'int'>
    17780289
    <class 'int'>
    7112115
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11853526
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8297468
    <class 'int'>
    None
    <class 'NoneType'>
    37931284
    <class 'int'>
    20150994
    <class 'int'>
    None
    <class 'NoneType'>
    20150994
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16594937
    <class 'int'>
    None
    <class 'NoneType'>
    130388790
    <class 'int'>
    None
    <class 'NoneType'>
    5926763
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35560579
    <class 'int'>
    14224231
    <class 'int'>
    None
    <class 'NoneType'>
    674722
    <class 'int'>
    5926763
    <class 'int'>
    355
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8297468
    <class 'int'>
    19795389
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    271692064
    <class 'int'>
    115469127
    <class 'int'>
    122261428
    <class 'int'>
    48904571
    <class 'int'>
    122261428
    <class 'int'>
    169807540
    <class 'int'>
    4754611
    <class 'int'>
    149430635
    <class 'int'>
    13584603
    <class 'int'>
    16301523
    <class 'int'>
    95092222
    <class 'int'>
    67923016
    <class 'int'>
    81507619
    <class 'int'>
    71998397
    <class 'int'>
    43470730
    <class 'int'>
    22414595
    <class 'int'>
    142638333
    <class 'int'>
    4754611
    <class 'int'>
    101884524
    <class 'int'>
    108676825
    <class 'int'>
    47546111
    <class 'int'>
    23093825
    <class 'int'>
    77432238
    <class 'int'>
    61130714
    <class 'int'>
    339615
    <class 'int'>
    157581397
    <class 'int'>
    217353651
    <class 'int'>
    47546111
    <class 'int'>
    32603047
    <class 'int'>
    61130714
    <class 'int'>
    None
    <class 'NoneType'>
    122261428
    <class 'int'>
    84224539
    <class 'int'>
    54338412
    <class 'int'>
    122261428
    <class 'int'>
    81507619
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    108676825
    <class 'int'>
    92375301
    <class 'int'>
    101884524
    <class 'int'>
    115469127
    <class 'int'>
    74715317
    <class 'int'>
    20376904
    <class 'int'>
    67923016
    <class 'int'>
    48904571
    <class 'int'>
    95092222
    <class 'int'>
    51621492
    <class 'int'>
    20376904
    <class 'int'>
    20376904
    <class 'int'>
    67923016
    <class 'int'>
    9509222
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24452285
    <class 'int'>
    122261428
    <class 'int'>
    47546111
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36678428
    <class 'int'>
    67923016
    <class 'int'>
    33961508
    <class 'int'>
    25810746
    <class 'int'>
    None
    <class 'NoneType'>
    40753809
    <class 'int'>
    27169206
    <class 'int'>
    None
    <class 'NoneType'>
    54338412
    <class 'int'>
    67923016
    <class 'int'>
    339615
    <class 'int'>
    38036888
    <class 'int'>
    108676825
    <class 'int'>
    67923016
    <class 'int'>
    81507619
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16301523
    <class 'int'>
    1358460
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40753809
    <class 'int'>
    None
    <class 'NoneType'>
    115469127
    <class 'int'>
    81507619
    <class 'int'>
    40753809
    <class 'int'>
    51621492
    <class 'int'>
    6792301
    <class 'int'>
    None
    <class 'NoneType'>
    40753809
    <class 'int'>
    6792301
    <class 'int'>
    20376904
    <class 'int'>
    47546111
    <class 'int'>
    27169206
    <class 'int'>
    33961508
    <class 'int'>
    24452285
    <class 'int'>
    None
    <class 'NoneType'>
    4075380
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    33961508
    <class 'int'>
    43470730
    <class 'int'>
    None
    <class 'NoneType'>
    40753809
    <class 'int'>
    None
    <class 'NoneType'>
    74715317
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6792301
    <class 'int'>
    47546111
    <class 'int'>
    65206095
    <class 'int'>
    13584603
    <class 'int'>
    None
    <class 'NoneType'>
    39395349
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27169206
    <class 'int'>
    74715317
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21735365
    <class 'int'>
    23093825
    <class 'int'>
    9509222
    <class 'int'>
    None
    <class 'NoneType'>
    38036888
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16301523
    <class 'int'>
    None
    <class 'NoneType'>
    25810746
    <class 'int'>
    47546111
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21735365
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    33961508
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12226142
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    47546111
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    71998397
    <class 'int'>
    None
    <class 'NoneType'>
    29886127
    <class 'int'>
    48904571
    <class 'int'>
    20376904
    <class 'int'>
    8150761
    <class 'int'>
    40753809
    <class 'int'>
    99167603
    <class 'int'>
    50263031
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24452285
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21735365
    <class 'int'>
    1765998
    <class 'int'>
    5433841
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1068666
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    140405002
    <class 'int'>
    98283502
    <class 'int'>
    159125669
    <class 'int'>
    187206670
    <class 'int'>
    70202501
    <class 'int'>
    121684335
    <class 'int'>
    93603335
    <class 'int'>
    168486003
    <class 'int'>
    112324002
    <class 'int'>
    234008338
    <class 'int'>
    210607504
    <class 'int'>
    21528767
    <class 'int'>
    71138534
    <class 'int'>
    177846337
    <class 'int'>
    18720667
    <class 'int'>
    35569267
    <class 'int'>
    12168433
    <class 'int'>
    98283502
    <class 'int'>
    None
    <class 'NoneType'>
    107643835
    <class 'int'>
    86115068
    <class 'int'>
    2808100
    <class 'int'>
    187206670
    <class 'int'>
    187206670
    <class 'int'>
    84243001
    <class 'int'>
    102963668
    <class 'int'>
    34633234
    <class 'int'>
    57098034
    <class 'int'>
    None
    <class 'NoneType'>
    35569267
    <class 'int'>
    121684335
    <class 'int'>
    10296366
    <class 'int'>
    65522334
    <class 'int'>
    84243001
    <class 'int'>
    43057534
    <class 'int'>
    18720667
    <class 'int'>
    37441334
    <class 'int'>
    32761167
    <class 'int'>
    5616200
    <class 'int'>
    46801667
    <class 'int'>
    163805836
    <class 'int'>
    3276116
    <class 'int'>
    56162001
    <class 'int'>
    93603335
    <class 'int'>
    37441334
    <class 'int'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    121684335
    <class 'int'>
    78626801
    <class 'int'>
    46801667
    <class 'int'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    12168433
    <class 'int'>
    15912566
    <class 'int'>
    26208933
    <class 'int'>
    28081000
    <class 'int'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    14040500
    <class 'int'>
    74882668
    <class 'int'>
    96411435
    <class 'int'>
    126364502
    <class 'int'>
    20592733
    <class 'int'>
    46801667
    <class 'int'>
    28081000
    <class 'int'>
    51481834
    <class 'int'>
    23400833
    <class 'int'>
    140405002
    <class 'int'>
    7488266
    <class 'int'>
    32761167
    <class 'int'>
    11232400
    <class 'int'>
    None
    <class 'NoneType'>
    182526503
    <class 'int'>
    28081000
    <class 'int'>
    51481834
    <class 'int'>
    10296366
    <class 'int'>
    24336867
    <class 'int'>
    29953067
    <class 'int'>
    14040500
    <class 'int'>
    32761167
    <class 'int'>
    121684335
    <class 'int'>
    6552233
    <class 'int'>
    40249434
    <class 'int'>
    32761167
    <class 'int'>
    26208933
    <class 'int'>
    4680166
    <class 'int'>
    238688504
    <class 'int'>
    56162001
    <class 'int'>
    32761167
    <class 'int'>
    28081000
    <class 'int'>
    11232400
    <class 'int'>
    964174
    <class 'int'>
    8424300
    <class 'int'>
    18720667
    <class 'int'>
    2340083
    <class 'int'>
    18720667
    <class 'int'>
    46801667
    <class 'int'>
    7020250
    <class 'int'>
    187206670
    <class 'int'>
    126364502
    <class 'int'>
    18720667
    <class 'int'>
    23400833
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    41185467
    <class 'int'>
    28081000
    <class 'int'>
    16848600
    <class 'int'>
    14040500
    <class 'int'>
    14976533
    <class 'int'>
    29953067
    <class 'int'>
    12168433
    <class 'int'>
    None
    <class 'NoneType'>
    23400833
    <class 'int'>
    5616200
    <class 'int'>
    12168433
    <class 'int'>
    37441334
    <class 'int'>
    37441334
    <class 'int'>
    10202763
    <class 'int'>
    None
    <class 'NoneType'>
    11232400
    <class 'int'>
    26208933
    <class 'int'>
    117647
    <class 'int'>
    None
    <class 'NoneType'>
    8892316
    <class 'int'>
    14040500
    <class 'int'>
    10296366
    <class 'int'>
    21528767
    <class 'int'>
    73010601
    <class 'int'>
    14976533
    <class 'int'>
    16848600
    <class 'int'>
    26208933
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2340083
    <class 'int'>
    4492960
    <class 'int'>
    28081000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12636450
    <class 'int'>
    28081000
    <class 'int'>
    9360333
    <class 'int'>
    None
    <class 'NoneType'>
    6552233
    <class 'int'>
    14976533
    <class 'int'>
    16848600
    <class 'int'>
    52417867
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    74882668
    <class 'int'>
    None
    <class 'NoneType'>
    25272900
    <class 'int'>
    2714496
    <class 'int'>
    7956283
    <class 'int'>
    56162001
    <class 'int'>
    25478827
    <class 'int'>
    98283502
    <class 'int'>
    14976533
    <class 'int'>
    37441334
    <class 'int'>
    9360333
    <class 'int'>
    None
    <class 'NoneType'>
    4305753
    <class 'int'>
    26208933
    <class 'int'>
    23400833
    <class 'int'>
    23868850
    <class 'int'>
    112324002
    <class 'int'>
    32761167
    <class 'int'>
    51481834
    <class 'int'>
    11232400
    <class 'int'>
    None
    <class 'NoneType'>
    18720667
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14602120
    <class 'int'>
    None
    <class 'NoneType'>
    23400833
    <class 'int'>
    7956283
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9360333
    <class 'int'>
    23400833
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2808100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11232400
    <class 'int'>
    3276116
    <class 'int'>
    936033
    <class 'int'>
    None
    <class 'NoneType'>
    32761167
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7207456
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    3744133
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14040500
    <class 'int'>
    None
    <class 'NoneType'>
    20592733
    <class 'int'>
    1404050
    <class 'int'>
    31825133
    <class 'int'>
    None
    <class 'NoneType'>
    7675473
    <class 'int'>
    None
    <class 'NoneType'>
    149765336
    <class 'int'>
    None
    <class 'NoneType'>
    561620
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    93603
    <class 'int'>
    18720667
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    93603
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3276116
    <class 'int'>
    None
    <class 'NoneType'>
    18720667
    <class 'int'>
    9360333
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1872066
    <class 'int'>
    None
    <class 'NoneType'>
    16380583
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3276116
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5820348
    <class 'int'>
    936033
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    936033
    <class 'int'>
    None
    <class 'NoneType'>
    4867373
    <class 'int'>
    3744133
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20592733
    <class 'int'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    6552233
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3744133
    <class 'int'>
    None
    <class 'NoneType'>
    9360333
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7488266
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6084216
    <class 'int'>
    16848600
    <class 'int'>
    6552233
    <class 'int'>
    None
    <class 'NoneType'>
    6552233
    <class 'int'>
    None
    <class 'NoneType'>
    65522334
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7488266
    <class 'int'>
    None
    <class 'NoneType'>
    4586563
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6084216
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15912566
    <class 'int'>
    None
    <class 'NoneType'>
    1033380
    <class 'int'>
    None
    <class 'NoneType'>
    889231
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1872066
    <class 'int'>
    2340083
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30889100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    234008
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7020250
    <class 'int'>
    11232400
    <class 'int'>
    None
    <class 'NoneType'>
    77690768
    <class 'int'>
    None
    <class 'NoneType'>
    14
    <class 'int'>
    None
    <class 'NoneType'>
    3744133
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5616200
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    530730
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    9360333
    <class 'int'>
    9360333
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    468016
    <class 'int'>
    None
    <class 'NoneType'>
    5616200
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3276116
    <class 'int'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5616200
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32761167
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    936033
    <class 'int'>
    None
    <class 'NoneType'>
    11232400
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    678624
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14040500
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    3744133
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7020250
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    46801
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    936033
    <class 'int'>
    1684860
    <class 'int'>
    126364
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3744133
    <class 'int'>
    16848600
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    985411
    <class 'int'>
    7488266
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9360333
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    932775
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    83
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2808100
    <class 'int'>
    None
    <class 'NoneType'>
    2808100
    <class 'int'>
    None
    <class 'NoneType'>
    2808100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4680166
    <class 'int'>
    None
    <class 'NoneType'>
    6552233
    <class 'int'>
    936033
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    748826
    <class 'int'>
    4785027
    <class 'int'>
    748826
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4212150
    <class 'int'>
    7488266
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10296366
    <class 'int'>
    None
    <class 'NoneType'>
    2340083
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7226177
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3744133
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3369720
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    936033
    <class 'int'>
    936033
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    217159
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1123240
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2808100
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    54289934
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    702
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1310446
    <class 'int'>
    608
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4212150
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28081000
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    468
    <class 'int'>
    None
    <class 'NoneType'>
    325893
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38516154
    <class 'int'>
    2027166
    <class 'int'>
    40543321
    <class 'int'>
    24945338
    <class 'int'>
    60814981
    <class 'int'>
    50679151
    <class 'int'>
    38516154
    <class 'int'>
    6081498
    <class 'int'>
    89195306
    <class 'int'>
    62842147
    <class 'int'>
    62842147
    <class 'int'>
    30407490
    <class 'int'>
    50679151
    <class 'int'>
    30407490
    <class 'int'>
    20271660
    <class 'int'>
    25339575
    <class 'int'>
    8108664
    <class 'int'>
    24325992
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18244494
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36286272
    <class 'int'>
    7095081
    <class 'int'>
    30407490
    <class 'int'>
    4459765
    <class 'int'>
    101358302
    <class 'int'>
    None
    <class 'NoneType'>
    2027166
    <class 'int'>
    2027166
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30407490
    <class 'int'>
    None
    <class 'NoneType'>
    7703230
    <class 'int'>
    50679151
    <class 'int'>
    50679151
    <class 'int'>
    1824449
    <class 'int'>
    18244494
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9122247
    <class 'int'>
    12162996
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40543321
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    52706317
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6081498
    <class 'int'>
    None
    <class 'NoneType'>
    1915671
    <class 'int'>
    36488988
    <class 'int'>
    10135830
    <class 'int'>
    None
    <class 'NoneType'>
    13176579
    <class 'int'>
    None
    <class 'NoneType'>
    18244494
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50679151
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30407490
    <class 'int'>
    10135830
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6486931
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20271660
    <class 'int'>
    3446182
    <class 'int'>
    None
    <class 'NoneType'>
    50679151
    <class 'int'>
    None
    <class 'NoneType'>
    48651985
    <class 'int'>
    38516154
    <class 'int'>
    231
    <class 'int'>
    23312409
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1743362
    <class 'int'>
    None
    <class 'NoneType'>
    4885470
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    167484493
    <class 'int'>
    167484493
    <class 'int'>
    200981392
    <class 'int'>
    60294417
    <class 'int'>
    44662531
    <class 'int'>
    167484493
    <class 'int'>
    126171651
    <class 'int'>
    55828164
    <class 'int'>
    140686974
    <class 'int'>
    83742246
    <class 'int'>
    111656328
    <class 'int'>
    111656328
    <class 'int'>
    44662531
    <class 'int'>
    69226923
    <class 'int'>
    62527544
    <class 'int'>
    147386354
    <class 'int'>
    94907879
    <class 'int'>
    29030645
    <class 'int'>
    167484493
    <class 'int'>
    78159430
    <class 'int'>
    35730025
    <class 'int'>
    None
    <class 'NoneType'>
    27914082
    <class 'int'>
    145153227
    <class 'int'>
    48012221
    <class 'int'>
    231128600
    <class 'int'>
    59177854
    <class 'int'>
    55828164
    <class 'int'>
    31263772
    <class 'int'>
    15631886
    <class 'int'>
    83742246
    <class 'int'>
    None
    <class 'NoneType'>
    97141006
    <class 'int'>
    27914082
    <class 'int'>
    122821961
    <class 'int'>
    None
    <class 'NoneType'>
    83742246
    <class 'int'>
    46895658
    <class 'int'>
    72576613
    <class 'int'>
    48012221
    <class 'int'>
    None
    <class 'NoneType'>
    44662531
    <class 'int'>
    33496898
    <class 'int'>
    78159430
    <class 'int'>
    2456439
    <class 'int'>
    32380335
    <class 'int'>
    80392556
    <class 'int'>
    None
    <class 'NoneType'>
    43545968
    <class 'int'>
    4466253
    <class 'int'>
    64760670
    <class 'int'>
    55828164
    <class 'int'>
    63644107
    <class 'int'>
    None
    <class 'NoneType'>
    50245348
    <class 'int'>
    5359503
    <class 'int'>
    16748449
    <class 'int'>
    98257569
    <class 'int'>
    16748449
    <class 'int'>
    16748449
    <class 'int'>
    27914082
    <class 'int'>
    94907879
    <class 'int'>
    None
    <class 'NoneType'>
    11165632
    <class 'int'>
    66993797
    <class 'int'>
    None
    <class 'NoneType'>
    66993797
    <class 'int'>
    33496898
    <class 'int'>
    33496898
    <class 'int'>
    22331265
    <class 'int'>
    89325063
    <class 'int'>
    98257569
    <class 'int'>
    33496898
    <class 'int'>
    58061291
    <class 'int'>
    55828164
    <class 'int'>
    78159430
    <class 'int'>
    21214702
    <class 'int'>
    7257661
    <class 'int'>
    24564392
    <class 'int'>
    55828164
    <class 'int'>
    93791316
    <class 'int'>
    24564392
    <class 'int'>
    55828164
    <class 'int'>
    48012221
    <class 'int'>
    13398759
    <class 'int'>
    33496898
    <class 'int'>
    7815943
    <class 'int'>
    33496898
    <class 'int'>
    35730025
    <class 'int'>
    21214702
    <class 'int'>
    111656328
    <class 'int'>
    50245348
    <class 'int'>
    50245348
    <class 'int'>
    39079715
    <class 'int'>
    188699
    <class 'int'>
    39079715
    <class 'int'>
    2233126
    <class 'int'>
    61410980
    <class 'int'>
    33496898
    <class 'int'>
    None
    <class 'NoneType'>
    11165632
    <class 'int'>
    22331265
    <class 'int'>
    None
    <class 'NoneType'>
    1116563
    <class 'int'>
    16748449
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31263772
    <class 'int'>
    None
    <class 'NoneType'>
    33496898
    <class 'int'>
    44662531
    <class 'int'>
    50245348
    <class 'int'>
    4466253
    <class 'int'>
    39079715
    <class 'int'>
    55828164
    <class 'int'>
    27914082
    <class 'int'>
    91558189
    <class 'int'>
    3349689
    <class 'int'>
    35730025
    <class 'int'>
    89325063
    <class 'int'>
    None
    <class 'NoneType'>
    10049069
    <class 'int'>
    20098139
    <class 'int'>
    150736044
    <class 'int'>
    16748449
    <class 'int'>
    27914082
    <class 'int'>
    27914082
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    145153227
    <class 'int'>
    11165632
    <class 'int'>
    7815943
    <class 'int'>
    39079715
    <class 'int'>
    27914082
    <class 'int'>
    None
    <class 'NoneType'>
    1060735
    <class 'int'>
    44662531
    <class 'int'>
    89325063
    <class 'int'>
    55828164
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27914082
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55828164
    <class 'int'>
    None
    <class 'NoneType'>
    27914082
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3907971
    <class 'int'>
    16748449
    <class 'int'>
    14515322
    <class 'int'>
    None
    <class 'NoneType'>
    1116563
    <class 'int'>
    33496898
    <class 'int'>
    11165632
    <class 'int'>
    None
    <class 'NoneType'>
    530367
    <class 'int'>
    33496898
    <class 'int'>
    None
    <class 'NoneType'>
    33496898
    <class 'int'>
    15855198
    <class 'int'>
    None
    <class 'NoneType'>
    39079715
    <class 'int'>
    1004906
    <class 'int'>
    29030645
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7815943
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1674844
    <class 'int'>
    50245348
    <class 'int'>
    None
    <class 'NoneType'>
    10049069
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50245348
    <class 'int'>
    None
    <class 'NoneType'>
    913815
    <class 'int'>
    None
    <class 'NoneType'>
    8932506
    <class 'int'>
    1116563
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2233126
    <class 'int'>
    22331265
    <class 'int'>
    22331265
    <class 'int'>
    14515322
    <class 'int'>
    None
    <class 'NoneType'>
    13398759
    <class 'int'>
    10049069
    <class 'int'>
    12282196
    <class 'int'>
    4466253
    <class 'int'>
    None
    <class 'NoneType'>
    6699379
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5582816
    <class 'int'>
    33496898
    <class 'int'>
    44662531
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13398759
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1674844
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27914082
    <class 'int'>
    2121470
    <class 'int'>
    20098139
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3349689
    <class 'int'>
    16748449
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2233126
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8374224
    <class 'int'>
    1729556
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22331265
    <class 'int'>
    None
    <class 'NoneType'>
    5303675
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11483320
    <class 'int'>
    27914082
    <class 'int'>
    39079715
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    502453
    <class 'int'>
    8932506
    <class 'int'>
    None
    <class 'NoneType'>
    13398759
    <class 'int'>
    33496898
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16748449
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18309
    <class 'int'>
    11
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15631886
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27914082
    <class 'int'>
    None
    <class 'NoneType'>
    2233126
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39079715
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10719007
    <class 'int'>
    10049069
    <class 'int'>
    2233126
    <class 'int'>
    6699379
    <class 'int'>
    3907971
    <class 'int'>
    33496898
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80392
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55828
    <class 'int'>
    5582816
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3349689
    <class 'int'>
    1674844
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3349689
    <class 'int'>
    None
    <class 'NoneType'>
    15631886
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8932506
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7815943
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    44662531
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5582816
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17865012
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12282196
    <class 'int'>
    None
    <class 'NoneType'>
    16748449
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3349689
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13398759
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1786501
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    54083457
    <class 'int'>
    216333831
    <class 'int'>
    129800298
    <class 'int'>
    162250373
    <class 'int'>
    135208644
    <class 'int'>
    12980029
    <class 'int'>
    86533532
    <class 'int'>
    162250373
    <class 'int'>
    37858420
    <class 'int'>
    97350224
    <class 'int'>
    91941878
    <class 'int'>
    43266766
    <class 'int'>
    118983607
    <class 'int'>
    27041728
    <class 'int'>
    82206856
    <class 'int'>
    None
    <class 'NoneType'>
    292050672
    <class 'int'>
    19470044
    <class 'int'>
    89237705
    <class 'int'>
    None
    <class 'NoneType'>
    56246796
    <class 'int'>
    48675112
    <class 'int'>
    86533532
    <class 'int'>
    54083457
    <class 'int'>
    108166915
    <class 'int'>
    None
    <class 'NoneType'>
    12980029
    <class 'int'>
    29205067
    <class 'int'>
    43266766
    <class 'int'>
    27041728
    <class 'int'>
    54083457
    <class 'int'>
    21633383
    <class 'int'>
    37858420
    <class 'int'>
    108166915
    <class 'int'>
    25960059
    <class 'int'>
    None
    <class 'NoneType'>
    161168704
    <class 'int'>
    86533532
    <class 'int'>
    81125186
    <class 'int'>
    108166915
    <class 'int'>
    43266766
    <class 'int'>
    48675112
    <class 'int'>
    81125186
    <class 'int'>
    4543010
    <class 'int'>
    54083457
    <class 'int'>
    43266766
    <class 'int'>
    54083457
    <class 'int'>
    16225037
    <class 'int'>
    None
    <class 'NoneType'>
    30286736
    <class 'int'>
    91941878
    <class 'int'>
    32450074
    <class 'int'>
    16225037
    <class 'int'>
    146025336
    <class 'int'>
    19470044
    <class 'int'>
    33531743
    <class 'int'>
    97350224
    <class 'int'>
    2163338
    <class 'int'>
    22715052
    <class 'int'>
    54083457
    <class 'int'>
    43266766
    <class 'int'>
    18388375
    <class 'int'>
    None
    <class 'NoneType'>
    54083457
    <class 'int'>
    None
    <class 'NoneType'>
    70308495
    <class 'int'>
    43266766
    <class 'int'>
    None
    <class 'NoneType'>
    18929210
    <class 'int'>
    27041728
    <class 'int'>
    59491803
    <class 'int'>
    16765871
    <class 'int'>
    43266766
    <class 'int'>
    75716841
    <class 'int'>
    5408345
    <class 'int'>
    78421013
    <class 'int'>
    None
    <class 'NoneType'>
    59491803
    <class 'int'>
    43266766
    <class 'int'>
    40741068
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6490014
    <class 'int'>
    10816691
    <class 'int'>
    32450074
    <class 'int'>
    34613413
    <class 'int'>
    54083457
    <class 'int'>
    9735022
    <class 'int'>
    173067065
    <class 'int'>
    18388375
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18172041
    <class 'int'>
    58410134
    <class 'int'>
    8653353
    <class 'int'>
    69226826
    <class 'int'>
    25960059
    <class 'int'>
    None
    <class 'NoneType'>
    37858420
    <class 'int'>
    12980029
    <class 'int'>
    27041728
    <class 'int'>
    20551714
    <class 'int'>
    None
    <class 'NoneType'>
    9194187
    <class 'int'>
    None
    <class 'NoneType'>
    32450074
    <class 'int'>
    68145156
    <class 'int'>
    None
    <class 'NoneType'>
    37858420
    <class 'int'>
    1298002
    <class 'int'>
    20984381
    <class 'int'>
    12980029
    <class 'int'>
    16765871
    <class 'int'>
    21633383
    <class 'int'>
    24878390
    <class 'int'>
    None
    <class 'NoneType'>
    91941878
    <class 'int'>
    None
    <class 'NoneType'>
    55165127
    <class 'int'>
    4002175
    <class 'int'>
    None
    <class 'NoneType'>
    75716841
    <class 'int'>
    35695082
    <class 'int'>
    None
    <class 'NoneType'>
    7030849
    <class 'int'>
    11898360
    <class 'int'>
    75716841
    <class 'int'>
    None
    <class 'NoneType'>
    18388375
    <class 'int'>
    91941878
    <class 'int'>
    None
    <class 'NoneType'>
    32450074
    <class 'int'>
    41103428
    <class 'int'>
    None
    <class 'NoneType'>
    16225037
    <class 'int'>
    54083457
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16225037
    <class 'int'>
    27041728
    <class 'int'>
    21633383
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12980029
    <class 'int'>
    54083457
    <class 'int'>
    43266766
    <class 'int'>
    6490014
    <class 'int'>
    12980029
    <class 'int'>
    None
    <class 'NoneType'>
    28123398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    91941878
    <class 'int'>
    32450074
    <class 'int'>
    37858420
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12980029
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10816691
    <class 'int'>
    7030849
    <class 'int'>
    None
    <class 'NoneType'>
    59491803
    <class 'int'>
    4326676
    <class 'int'>
    5408345
    <class 'int'>
    48675112
    <class 'int'>
    21633383
    <class 'int'>
    16225037
    <class 'int'>
    16225037
    <class 'int'>
    1075179
    <class 'int'>
    22715052
    <class 'int'>
    None
    <class 'NoneType'>
    16225037
    <class 'int'>
    None
    <class 'NoneType'>
    64900149
    <class 'int'>
    None
    <class 'NoneType'>
    34613413
    <class 'int'>
    64900149
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16225037
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6490014
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27041728
    <class 'int'>
    21633383
    <class 'int'>
    17306706
    <class 'int'>
    None
    <class 'NoneType'>
    8653353
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8653353
    <class 'int'>
    11898360
    <class 'int'>
    48675112
    <class 'int'>
    19470044
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2163338
    <class 'int'>
    25960059
    <class 'int'>
    23796721
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21633383
    <class 'int'>
    None
    <class 'NoneType'>
    21633383
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5408345
    <class 'int'>
    162250
    <class 'int'>
    16225037
    <class 'int'>
    14061699
    <class 'int'>
    540834
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11898360
    <class 'int'>
    9735022
    <class 'int'>
    14061699
    <class 'int'>
    8653353
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8653353
    <class 'int'>
    None
    <class 'NoneType'>
    21633383
    <class 'int'>
    757168
    <class 'int'>
    4326676
    <class 'int'>
    None
    <class 'NoneType'>
    81125186
    <class 'int'>
    28123398
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    973502
    <class 'int'>
    16225037
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32450074
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    919418
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7571684
    <class 'int'>
    108166
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8653353
    <class 'int'>
    10129615
    <class 'int'>
    108166
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4326676
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12980029
    <class 'int'>
    None
    <class 'NoneType'>
    1081669
    <class 'int'>
    4326676
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5408345
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2574372
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3245007
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21633383
    <class 'int'>
    None
    <class 'NoneType'>
    28123398
    <class 'int'>
    None
    <class 'NoneType'>
    15684202
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16225037
    <class 'int'>
    None
    <class 'NoneType'>
    2487839
    <class 'int'>
    9735022
    <class 'int'>
    None
    <class 'NoneType'>
    16225037
    <class 'int'>
    None
    <class 'NoneType'>
    2163338
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    227150523
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1081669
    <class 'int'>
    5408345
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8653353
    <class 'int'>
    10816691
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3785842
    <class 'int'>
    540834
    <class 'int'>
    7571684
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5408345
    <class 'int'>
    8977854
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    324500
    <class 'int'>
    8653353
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12980029
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2704
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    85451
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1081669
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2920506
    <class 'int'>
    None
    <class 'NoneType'>
    150077901
    <class 'int'>
    34633361
    <class 'int'>
    57722269
    <class 'int'>
    23088907
    <class 'int'>
    33478916
    <class 'int'>
    106208976
    <class 'int'>
    86583404
    <class 'int'>
    202027944
    <class 'int'>
    115444539
    <class 'int'>
    138533447
    <class 'int'>
    86583404
    <class 'int'>
    19625571
    <class 'int'>
    184711263
    <class 'int'>
    76193396
    <class 'int'>
    190483490
    <class 'int'>
    173166809
    <class 'int'>
    15007790
    <class 'int'>
    51950042
    <class 'int'>
    126988993
    <class 'int'>
    92355631
    <class 'int'>
    144305674
    <class 'int'>
    133915666
    <class 'int'>
    75038950
    <class 'int'>
    4617781
    <class 'int'>
    75038950
    <class 'int'>
    8081117
    <class 'int'>
    69266723
    <class 'int'>
    161622355
    <class 'int'>
    34633361
    <class 'int'>
    57722269
    <class 'int'>
    None
    <class 'NoneType'>
    86583404
    <class 'int'>
    80811177
    <class 'int'>
    None
    <class 'NoneType'>
    34633361
    <class 'int'>
    None
    <class 'NoneType'>
    7503895
    <class 'int'>
    30015580
    <class 'int'>
    42714479
    <class 'int'>
    35787807
    <class 'int'>
    28861134
    <class 'int'>
    None
    <class 'NoneType'>
    28861134
    <class 'int'>
    21934462
    <class 'int'>
    80811177
    <class 'int'>
    23088907
    <class 'int'>
    13853344
    <class 'int'>
    178939036
    <class 'int'>
    10390008
    <class 'int'>
    1385334
    <class 'int'>
    230889079
    <class 'int'>
    28861134
    <class 'int'>
    80811177
    <class 'int'>
    48486706
    <class 'int'>
    31170025
    <class 'int'>
    38096698
    <class 'int'>
    115444539
    <class 'int'>
    69266723
    <class 'int'>
    1154445
    <class 'int'>
    32324471
    <class 'int'>
    None
    <class 'NoneType'>
    138533447
    <class 'int'>
    121216766
    <class 'int'>
    92355631
    <class 'int'>
    46177815
    <class 'int'>
    7503895
    <class 'int'>
    69266723
    <class 'int'>
    65803387
    <class 'int'>
    46177815
    <class 'int'>
    167394582
    <class 'int'>
    3117002
    <class 'int'>
    11544453
    <class 'int'>
    28861134
    <class 'int'>
    92355631
    <class 'int'>
    5772226
    <class 'int'>
    13853344
    <class 'int'>
    28861134
    <class 'int'>
    69266723
    <class 'int'>
    None
    <class 'NoneType'>
    57722269
    <class 'int'>
    40405588
    <class 'int'>
    103900085
    <class 'int'>
    8081
    <class 'int'>
    11544453
    <class 'int'>
    126988993
    <class 'int'>
    115444539
    <class 'int'>
    25397798
    <class 'int'>
    6926672
    <class 'int'>
    19625571
    <class 'int'>
    42714479
    <class 'int'>
    20202794
    <class 'int'>
    3232447
    <class 'int'>
    64648942
    <class 'int'>
    57722269
    <class 'int'>
    None
    <class 'NoneType'>
    150077
    <class 'int'>
    54258933
    <class 'int'>
    36942252
    <class 'int'>
    5772226
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28475330
    <class 'int'>
    None
    <class 'NoneType'>
    18471126
    <class 'int'>
    21934462
    <class 'int'>
    None
    <class 'NoneType'>
    69266723
    <class 'int'>
    None
    <class 'NoneType'>
    17316680
    <class 'int'>
    69266723
    <class 'int'>
    46177815
    <class 'int'>
    34633361
    <class 'int'>
    23088907
    <class 'int'>
    46177815
    <class 'int'>
    28861134
    <class 'int'>
    None
    <class 'NoneType'>
    8081117
    <class 'int'>
    23088907
    <class 'int'>
    35787807
    <class 'int'>
    51950042
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80811177
    <class 'int'>
    952417
    <class 'int'>
    2886113
    <class 'int'>
    461778
    <class 'int'>
    34633361
    <class 'int'>
    92355631
    <class 'int'>
    None
    <class 'NoneType'>
    34633361
    <class 'int'>
    26552244
    <class 'int'>
    57722269
    <class 'int'>
    126988993
    <class 'int'>
    20780017
    <class 'int'>
    61185606
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    48486706
    <class 'int'>
    51950042
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12698899
    <class 'int'>
    6234005
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8312006
    <class 'int'>
    51950042
    <class 'int'>
    26552244
    <class 'int'>
    28861134
    <class 'int'>
    None
    <class 'NoneType'>
    2308890
    <class 'int'>
    34633361
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    57722269
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9235563
    <class 'int'>
    23088907
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17316680
    <class 'int'>
    26552244
    <class 'int'>
    None
    <class 'NoneType'>
    26552244
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    86006182
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    46177815
    <class 'int'>
    32324471
    <class 'int'>
    None
    <class 'NoneType'>
    31170025
    <class 'int'>
    7850228
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18471126
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9812785
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26552244
    <class 'int'>
    31170025
    <class 'int'>
    None
    <class 'NoneType'>
    57722269
    <class 'int'>
    None
    <class 'NoneType'>
    6926672
    <class 'int'>
    None
    <class 'NoneType'>
    46177815
    <class 'int'>
    85486
    <class 'int'>
    34633361
    <class 'int'>
    230889
    <class 'int'>
    23088907
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22223073
    <class 'int'>
    30015580
    <class 'int'>
    13853344
    <class 'int'>
    None
    <class 'NoneType'>
    5772226
    <class 'int'>
    None
    <class 'NoneType'>
    75038
    <class 'int'>
    7503895
    <class 'int'>
    40405588
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45023370
    <class 'int'>
    None
    <class 'NoneType'>
    57722
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17316680
    <class 'int'>
    None
    <class 'NoneType'>
    23088907
    <class 'int'>
    16
    <class 'int'>
    None
    <class 'NoneType'>
    69266723
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1616223
    <class 'int'>
    20780017
    <class 'int'>
    3001558
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1731668
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    103
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3001558
    <class 'int'>
    None
    <class 'NoneType'>
    1154445
    <class 'int'>
    None
    <class 'NoneType'>
    577222
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3463336
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23088907
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34633361
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11544453
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3463336
    <class 'int'>
    None
    <class 'NoneType'>
    5772226
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6926672
    <class 'int'>
    950406
    <class 'int'>
    None
    <class 'NoneType'>
    4040558
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10159119
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    404055
    <class 'int'>
    31287365
    <class 'int'>
    10429121
    <class 'int'>
    9386209
    <class 'int'>
    31287365
    <class 'int'>
    18250963
    <class 'int'>
    None
    <class 'NoneType'>
    5214560
    <class 'int'>
    16686594
    <class 'int'>
    62574
    <class 'int'>
    10429121
    <class 'int'>
    1564368
    <class 'int'>
    469310
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26072804
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17480532
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3128736
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4432376
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2607280
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    391092
    <class 'int'>
    None
    <class 'NoneType'>
    47628661
    <class 'int'>
    50274698
    <class 'int'>
    9261128
    <class 'int'>
    71442992
    <class 'int'>
    11907165
    <class 'int'>
    47628661
    <class 'int'>
    142885984
    <class 'int'>
    1455320
    <class 'int'>
    None
    <class 'NoneType'>
    92611286
    <class 'int'>
    13230183
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2646036
    <class 'int'>
    None
    <class 'NoneType'>
    15876220
    <class 'int'>
    17199238
    <class 'int'>
    21168294
    <class 'int'>
    7938110
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    116425617
    <class 'int'>
    9261128
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13494787
    <class 'int'>
    52920735
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39690551
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12700976
    <class 'int'>
    None
    <class 'NoneType'>
    3969055
    <class 'int'>
    None
    <class 'NoneType'>
    19051464
    <class 'int'>
    None
    <class 'NoneType'>
    15876220
    <class 'int'>
    26460367
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2646036
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5292073
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    52920735
    <class 'int'>
    26460367
    <class 'int'>
    926112
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7938110
    <class 'int'>
    315500574
    <class 'int'>
    157750287
    <class 'int'>
    157750287
    <class 'int'>
    73616800
    <class 'int'>
    26291714
    <class 'int'>
    157750287
    <class 'int'>
    136716915
    <class 'int'>
    271330494
    <class 'int'>
    18930034
    <class 'int'>
    73616800
    <class 'int'>
    26291714
    <class 'int'>
    115683544
    <class 'int'>
    147233601
    <class 'int'>
    15775028
    <class 'int'>
    68358457
    <class 'int'>
    115683544
    <class 'int'>
    7887514
    <class 'int'>
    63100114
    <class 'int'>
    64151783
    <class 'int'>
    31550057
    <class 'int'>
    26291714
    <class 'int'>
    3996340
    <class 'int'>
    105166858
    <class 'int'>
    None
    <class 'NoneType'>
    168266973
    <class 'int'>
    52583429
    <class 'int'>
    21033371
    <class 'int'>
    63100114
    <class 'int'>
    157750287
    <class 'int'>
    157750287
    <class 'int'>
    31550057
    <class 'int'>
    31550057
    <class 'int'>
    31550057
    <class 'int'>
    None
    <class 'NoneType'>
    136716915
    <class 'int'>
    89391829
    <class 'int'>
    68358457
    <class 'int'>
    None
    <class 'NoneType'>
    89391829
    <class 'int'>
    35756731
    <class 'int'>
    78875143
    <class 'int'>
    78875143
    <class 'int'>
    15775028
    <class 'int'>
    73616800
    <class 'int'>
    42066743
    <class 'int'>
    52583429
    <class 'int'>
    26291714
    <class 'int'>
    21033371
    <class 'int'>
    31550057
    <class 'int'>
    189300344
    <class 'int'>
    168266
    <class 'int'>
    None
    <class 'NoneType'>
    31550057
    <class 'int'>
    73616800
    <class 'int'>
    21033371
    <class 'int'>
    36808400
    <class 'int'>
    57841772
    <class 'int'>
    26291714
    <class 'int'>
    90443498
    <class 'int'>
    26291714
    <class 'int'>
    15775
    <class 'int'>
    184042001
    <class 'int'>
    89391829
    <class 'int'>
    73616800
    <class 'int'>
    8413348
    <class 'int'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    105166858
    <class 'int'>
    None
    <class 'NoneType'>
    52583429
    <class 'int'>
    15775028
    <class 'int'>
    None
    <class 'NoneType'>
    41015074
    <class 'int'>
    54160932
    <class 'int'>
    73616800
    <class 'int'>
    29446720
    <class 'int'>
    None
    <class 'NoneType'>
    63100114
    <class 'int'>
    89391829
    <class 'int'>
    26291714
    <class 'int'>
    25240045
    <class 'int'>
    23136708
    <class 'int'>
    None
    <class 'NoneType'>
    70461795
    <class 'int'>
    10516685
    <class 'int'>
    16826697
    <class 'int'>
    157750287
    <class 'int'>
    47325086
    <class 'int'>
    10727019
    <class 'int'>
    None
    <class 'NoneType'>
    55738434
    <class 'int'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    4206674
    <class 'int'>
    12620022
    <class 'int'>
    15775028
    <class 'int'>
    17352531
    <class 'int'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    63100114
    <class 'int'>
    26291714
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    84133486
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18930034
    <class 'int'>
    None
    <class 'NoneType'>
    6310011
    <class 'int'>
    None
    <class 'NoneType'>
    2103337
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8413348
    <class 'int'>
    47325086
    <class 'int'>
    2103337
    <class 'int'>
    31550057
    <class 'int'>
    16826697
    <class 'int'>
    9465017
    <class 'int'>
    42066743
    <class 'int'>
    None
    <class 'NoneType'>
    63936191
    <class 'int'>
    5468676
    <class 'int'>
    21033371
    <class 'int'>
    63100114
    <class 'int'>
    42066743
    <class 'int'>
    28920886
    <class 'int'>
    21033371
    <class 'int'>
    8413348
    <class 'int'>
    None
    <class 'NoneType'>
    63100114
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24188377
    <class 'int'>
    None
    <class 'NoneType'>
    26291714
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    4206674
    <class 'int'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22085040
    <class 'int'>
    None
    <class 'NoneType'>
    31550057
    <class 'int'>
    None
    <class 'NoneType'>
    23136708
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3680840
    <class 'int'>
    22085040
    <class 'int'>
    21033371
    <class 'int'>
    19981703
    <class 'int'>
    26291714
    <class 'int'>
    None
    <class 'NoneType'>
    17352531
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17878365
    <class 'int'>
    242304
    <class 'int'>
    36808400
    <class 'int'>
    25240045
    <class 'int'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    26291714
    <class 'int'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    47325086
    <class 'int'>
    21033371
    <class 'int'>
    23136708
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    736
    <class 'int'>
    2103337
    <class 'int'>
    None
    <class 'NoneType'>
    16826697
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4732508
    <class 'int'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2734338
    <class 'int'>
    9465017
    <class 'int'>
    47325086
    <class 'int'>
    19981703
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6310011
    <class 'int'>
    24188377
    <class 'int'>
    1577502
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36808400
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5258342
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16826697
    <class 'int'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13671691
    <class 'int'>
    21033371
    <class 'int'>
    47325086
    <class 'int'>
    None
    <class 'NoneType'>
    27343383
    <class 'int'>
    None
    <class 'NoneType'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9465017
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26291714
    <class 'int'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    70461795
    <class 'int'>
    None
    <class 'NoneType'>
    3155005
    <class 'int'>
    None
    <class 'NoneType'>
    2103337
    <class 'int'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    4206674
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    4206674
    <class 'int'>
    16826697
    <class 'int'>
    2103337
    <class 'int'>
    13671691
    <class 'int'>
    39963406
    <class 'int'>
    None
    <class 'NoneType'>
    8413348
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6835845
    <class 'int'>
    3680840
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4837675
    <class 'int'>
    None
    <class 'NoneType'>
    52583
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    867626
    <class 'int'>
    12
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8413348
    <class 'int'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5258342
    <class 'int'>
    None
    <class 'NoneType'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    525834
    <class 'int'>
    4206674
    <class 'int'>
    11568354
    <class 'int'>
    None
    <class 'NoneType'>
    26291714
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4206674
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    52583429
    <class 'int'>
    1051668
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12620022
    <class 'int'>
    None
    <class 'NoneType'>
    7361680
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31550057
    <class 'int'>
    None
    <class 'NoneType'>
    11568354
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16826697
    <class 'int'>
    None
    <class 'NoneType'>
    8413348
    <class 'int'>
    None
    <class 'NoneType'>
    3155005
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15775028
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    126200
    <class 'int'>
    736168
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1577502
    <class 'int'>
    11568354
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26291
    <class 'int'>
    None
    <class 'NoneType'>
    13671691
    <class 'int'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2629171
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21033371
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    464723
    <class 'int'>
    None
    <class 'NoneType'>
    10516685
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6310
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2103337
    <class 'int'>
    None
    <class 'NoneType'>
    1156835
    <class 'int'>
    210333
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    33050094
    <class 'int'>
    94643451
    <class 'int'>
    1201821
    <class 'int'>
    105159390
    <class 'int'>
    102154836
    <class 'int'>
    12018216
    <class 'int'>
    718088
    <class 'int'>
    None
    <class 'NoneType'>
    21031878
    <class 'int'>
    52579695
    <class 'int'>
    None
    <class 'NoneType'>
    901366
    <class 'int'>
    33050094
    <class 'int'>
    None
    <class 'NoneType'>
    24036432
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60091080
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12018216
    <class 'int'>
    42063756
    <class 'int'>
    24036432
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    105159390
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6910474
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18027324
    <class 'int'>
    None
    <class 'NoneType'>
    8112295
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1261912
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4071170
    <class 'int'>
    2403643
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    901366
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13433869
    <class 'int'>
    58773177
    <class 'int'>
    62971261
    <class 'int'>
    23089462
    <class 'int'>
    56674135
    <class 'int'>
    31485630
    <class 'int'>
    62971261
    <class 'int'>
    37782756
    <class 'int'>
    16792336
    <class 'int'>
    3778275
    <class 'int'>
    37782756
    <class 'int'>
    58773177
    <class 'int'>
    83961681
    <class 'int'>
    20990420
    <class 'int'>
    9445689
    <class 'int'>
    None
    <class 'NoneType'>
    37782756
    <class 'int'>
    None
    <class 'NoneType'>
    17212144
    <class 'int'>
    None
    <class 'NoneType'>
    3148563
    <class 'int'>
    None
    <class 'NoneType'>
    73466471
    <class 'int'>
    None
    <class 'NoneType'>
    46178924
    <class 'int'>
    12594252
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13643773
    <class 'int'>
    30226205
    <class 'int'>
    6297126
    <class 'int'>
    53105763
    <class 'int'>
    None
    <class 'NoneType'>
    31485630
    <class 'int'>
    None
    <class 'NoneType'>
    52476050
    <class 'int'>
    3148563
    <class 'int'>
    16792336
    <class 'int'>
    3666950
    <class 'int'>
    20990420
    <class 'int'>
    None
    <class 'NoneType'>
    52476050
    <class 'int'>
    None
    <class 'NoneType'>
    18051761
    <class 'int'>
    121744438
    <class 'int'>
    None
    <class 'NoneType'>
    16792336
    <class 'int'>
    18891378
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15113102
    <class 'int'>
    None
    <class 'NoneType'>
    5352557
    <class 'int'>
    None
    <class 'NoneType'>
    58773177
    <class 'int'>
    None
    <class 'NoneType'>
    1679233
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1469329
    <class 'int'>
    16792336
    <class 'int'>
    30436109
    <class 'int'>
    2518850
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    58773177
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2235479
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14693294
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2623802
    <class 'int'>
    2099042
    <class 'int'>
    None
    <class 'NoneType'>
    37782756
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18891378
    <class 'int'>
    11544731
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16792336
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4198084
    <class 'int'>
    70824243
    <class 'int'>
    54732799
    <class 'int'>
    78815231
    <class 'int'>
    60206079
    <class 'int'>
    None
    <class 'NoneType'>
    85383167
    <class 'int'>
    88886067
    <class 'int'>
    26271743
    <class 'int'>
    32839679
    <class 'int'>
    44880895
    <class 'int'>
    None
    <class 'NoneType'>
    17514495
    <class 'int'>
    None
    <class 'NoneType'>
    48164863
    <class 'int'>
    13573734
    <class 'int'>
    59111423
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6567935
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21893119
    <class 'int'>
    14230527
    <class 'int'>
    None
    <class 'NoneType'>
    22112051
    <class 'int'>
    None
    <class 'NoneType'>
    10946559
    <class 'int'>
    26271743
    <class 'int'>
    48164863
    <class 'int'>
    21893119
    <class 'int'>
    766259
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8757247
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21893119
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    59
    <class 'int'>
    21893119
    <class 'int'>
    48164863
    <class 'int'>
    10946559
    <class 'int'>
    None
    <class 'NoneType'>
    10946559
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13030785
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2408243
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2627174
    <class 'int'>
    17514495
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32839679
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13135871
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2146368
    <class 'int'>
    41596927
    <class 'int'>
    766259
    <class 'int'>
    None
    <class 'NoneType'>
    930457
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32839679
    <class 'int'>
    47221906
    <class 'int'>
    42929006
    <class 'int'>
    8585801
    <class 'int'>
    103029614
    <class 'int'>
    82996078
    <class 'int'>
    93012846
    <class 'int'>
    128787018
    <class 'int'>
    50083840
    <class 'int'>
    42213522
    <class 'int'>
    74410277
    <class 'int'>
    27188370
    <class 'int'>
    78703177
    <class 'int'>
    85858012
    <class 'int'>
    3577417
    <class 'int'>
    128787018
    <class 'int'>
    143096686
    <class 'int'>
    74410277
    <class 'int'>
    17171602
    <class 'int'>
    42929006
    <class 'int'>
    71548343
    <class 'int'>
    None
    <class 'NoneType'>
    5723867
    <class 'int'>
    250419201
    <class 'int'>
    None
    <class 'NoneType'>
    42929006
    <class 'int'>
    34343204
    <class 'int'>
    5151480
    <class 'int'>
    31481271
    <class 'int'>
    None
    <class 'NoneType'>
    37205138
    <class 'int'>
    10016768
    <class 'int'>
    24326436
    <class 'int'>
    None
    <class 'NoneType'>
    28619337
    <class 'int'>
    85858012
    <class 'int'>
    140234752
    <class 'int'>
    25757403
    <class 'int'>
    23610953
    <class 'int'>
    14309668
    <class 'int'>
    21464503
    <class 'int'>
    43286747
    <class 'int'>
    71548343
    <class 'int'>
    45790939
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35774171
    <class 'int'>
    5008384
    <class 'int'>
    15740635
    <class 'int'>
    75841243
    <class 'int'>
    64393509
    <class 'int'>
    21464503
    <class 'int'>
    12878701
    <class 'int'>
    28619337
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42929006
    <class 'int'>
    11
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    85858012
    <class 'int'>
    None
    <class 'NoneType'>
    40067072
    <class 'int'>
    None
    <class 'NoneType'>
    28619337
    <class 'int'>
    None
    <class 'NoneType'>
    62962542
    <class 'int'>
    None
    <class 'NoneType'>
    78703177
    <class 'int'>
    30050304
    <class 'int'>
    7154834
    <class 'int'>
    71548343
    <class 'int'>
    None
    <class 'NoneType'>
    14309668
    <class 'int'>
    8585801
    <class 'int'>
    None
    <class 'NoneType'>
    88719945
    <class 'int'>
    24326436
    <class 'int'>
    15740635
    <class 'int'>
    82996078
    <class 'int'>
    60100608
    <class 'int'>
    715483
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11690087
    <class 'int'>
    None
    <class 'NoneType'>
    10016768
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28619337
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2146450
    <class 'int'>
    24326436
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    71548343
    <class 'int'>
    None
    <class 'NoneType'>
    8728897
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20033536
    <class 'int'>
    50083840
    <class 'int'>
    None
    <class 'NoneType'>
    11447734
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    57238674
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    71548343
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10732251
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    71548343
    <class 'int'>
    38636105
    <class 'int'>
    715483
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17171602
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27188370
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    64393509
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22895469
    <class 'int'>
    None
    <class 'NoneType'>
    50083840
    <class 'int'>
    None
    <class 'NoneType'>
    14309668
    <class 'int'>
    40067072
    <class 'int'>
    9158187
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35774171
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11447734
    <class 'int'>
    None
    <class 'NoneType'>
    8
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19124872
    <class 'int'>
    None
    <class 'NoneType'>
    8585801
    <class 'int'>
    1864829
    <class 'int'>
    43512679
    <class 'int'>
    27972436
    <class 'int'>
    124321940
    <class 'int'>
    77701212
    <class 'int'>
    54390848
    <class 'int'>
    62160970
    <class 'int'>
    17094266
    <class 'int'>
    54390848
    <class 'int'>
    None
    <class 'NoneType'>
    62160970
    <class 'int'>
    31080485
    <class 'int'>
    None
    <class 'NoneType'>
    76147188
    <class 'int'>
    35742557
    <class 'int'>
    19425303
    <class 'int'>
    62160970
    <class 'int'>
    None
    <class 'NoneType'>
    1554024
    <class 'int'>
    21756339
    <class 'int'>
    48174751
    <class 'int'>
    None
    <class 'NoneType'>
    73039139
    <class 'int'>
    12432194
    <class 'int'>
    48174751
    <class 'int'>
    23310363
    <class 'int'>
    4662072
    <class 'int'>
    12432194
    <class 'int'>
    None
    <class 'NoneType'>
    38850606
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    69931091
    <class 'int'>
    15540242
    <class 'int'>
    15540242
    <class 'int'>
    12432194
    <class 'int'>
    7770121
    <class 'int'>
    52836824
    <class 'int'>
    6216097
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    62160970
    <class 'int'>
    341885
    <class 'int'>
    18182083
    <class 'int'>
    7770121
    <class 'int'>
    None
    <class 'NoneType'>
    6216097
    <class 'int'>
    65269018
    <class 'int'>
    41958654
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    62160970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17094266
    <class 'int'>
    None
    <class 'NoneType'>
    5758495
    <class 'int'>
    None
    <class 'NoneType'>
    62160970
    <class 'int'>
    15540242
    <class 'int'>
    9324145
    <class 'int'>
    None
    <class 'NoneType'>
    62160970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35742557
    <class 'int'>
    37296582
    <class 'int'>
    62160970
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15540242
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    93241455
    <class 'int'>
    54390848
    <class 'int'>
    7770121
    <class 'int'>
    None
    <class 'NoneType'>
    15540242
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24864388
    <class 'int'>
    12432194
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31080485
    <class 'int'>
    None
    <class 'NoneType'>
    38850606
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23310363
    <class 'int'>
    26418412
    <class 'int'>
    None
    <class 'NoneType'>
    18648291
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26418412
    <class 'int'>
    54390848
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6899606
    <class 'int'>
    None
    <class 'NoneType'>
    4662072
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43167434
    <class 'int'>
    6714934
    <class 'int'>
    67149342
    <class 'int'>
    28778289
    <class 'int'>
    4796381
    <class 'int'>
    26380098
    <class 'int'>
    23981908
    <class 'int'>
    22303174
    <class 'int'>
    839366
    <class 'int'>
    14389144
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13190049
    <class 'int'>
    2997738
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5995477
    <class 'int'>
    28778289
    <class 'int'>
    16787335
    <class 'int'>
    4196833
    <class 'int'>
    None
    <class 'NoneType'>
    9592763
    <class 'int'>
    76742105
    <class 'int'>
    43167434
    <class 'int'>
    28778289
    <class 'int'>
    11990954
    <class 'int'>
    9832582
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    349622
    <class 'int'>
    23981908
    <class 'int'>
    38371052
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7194572
    <class 'int'>
    43167434
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43167434
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2398190
    <class 'int'>
    11990954
    <class 'int'>
    None
    <class 'NoneType'>
    33574671
    <class 'int'>
    148687
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35972862
    <class 'int'>
    83936678
    <class 'int'>
    None
    <class 'NoneType'>
    7194572
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1678733
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3597286
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5276019
    <class 'int'>
    None
    <class 'NoneType'>
    40769243
    <class 'int'>
    20384621
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    239819
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2398190
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    104266255
    <class 'int'>
    111217339
    <class 'int'>
    5560866
    <class 'int'>
    9731517
    <class 'int'>
    104266255
    <class 'int'>
    97315171
    <class 'int'>
    20158142
    <class 'int'>
    139021673
    <class 'int'>
    None
    <class 'NoneType'>
    111217339
    <class 'int'>
    139021673
    <class 'int'>
    75071703
    <class 'int'>
    44486935
    <class 'int'>
    26414118
    <class 'int'>
    55608669
    <class 'int'>
    63949969
    <class 'int'>
    16682600
    <class 'int'>
    83413004
    <class 'int'>
    19463034
    <class 'int'>
    90364088
    <class 'int'>
    69510836
    <class 'int'>
    50047802
    <class 'int'>
    79242354
    <class 'int'>
    75071703
    <class 'int'>
    111217339
    <class 'int'>
    41706502
    <class 'int'>
    69510836
    <class 'int'>
    37535851
    <class 'int'>
    76461920
    <class 'int'>
    127899939
    <class 'int'>
    111217339
    <class 'int'>
    None
    <class 'NoneType'>
    65340186
    <class 'int'>
    61169536
    <class 'int'>
    41706502
    <class 'int'>
    69510836
    <class 'int'>
    13902167
    <class 'int'>
    69510836
    <class 'int'>
    50047802
    <class 'int'>
    34755418
    <class 'int'>
    6255975
    <class 'int'>
    278043
    <class 'int'>
    69510836
    <class 'int'>
    76461920
    <class 'int'>
    62559753
    <class 'int'>
    41706502
    <class 'int'>
    None
    <class 'NoneType'>
    63949969
    <class 'int'>
    27804334
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12511950
    <class 'int'>
    None
    <class 'NoneType'>
    20853251
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5560866
    <class 'int'>
    None
    <class 'NoneType'>
    37535851
    <class 'int'>
    55608669
    <class 'int'>
    None
    <class 'NoneType'>
    62559753
    <class 'int'>
    34755418
    <class 'int'>
    None
    <class 'NoneType'>
    20853251
    <class 'int'>
    76461920
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9731517
    <class 'int'>
    None
    <class 'NoneType'>
    8341300
    <class 'int'>
    62559753
    <class 'int'>
    4117892
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    93144521
    <class 'int'>
    34755418
    <class 'int'>
    34755418
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4865758
    <class 'int'>
    None
    <class 'NoneType'>
    27804334
    <class 'int'>
    16682600
    <class 'int'>
    18072817
    <class 'int'>
    34755418
    <class 'int'>
    None
    <class 'NoneType'>
    9731517
    <class 'int'>
    52828236
    <class 'int'>
    None
    <class 'NoneType'>
    8341300
    <class 'int'>
    None
    <class 'NoneType'>
    62559753
    <class 'int'>
    None
    <class 'NoneType'>
    36145635
    <class 'int'>
    None
    <class 'NoneType'>
    4170650
    <class 'int'>
    1390216
    <class 'int'>
    6255975
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36145635
    <class 'int'>
    65340186
    <class 'int'>
    None
    <class 'NoneType'>
    69510836
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25023901
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    41706502
    <class 'int'>
    9731517
    <class 'int'>
    None
    <class 'NoneType'>
    61169536
    <class 'int'>
    6951083
    <class 'int'>
    35492233
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34755418
    <class 'int'>
    None
    <class 'NoneType'>
    5560866
    <class 'int'>
    27804334
    <class 'int'>
    69510836
    <class 'int'>
    52828236
    <class 'int'>
    None
    <class 'NoneType'>
    4170650
    <class 'int'>
    25023901
    <class 'int'>
    None
    <class 'NoneType'>
    48657585
    <class 'int'>
    62559753
    <class 'int'>
    None
    <class 'NoneType'>
    20853251
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    52828236
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38926068
    <class 'int'>
    23633684
    <class 'int'>
    None
    <class 'NoneType'>
    58389103
    <class 'int'>
    83413004
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38926068
    <class 'int'>
    4170650
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6951083
    <class 'int'>
    None
    <class 'NoneType'>
    34755418
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4170650
    <class 'int'>
    55608669
    <class 'int'>
    24328792
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31974984
    <class 'int'>
    20853251
    <class 'int'>
    None
    <class 'NoneType'>
    18072817
    <class 'int'>
    11121733
    <class 'int'>
    3475541
    <class 'int'>
    None
    <class 'NoneType'>
    4170650
    <class 'int'>
    None
    <class 'NoneType'>
    486575
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4170650
    <class 'int'>
    16682600
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18072817
    <class 'int'>
    9731517
    <class 'int'>
    34755418
    <class 'int'>
    44486935
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    130428087
    <class 'int'>
    11396629
    <class 'int'>
    155753930
    <class 'int'>
    113966290
    <class 'int'>
    12662921
    <class 'int'>
    158286514
    <class 'int'>
    60782021
    <class 'int'>
    24059550
    <class 'int'>
    126629211
    <class 'int'>
    88640448
    <class 'int'>
    161452244
    <class 'int'>
    116498874
    <class 'int'>
    56983145
    <class 'int'>
    139292132
    <class 'int'>
    29124718
    <class 'int'>
    56983145
    <class 'int'>
    31657302
    <class 'int'>
    78510111
    <class 'int'>
    6331460
    <class 'int'>
    113966290
    <class 'int'>
    65847189
    <class 'int'>
    69646066
    <class 'int'>
    32923594
    <class 'int'>
    29124718
    <class 'int'>
    32923594
    <class 'int'>
    120297750
    <class 'int'>
    56983145
    <class 'int'>
    39255055
    <class 'int'>
    5698314
    <class 'int'>
    50651684
    <class 'int'>
    None
    <class 'NoneType'>
    120297750
    <class 'int'>
    37988763
    <class 'int'>
    8864044
    <class 'int'>
    16208539
    <class 'int'>
    20260673
    <class 'int'>
    37988763
    <class 'int'>
    12662921
    <class 'int'>
    101303369
    <class 'int'>
    60782021
    <class 'int'>
    16461797
    <class 'int'>
    75977526
    <class 'int'>
    None
    <class 'NoneType'>
    103835953
    <class 'int'>
    18994381
    <class 'int'>
    None
    <class 'NoneType'>
    64580897
    <class 'int'>
    69646066
    <class 'int'>
    94971908
    <class 'int'>
    82308987
    <class 'int'>
    52297864
    <class 'int'>
    75977526
    <class 'int'>
    35456179
    <class 'int'>
    31657302
    <class 'int'>
    37988763
    <class 'int'>
    101303369
    <class 'int'>
    113966290
    <class 'int'>
    41787639
    <class 'int'>
    58249437
    <class 'int'>
    None
    <class 'NoneType'>
    80536178
    <class 'int'>
    None
    <class 'NoneType'>
    41787639
    <class 'int'>
    105102245
    <class 'int'>
    32923594
    <class 'int'>
    101303369
    <class 'int'>
    50651684
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    151955053
    <class 'int'>
    None
    <class 'NoneType'>
    107634829
    <class 'int'>
    None
    <class 'NoneType'>
    94971908
    <class 'int'>
    54450560
    <class 'int'>
    50651684
    <class 'int'>
    None
    <class 'NoneType'>
    94971908
    <class 'int'>
    53184268
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37988763
    <class 'int'>
    13929213
    <class 'int'>
    35456179
    <class 'int'>
    20260673
    <class 'int'>
    None
    <class 'NoneType'>
    44320224
    <class 'int'>
    17728089
    <class 'int'>
    40521347
    <class 'int'>
    126629211
    <class 'int'>
    75977526
    <class 'int'>
    54450560
    <class 'int'>
    12662921
    <class 'int'>
    25325842
    <class 'int'>
    6964606
    <class 'int'>
    20260673
    <class 'int'>
    6331460
    <class 'int'>
    107634829
    <class 'int'>
    106368537
    <class 'int'>
    None
    <class 'NoneType'>
    37988763
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    113966290
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25325842
    <class 'int'>
    44320224
    <class 'int'>
    11396629
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30391010
    <class 'int'>
    82308987
    <class 'int'>
    50651684
    <class 'int'>
    18994381
    <class 'int'>
    44320224
    <class 'int'>
    82308987
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    31657302
    <class 'int'>
    25325842
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10130336
    <class 'int'>
    6331460
    <class 'int'>
    50651684
    <class 'int'>
    17728089
    <class 'int'>
    17094943
    <class 'int'>
    10130336
    <class 'int'>
    31657302
    <class 'int'>
    None
    <class 'NoneType'>
    44320224
    <class 'int'>
    12662921
    <class 'int'>
    30391010
    <class 'int'>
    None
    <class 'NoneType'>
    2532584
    <class 'int'>
    31657302
    <class 'int'>
    30391010
    <class 'int'>
    None
    <class 'NoneType'>
    22793258
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30391010
    <class 'int'>
    30391010
    <class 'int'>
    None
    <class 'NoneType'>
    11396629
    <class 'int'>
    12662921
    <class 'int'>
    1266292
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75977526
    <class 'int'>
    None
    <class 'NoneType'>
    20260673
    <class 'int'>
    18994381
    <class 'int'>
    12662921
    <class 'int'>
    2532584
    <class 'int'>
    None
    <class 'NoneType'>
    6331460
    <class 'int'>
    1899438
    <class 'int'>
    43053931
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    44320224
    <class 'int'>
    94971908
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12662921
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15195505
    <class 'int'>
    82308987
    <class 'int'>
    18994381
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5065168
    <class 'int'>
    82308987
    <class 'int'>
    None
    <class 'NoneType'>
    8864044
    <class 'int'>
    6331460
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27858426
    <class 'int'>
    None
    <class 'NoneType'>
    10130336
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    96238200
    <class 'int'>
    None
    <class 'NoneType'>
    8864044
    <class 'int'>
    None
    <class 'NoneType'>
    72178650
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1899438
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    50651684
    <class 'int'>
    None
    <class 'NoneType'>
    18994381
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    63269987
    <class 'int'>
    23726245
    <class 'int'>
    22596424
    <class 'int'>
    33894636
    <class 'int'>
    None
    <class 'NoneType'>
    38413920
    <class 'int'>
    38413920
    <class 'int'>
    24178173
    <class 'int'>
    45192848
    <class 'int'>
    2259642
    <class 'int'>
    27115708
    <class 'int'>
    10168390
    <class 'int'>
    None
    <class 'NoneType'>
    33894636
    <class 'int'>
    None
    <class 'NoneType'>
    49712132
    <class 'int'>
    47452490
    <class 'int'>
    None
    <class 'NoneType'>
    45192848
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6778927
    <class 'int'>
    15817496
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27115708
    <class 'int'>
    None
    <class 'NoneType'>
    29827279
    <class 'int'>
    18077139
    <class 'int'>
    112982120
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    135578
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13557854
    <class 'int'>
    None
    <class 'NoneType'>
    9038569
    <class 'int'>
    9716462
    <class 'int'>
    5649106
    <class 'int'>
    36154278
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40673563
    <class 'int'>
    49712132
    <class 'int'>
    10168390
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7908748
    <class 'int'>
    18077139
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    72308556
    <class 'int'>
    None
    <class 'NoneType'>
    79087
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80261728
    <class 'int'>
    187277365
    <class 'int'>
    120392592
    <class 'int'>
    60196296
    <class 'int'>
    20065432
    <class 'int'>
    93638682
    <class 'int'>
    None
    <class 'NoneType'>
    44143950
    <class 'int'>
    160523456
    <class 'int'>
    93638682
    <class 'int'>
    80261728
    <class 'int'>
    46819341
    <class 'int'>
    73573250
    <class 'int'>
    36117777
    <class 'int'>
    None
    <class 'NoneType'>
    26753909
    <class 'int'>
    30766995
    <class 'int'>
    88287900
    <class 'int'>
    26753909
    <class 'int'>
    113704114
    <class 'int'>
    120392592
    <class 'int'>
    None
    <class 'NoneType'>
    100327160
    <class 'int'>
    107015637
    <class 'int'>
    66884773
    <class 'int'>
    120392592
    <class 'int'>
    187277365
    <class 'int'>
    86950205
    <class 'int'>
    173900410
    <class 'int'>
    100327160
    <class 'int'>
    100327160
    <class 'int'>
    93638682
    <class 'int'>
    24747366
    <class 'int'>
    33442386
    <class 'int'>
    127081069
    <class 'int'>
    30766995
    <class 'int'>
    80261728
    <class 'int'>
    80261
    <class 'int'>
    73573250
    <class 'int'>
    None
    <class 'NoneType'>
    69560164
    <class 'int'>
    36117777
    <class 'int'>
    53507818
    <class 'int'>
    94976378
    <class 'int'>
    93638682
    <class 'int'>
    86950205
    <class 'int'>
    80261728
    <class 'int'>
    40130864
    <class 'int'>
    64209382
    <class 'int'>
    24078518
    <class 'int'>
    None
    <class 'NoneType'>
    34780082
    <class 'int'>
    66884773
    <class 'int'>
    None
    <class 'NoneType'>
    18727736
    <class 'int'>
    None
    <class 'NoneType'>
    16052345
    <class 'int'>
    None
    <class 'NoneType'>
    20065432
    <class 'int'>
    120392592
    <class 'int'>
    None
    <class 'NoneType'>
    1805888
    <class 'int'>
    None
    <class 'NoneType'>
    40130864
    <class 'int'>
    None
    <class 'NoneType'>
    22740822
    <class 'int'>
    80261728
    <class 'int'>
    120392592
    <class 'int'>
    None
    <class 'NoneType'>
    33442386
    <class 'int'>
    26753909
    <class 'int'>
    17390041
    <class 'int'>
    33442386
    <class 'int'>
    None
    <class 'NoneType'>
    93638682
    <class 'int'>
    97651769
    <class 'int'>
    None
    <class 'NoneType'>
    86950205
    <class 'int'>
    8026172
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10701563
    <class 'int'>
    80261728
    <class 'int'>
    None
    <class 'NoneType'>
    80261728
    <class 'int'>
    26753909
    <class 'int'>
    17390041
    <class 'int'>
    None
    <class 'NoneType'>
    37455473
    <class 'int'>
    33442386
    <class 'int'>
    93638682
    <class 'int'>
    93638682
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4013086
    <class 'int'>
    40130864
    <class 'int'>
    66884773
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26753909
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20065432
    <class 'int'>
    26753909
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80261728
    <class 'int'>
    None
    <class 'NoneType'>
    869502
    <class 'int'>
    None
    <class 'NoneType'>
    13376954
    <class 'int'>
    22740822
    <class 'int'>
    80261728
    <class 'int'>
    None
    <class 'NoneType'>
    13376954
    <class 'int'>
    60196296
    <class 'int'>
    24078518
    <class 'int'>
    None
    <class 'NoneType'>
    40130864
    <class 'int'>
    100327160
    <class 'int'>
    33442386
    <class 'int'>
    6688477
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40130864
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2006543
    <class 'int'>
    10701563
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13376954
    <class 'int'>
    33442386
    <class 'int'>
    None
    <class 'NoneType'>
    13376954
    <class 'int'>
    None
    <class 'NoneType'>
    6688477
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2675390
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2006543
    <class 'int'>
    44143950
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20065432
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1471465
    <class 'int'>
    None
    <class 'NoneType'>
    20065432
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36117777
    <class 'int'>
    26753909
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1337695
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6688477
    <class 'int'>
    None
    <class 'NoneType'>
    10701563
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    70359397
    <class 'int'>
    84431277
    <class 'int'>
    28143759
    <class 'int'>
    61564472
    <class 'int'>
    70359397
    <class 'int'>
    28847353
    <class 'int'>
    65082442
    <class 'int'>
    56287518
    <class 'int'>
    47492593
    <class 'int'>
    31661728
    <class 'int'>
    123128945
    <class 'int'>
    43974623
    <class 'int'>
    96744171
    <class 'int'>
    17589849
    <class 'int'>
    13192387
    <class 'int'>
    56287518
    <class 'int'>
    52769548
    <class 'int'>
    2638477
    <class 'int'>
    14071879
    <class 'int'>
    879492
    <class 'int'>
    31661728
    <class 'int'>
    20228326
    <class 'int'>
    None
    <class 'NoneType'>
    31661728
    <class 'int'>
    24625789
    <class 'int'>
    26384774
    <class 'int'>
    None
    <class 'NoneType'>
    2110781
    <class 'int'>
    19348834
    <class 'int'>
    35179698
    <class 'int'>
    11433402
    <class 'int'>
    None
    <class 'NoneType'>
    32714340
    <class 'int'>
    39577161
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42215638
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8794924
    <class 'int'>
    54528533
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5276954
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24273992
    <class 'int'>
    None
    <class 'NoneType'>
    52769548
    <class 'int'>
    33420713
    <class 'int'>
    None
    <class 'NoneType'>
    17589849
    <class 'int'>
    12312894
    <class 'int'>
    None
    <class 'NoneType'>
    8794924
    <class 'int'>
    None
    <class 'NoneType'>
    13192387
    <class 'int'>
    None
    <class 'NoneType'>
    33420713
    <class 'int'>
    52769548
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26384774
    <class 'int'>
    12312894
    <class 'int'>
    1384321
    <class 'int'>
    22866804
    <class 'int'>
    None
    <class 'NoneType'>
    1134861
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11433402
    <class 'int'>
    None
    <class 'NoneType'>
    15830864
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4925157
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8794924
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21107819
    <class 'int'>
    1319238
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26384774
    <class 'int'>
    14152370
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26384774
    <class 'int'>
    None
    <class 'NoneType'>
    3517969
    <class 'int'>
    58046502
    <class 'int'>
    4397462
    <class 'int'>
    10553909
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8794924
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7035939
    <class 'int'>
    None
    <class 'NoneType'>
    12312894
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    228668
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40027321
    <class 'int'>
    160109284
    <class 'int'>
    112076498
    <class 'int'>
    76852456
    <class 'int'>
    48032785
    <class 'int'>
    56038249
    <class 'int'>
    43229506
    <class 'int'>
    36825135
    <class 'int'>
    38426228
    <class 'int'>
    None
    <class 'NoneType'>
    120081963
    <class 'int'>
    48032785
    <class 'int'>
    104071034
    <class 'int'>
    67245899
    <class 'int'>
    25617485
    <class 'int'>
    60841527
    <class 'int'>
    40027321
    <class 'int'>
    64043713
    <class 'int'>
    None
    <class 'NoneType'>
    30420763
    <class 'int'>
    14409835
    <class 'int'>
    26418031
    <class 'int'>
    None
    <class 'NoneType'>
    17612021
    <class 'int'>
    4002732
    <class 'int'>
    46431692
    <class 'int'>
    64043713
    <class 'int'>
    10407103
    <class 'int'>
    36825135
    <class 'int'>
    24016392
    <class 'int'>
    None
    <class 'NoneType'>
    56038249
    <class 'int'>
    None
    <class 'NoneType'>
    20814206
    <class 'int'>
    9606557
    <class 'int'>
    41628413
    <class 'int'>
    41628413
    <class 'int'>
    None
    <class 'NoneType'>
    17612021
    <class 'int'>
    None
    <class 'NoneType'>
    17612021
    <class 'int'>
    None
    <class 'NoneType'>
    12808742
    <class 'int'>
    30420763
    <class 'int'>
    None
    <class 'NoneType'>
    28819671
    <class 'int'>
    None
    <class 'NoneType'>
    25617485
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38426228
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16010928
    <class 'int'>
    22415299
    <class 'int'>
    48032785
    <class 'int'>
    9606557
    <class 'int'>
    22415299
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12808742
    <class 'int'>
    48032785
    <class 'int'>
    None
    <class 'NoneType'>
    22415299
    <class 'int'>
    None
    <class 'NoneType'>
    3202185
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13609289
    <class 'int'>
    12808742
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    56038249
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    56038249
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    76852456
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36825135
    <class 'int'>
    52836
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35224042
    <class 'int'>
    22415299
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27218578
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20814206
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1280874
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26257922
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    51628948
    <class 'int'>
    33190038
    <class 'int'>
    22126692
    <class 'int'>
    14751128
    <class 'int'>
    27658365
    <class 'int'>
    46097275
    <class 'int'>
    2765836
    <class 'int'>
    71911749
    <class 'int'>
    116165134
    <class 'int'>
    36877820
    <class 'int'>
    None
    <class 'NoneType'>
    27658365
    <class 'int'>
    129072371
    <class 'int'>
    22679859
    <class 'int'>
    36877820
    <class 'int'>
    None
    <class 'NoneType'>
    16595019
    <class 'int'>
    64536185
    <class 'int'>
    None
    <class 'NoneType'>
    12907237
    <class 'int'>
    None
    <class 'NoneType'>
    12907237
    <class 'int'>
    53472839
    <class 'int'>
    13829182
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35033929
    <class 'int'>
    32268092
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27658365
    <class 'int'>
    None
    <class 'NoneType'>
    51628948
    <class 'int'>
    7375564
    <class 'int'>
    5531673
    <class 'int'>
    29502256
    <class 'int'>
    None
    <class 'NoneType'>
    9219455
    <class 'int'>
    12907237
    <class 'int'>
    18438910
    <class 'int'>
    None
    <class 'NoneType'>
    5162894
    <class 'int'>
    None
    <class 'NoneType'>
    25814474
    <class 'int'>
    36877820
    <class 'int'>
    36877820
    <class 'int'>
    None
    <class 'NoneType'>
    3687782
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11432124
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19360855
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    55316730
    <class 'int'>
    None
    <class 'NoneType'>
    13829182
    <class 'int'>
    12907237
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2397058
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    40565602
    <class 'int'>
    44253384
    <class 'int'>
    40565602
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    23970583
    <class 'int'>
    None
    <class 'NoneType'>
    11063346
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7375564
    <class 'int'>
    9219455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2212669
    <class 'int'>
    None
    <class 'NoneType'>
    7375564
    <class 'int'>
    None
    <class 'NoneType'>
    18438910
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    42409493
    <class 'int'>
    27658365
    <class 'int'>
    8297509
    <class 'int'>
    None
    <class 'NoneType'>
    12907237
    <class 'int'>
    None
    <class 'NoneType'>
    42409
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2028280
    <class 'int'>
    4609727
    <class 'int'>
    12907237
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11063346
    <class 'int'>
    36877820
    <class 'int'>
    None
    <class 'NoneType'>
    3687782
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    783653
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4609727
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27658365
    <class 'int'>
    None
    <class 'NoneType'>
    29502256
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9219455
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34543447
    <class 'int'>
    32624367
    <class 'int'>
    11514482
    <class 'int'>
    43563125
    <class 'int'>
    76763217
    <class 'int'>
    24948045
    <class 'int'>
    6908689
    <class 'int'>
    28786206
    <class 'int'>
    51815171
    <class 'int'>
    44138850
    <class 'int'>
    30705287
    <class 'int'>
    9595402
    <class 'int'>
    28786206
    <class 'int'>
    38381608
    <class 'int'>
    32624367
    <class 'int'>
    11514482
    <class 'int'>
    47977010
    <class 'int'>
    24948045
    <class 'int'>
    44138850
    <class 'int'>
    26867126
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32624367
    <class 'int'>
    47977010
    <class 'int'>
    None
    <class 'NoneType'>
    16312183
    <class 'int'>
    None
    <class 'NoneType'>
    6716781
    <class 'int'>
    49339
    <class 'int'>
    3607882
    <class 'int'>
    5757241
    <class 'int'>
    38381608
    <class 'int'>
    None
    <class 'NoneType'>
    34543447
    <class 'int'>
    None
    <class 'NoneType'>
    38381608
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28786206
    <class 'int'>
    None
    <class 'NoneType'>
    72925056
    <class 'int'>
    4797701
    <class 'int'>
    19190804
    <class 'int'>
    32624367
    <class 'int'>
    30705287
    <class 'int'>
    28786206
    <class 'int'>
    None
    <class 'NoneType'>
    23028965
    <class 'int'>
    None
    <class 'NoneType'>
    9595402
    <class 'int'>
    21109884
    <class 'int'>
    None
    <class 'NoneType'>
    3838160
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5757241
    <class 'int'>
    None
    <class 'NoneType'>
    1919080
    <class 'int'>
    13817379
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12474022
    <class 'int'>
    32624367
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34543447
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26867126
    <class 'int'>
    None
    <class 'NoneType'>
    23028965
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3454344
    <class 'int'>
    1919080
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11514482
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36462528
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5757241
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    479770
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    182
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    75227563
    <class 'int'>
    36359988
    <class 'int'>
    714661
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34479299
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20060683
    <class 'int'>
    None
    <class 'NoneType'>
    62689636
    <class 'int'>
    11284134
    <class 'int'>
    48271019
    <class 'int'>
    9403445
    <class 'int'>
    26956543
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43882745
    <class 'int'>
    62689636
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11284134
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1723964
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5955515
    <class 'int'>
    None
    <class 'NoneType'>
    57489690
    <class 'int'>
    26533703
    <class 'int'>
    30955987
    <class 'int'>
    12382394
    <class 'int'>
    375894
    <class 'int'>
    265337
    <class 'int'>
    7075654
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11497938
    <class 'int'>
    None
    <class 'NoneType'>
    13266851
    <class 'int'>
    13266851
    <class 'int'>
    None
    <class 'NoneType'>
    28744845
    <class 'int'>
    2211141
    <class 'int'>
    884456
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    110557
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13266851
    <class 'int'>
    None
    <class 'NoneType'>
    17689135
    <class 'int'>
    None
    <class 'NoneType'>
    61911974
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    6633425
    <class 'int'>
    30955987
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3095598
    <class 'int'>
    1768913
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12155463
    <class 'int'>
    28362748
    <class 'int'>
    1620728
    <class 'int'>
    4862185
    <class 'int'>
    44570032
    <class 'int'>
    20259105
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1215546
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10939917
    <class 'int'>
    None
    <class 'NoneType'>
    20259105
    <class 'int'>
    None
    <class 'NoneType'>
    8914006
    <class 'int'>
    None
    <class 'NoneType'>
    6077731
    <class 'int'>
    None
    <class 'NoneType'>
    7293278
    <class 'int'>
    None
    <class 'NoneType'>
    725275
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    60777317
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7929293
    <class 'int'>
    108126733
    <class 'int'>
    14416897
    <class 'int'>
    14416897
    <class 'int'>
    15858587
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    108126733
    <class 'int'>
    None
    <class 'NoneType'>
    23067036
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7064279
    <class 'int'>
    None
    <class 'NoneType'>
    19102389
    <class 'int'>
    908
    <class 'int'>
    72084488
    <class 'int'>
    446923
    <class 'int'>
    None
    <class 'NoneType'>
    3604224
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    216253
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24605935
    <class 'int'>
    12654481
    <class 'int'>
    42181604
    <class 'int'>
    3515133
    <class 'int'>
    None
    <class 'NoneType'>
    15010752
    <class 'int'>
    119514545
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5504699
    <class 'int'>
    21090802
    <class 'int'>
    None
    <class 'NoneType'>
    8436320
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    24605935
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    456967
    <class 'int'>
    11847428
    <class 'int'>
    16155584
    <class 'int'>
    38773403
    <class 'int'>
    None
    <class 'NoneType'>
    21540779
    <class 'int'>
    13462987
    <class 'int'>
    11847428
    <class 'int'>
    None
    <class 'NoneType'>
    107703898
    <class 'int'>
    4184296
    <class 'int'>
    6462233
    <class 'int'>
    9693350
    <class 'int'>
    3904266
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    8858645
    <class 'int'>
    2423337
    <class 'int'>
    4038896
    <class 'int'>
    10770389
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35003766
    <class 'int'>
    None
    <class 'NoneType'>
    4577415
    <class 'int'>
    None
    <class 'NoneType'>
    21540779
    <class 'int'>
    None
    <class 'NoneType'>
    48466754
    <class 'int'>
    7000753
    <class 'int'>
    7000753
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    13462987
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    538519
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    108478405
    <class 'int'>
    41722463
    <class 'int'>
    116822897
    <class 'int'>
    66755941
    <class 'int'>
    33377970
    <class 'int'>
    23364579
    <class 'int'>
    90120521
    <class 'int'>
    36715767
    <class 'int'>
    36715767
    <class 'int'>
    22530130
    <class 'int'>
    33377970
    <class 'int'>
    70093738
    <class 'int'>
    21695681
    <class 'int'>
    None
    <class 'NoneType'>
    50
    <class 'int'>
    58411448
    <class 'int'>
    100133912
    <class 'int'>
    None
    <class 'NoneType'>
    23364579
    <class 'int'>
    83444927
    <class 'int'>
    18357883
    <class 'int'>
    58411448
    <class 'int'>
    53404753
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    63418144
    <class 'int'>
    None
    <class 'NoneType'>
    10013391
    <class 'int'>
    6675594
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    15854536
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43391362
    <class 'int'>
    16688985
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    41722463
    <class 'int'>
    None
    <class 'NoneType'>
    18357883
    <class 'int'>
    16688985
    <class 'int'>
    11682289
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    78438231
    <class 'int'>
    26702376
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7009373
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    78438231
    <class 'int'>
    20026782
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    25033478
    <class 'int'>
    36715767
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1668898
    <class 'int'>
    17523434
    <class 'int'>
    None
    <class 'NoneType'>
    41722463
    <class 'int'>
    2503347
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5
    <class 'int'>
    None
    <class 'NoneType'>
    16688985
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    63302990
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4172246
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    18357883
    <class 'int'>
    31709072
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    81776028
    <class 'int'>
    38384666
    <class 'int'>
    6675594
    <class 'int'>
    None
    <class 'NoneType'>
    33377970
    <class 'int'>
    36715767
    <class 'int'>
    375502
    <class 'int'>
    None
    <class 'NoneType'>
    15020086
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    38381315
    <class 'int'>
    834449
    <class 'int'>
    30040173
    <class 'int'>
    51735854
    <class 'int'>
    4172246
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3451350
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    29179444
    <class 'int'>
    18237153
    <class 'int'>
    43769167
    <class 'int'>
    43769167
    <class 'int'>
    None
    <class 'NoneType'>
    21884583
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    14589722
    <class 'int'>
    36744215
    <class 'int'>
    29179444
    <class 'int'>
    None
    <class 'NoneType'>
    1458972
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    43769167
    <class 'int'>
    21884583
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21155097
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10942291
    <class 'int'>
    5949601
    <class 'int'>
    14745934
    <class 'int'>
    88475609
    <class 'int'>
    22118902
    <class 'int'>
    None
    <class 'NoneType'>
    5529725
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1990701
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    221189
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1474593
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22118902
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4981207
    <class 'int'>
    4214867
    <class 'int'>
    6897056
    <class 'int'>
    10728754
    <class 'int'>
    34485282
    <class 'int'>
    91960752
    <class 'int'>
    32569433
    <class 'int'>
    34485282
    <class 'int'>
    7663396
    <class 'int'>
    14177282
    <class 'int'>
    24906037
    <class 'int'>
    None
    <class 'NoneType'>
    14560452
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    574754
    <class 'int'>
    22990188
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5747547
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22990188
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    134109430
    <class 'int'>
    None
    <class 'NoneType'>
    22038261
    <class 'int'>
    27170459
    <class 'int'>
    33208339
    <class 'int'>
    95096607
    <class 'int'>
    86039787
    <class 'int'>
    37736749
    <class 'int'>
    66416678
    <class 'int'>
    60378798
    <class 'int'>
    39246219
    <class 'int'>
    10415342
    <class 'int'>
    128304946
    <class 'int'>
    45284098
    <class 'int'>
    30189399
    <class 'int'>
    None
    <class 'NoneType'>
    105662897
    <class 'int'>
    60378798
    <class 'int'>
    31
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21132579
    <class 'int'>
    31698869
    <class 'int'>
    37736749
    <class 'int'>
    None
    <class 'NoneType'>
    45284098
    <class 'int'>
    45284098
    <class 'int'>
    37736749
    <class 'int'>
    18868374
    <class 'int'>
    63397738
    <class 'int'>
    57359858
    <class 'int'>
    None
    <class 'NoneType'>
    67926148
    <class 'int'>
    45284098
    <class 'int'>
    33208339
    <class 'int'>
    16604169
    <class 'int'>
    52831448
    <class 'int'>
    13585229
    <class 'int'>
    33208339
    <class 'int'>
    20377844
    <class 'int'>
    37736749
    <class 'int'>
    None
    <class 'NoneType'>
    63397738
    <class 'int'>
    22642049
    <class 'int'>
    None
    <class 'NoneType'>
    42265158
    <class 'int'>
    None
    <class 'NoneType'>
    19623109
    <class 'int'>
    19623109
    <class 'int'>
    30189399
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45
    <class 'int'>
    10566289
    <class 'int'>
    52831448
    <class 'int'>
    52831448
    <class 'int'>
    None
    <class 'NoneType'>
    12075759
    <class 'int'>
    25660989
    <class 'int'>
    None
    <class 'NoneType'>
    27170459
    <class 'int'>
    17358904
    <class 'int'>
    9056819
    <class 'int'>
    51321978
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    48303038
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4528409
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12
    <class 'int'>
    None
    <class 'NoneType'>
    18113639
    <class 'int'>
    33208339
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22642049
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    30189399
    <class 'int'>
    37736749
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    57359858
    <class 'int'>
    28679929
    <class 'int'>
    15094699
    <class 'int'>
    5283144
    <class 'int'>
    18
    <class 'int'>
    45284098
    <class 'int'>
    None
    <class 'NoneType'>
    49812508
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12075759
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1358522
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    37736749
    <class 'int'>
    1132102
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    52831448
    <class 'int'>
    2264204
    <class 'int'>
    None
    <class 'NoneType'>
    21132579
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    12075759
    <class 'int'>
    None
    <class 'NoneType'>
    1358522
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    16604169
    <class 'int'>
    33208339
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9056819
    <class 'int'>
    30016643
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7547349
    <class 'int'>
    None
    <class 'NoneType'>
    9056819
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9056819
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    21132579
    <class 'int'>
    188683
    <class 'int'>
    None
    <class 'NoneType'>
    3018939
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    45
    <class 'int'>
    21132579
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3018939
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1509469
    <class 'int'>
    102643
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26147054
    <class 'int'>
    19610291
    <class 'int'>
    78441164
    <class 'int'>
    62099255
    <class 'int'>
    16341909
    <class 'int'>
    None
    <class 'NoneType'>
    19610291
    <class 'int'>
    26147054
    <class 'int'>
    None
    <class 'NoneType'>
    6190315
    <class 'int'>
    None
    <class 'NoneType'>
    163419093
    <class 'int'>
    13073527
    <class 'int'>
    None
    <class 'NoneType'>
    4183528
    <class 'int'>
    26147054
    <class 'int'>
    None
    <class 'NoneType'>
    22878673
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    117661747
    <class 'int'>
    None
    <class 'NoneType'>
    32683818
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22878673
    <class 'int'>
    None
    <class 'NoneType'>
    424889
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1307
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    17800448
    <class 'int'>
    None
    <class 'NoneType'>
    17800448
    <class 'int'>
    28480717
    <class 'int'>
    None
    <class 'NoneType'>
    221544381
    <class 'int'>
    66929686
    <class 'int'>
    None
    <class 'NoneType'>
    28480717
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5340134
    <class 'int'>
    None
    <class 'NoneType'>
    7120179
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    121043049
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7120179
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    473491
    <class 'int'>
    36804018
    <class 'int'>
    31830502
    <class 'int'>
    11936438
    <class 'int'>
    29841096
    <class 'int'>
    11936438
    <class 'int'>
    15915251
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    29841096
    <class 'int'>
    33819908
    <class 'int'>
    27851689
    <class 'int'>
    79576256
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    11936438
    <class 'int'>
    47745753
    <class 'int'>
    73608037
    <class 'int'>
    49735160
    <class 'int'>
    27453808
    <class 'int'>
    39788128
    <class 'int'>
    49735160
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    224
    <class 'int'>
    None
    <class 'NoneType'>
    49735160
    <class 'int'>
    16909954
    <class 'int'>
    9947032
    <class 'int'>
    29841096
    <class 'int'>
    49735160
    <class 'int'>
    None
    <class 'NoneType'>
    37798721
    <class 'int'>
    19894064
    <class 'int'>
    49735160
    <class 'int'>
    None
    <class 'NoneType'>
    48740457
    <class 'int'>
    17904657
    <class 'int'>
    5968219
    <class 'int'>
    None
    <class 'NoneType'>
    8952328
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3978812
    <class 'int'>
    None
    <class 'NoneType'>
    7957625
    <class 'int'>
    None
    <class 'NoneType'>
    2984109
    <class 'int'>
    12732201
    <class 'int'>
    None
    <class 'NoneType'>
    19894064
    <class 'int'>
    8952328
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    27851689
    <class 'int'>
    None
    <class 'NoneType'>
    19894064
    <class 'int'>
    5968219
    <class 'int'>
    None
    <class 'NoneType'>
    35809315
    <class 'int'>
    None
    <class 'NoneType'>
    9947032
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    220824
    <class 'int'>
    44761
    <class 'int'>
    5371397
    <class 'int'>
    None
    <class 'NoneType'>
    9350210
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    19894064
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    5968219
    <class 'int'>
    23872876
    <class 'int'>
    None
    <class 'NoneType'>
    3978812
    <class 'int'>
    None
    <class 'NoneType'>
    35809315
    <class 'int'>
    None
    <class 'NoneType'>
    12931141
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2188347
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    29841096
    <class 'int'>
    None
    <class 'NoneType'>
    1591525
    <class 'int'>
    23872876
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    29841096
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    35809315
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3779872
    <class 'int'>
    29841096
    <class 'int'>
    None
    <class 'NoneType'>
    16909954
    <class 'int'>
    None
    <class 'NoneType'>
    73667393
    <class 'int'>
    39289276
    <class 'int'>
    34378117
    <class 'int'>
    2455579
    <class 'int'>
    4174485
    <class 'int'>
    None
    <class 'NoneType'>
    8348971
    <class 'int'>
    7366739
    <class 'int'>
    27011377
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    3815970
    <class 'int'>
    None
    <class 'NoneType'>
    58933914
    <class 'int'>
    9822319
    <class 'int'>
    None
    <class 'NoneType'>
    14733478
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    22781803
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2455579
    <class 'int'>
    None
    <class 'NoneType'>
    2455579
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2210021
    <class 'int'>
    1350568
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4911159
    <class 'int'>
    22465473
    <class 'int'>
    16849105
    <class 'int'>
    19657289
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    67396420
    <class 'int'>
    12356010
    <class 'int'>
    None
    <class 'NoneType'>
    143133147
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    28081
    <class 'int'>
    56163683
    <class 'int'>
    39314578
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    84245525
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    336982
    <class 'int'>
    None
    <class 'NoneType'>
    20875737
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    33698210
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    56163683
    <class 'int'>
    140409208
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    76126200
    <class 'int'>
    56748622
    <class 'int'>
    96887891
    <class 'int'>
    38755156
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10380845
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    311425
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    39862446
    <class 'int'>
    795
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2076169
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    138411273
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26298142
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    41609849
    <class 'int'>
    35665585
    <class 'int'>
    2139935
    <class 'int'>
    21399351
    <class 'int'>
    37116502
    <class 'int'>
    8916396
    <class 'int'>
    71331170
    <class 'int'>
    23777056
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7430330
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    1040246
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    20056204
    <class 'int'>
    66854013
    <class 'int'>
    2172755
    <class 'int'>
    183848538
    <class 'int'>
    1002810
    <class 'int'>
    9025291
    <class 'int'>
    None
    <class 'NoneType'>
    50140510
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    7688211
    <class 'int'>
    13370802
    <class 'int'>
    16713503
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    26474189
    <class 'int'>
    None
    <class 'NoneType'>
    36
    <class 'int'>
    16713503
    <class 'int'>
    40112408
    <class 'int'>
    11699452
    <class 'int'>
    2206182
    <class 'int'>
    33427006
    <class 'int'>
    None
    <class 'NoneType'>
    2172755
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80224816
    <class 'int'>
    None
    <class 'NoneType'>
    9025291
    <class 'int'>
    22730364
    <class 'int'>
    None
    <class 'NoneType'>
    3342700
    <class 'int'>
    None
    <class 'NoneType'>
    10028102
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    36769707
    <class 'int'>
    None
    <class 'NoneType'>
    22486347
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    300843
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    10028102
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    2116174
    <class 'int'>
    None
    <class 'NoneType'>
    50385110
    <class 'int'>
    None
    <class 'NoneType'>
    120924264
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    9256080
    <class 'int'>
    20154044
    <class 'int'>
    31258922
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    32246470
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    80616176
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    503851
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    34362645
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    4702610
    <class 'int'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    None
    <class 'NoneType'>
    127642
    <class 'int'>
    


```python
for tmdbmovie in tmdbmovies:
    tmdbmovie['vote_average'] = float(tmdbmovie['vote_average'])
    tmdbmovie['vote_count'] = int(tmdbmovie['vote_count'])
print(tmdbmovies[0]['vote_average'])
print(tmdbmovies[0]['vote_count'])
```

    6.5
    5562
    

Also parsing the dates from string to datetime


```python
#I created a copy for the new list of dicts with parsed dates
from datetime import datetime as dt
def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%m/%d/%y')

tmdbmovies_with_datetime = tmdbmovies.copy()
    
for tmdbmovie_with_datetime in tmdbmovies_with_datetime:
    print(tmdbmovie_with_datetime['release_date'])
    tmdbmovie_with_datetime['release_date'] = parse_date(tmdbmovie_with_datetime['release_date'])
    print(tmdbmovie_with_datetime['release_date'])
```

    6/9/15
    2015-06-09 00:00:00
    5/13/15
    2015-05-13 00:00:00
    3/18/15
    2015-03-18 00:00:00
    12/15/15
    2015-12-15 00:00:00
    4/1/15
    2015-04-01 00:00:00
    12/25/15
    2015-12-25 00:00:00
    6/23/15
    2015-06-23 00:00:00
    9/30/15
    2015-09-30 00:00:00
    6/17/15
    2015-06-17 00:00:00
    6/9/15
    2015-06-09 00:00:00
    10/26/15
    2015-10-26 00:00:00
    2/4/15
    2015-02-04 00:00:00
    1/21/15
    2015-01-21 00:00:00
    7/16/15
    2015-07-16 00:00:00
    4/22/15
    2015-04-22 00:00:00
    12/25/15
    2015-12-25 00:00:00
    1/1/15
    2015-01-01 00:00:00
    7/14/15
    2015-07-14 00:00:00
    3/12/15
    2015-03-12 00:00:00
    11/18/15
    2015-11-18 00:00:00
    5/19/15
    2015-05-19 00:00:00
    6/15/15
    2015-06-15 00:00:00
    5/27/15
    2015-05-27 00:00:00
    2/11/15
    2015-02-11 00:00:00
    12/11/15
    2015-12-11 00:00:00
    7/23/15
    2015-07-23 00:00:00
    6/25/15
    2015-06-25 00:00:00
    1/24/15
    2015-01-24 00:00:00
    11/6/15
    2015-11-06 00:00:00
    9/9/15
    2015-09-09 00:00:00
    6/19/15
    2015-06-19 00:00:00
    3/4/15
    2015-03-04 00:00:00
    5/7/15
    2015-05-07 00:00:00
    10/15/15
    2015-10-15 00:00:00
    8/5/15
    2015-08-05 00:00:00
    10/16/15
    2015-10-16 00:00:00
    9/3/15
    2015-09-03 00:00:00
    11/14/15
    2015-11-14 00:00:00
    3/11/15
    2015-03-11 00:00:00
    11/4/15
    2015-11-04 00:00:00
    8/13/15
    2015-08-13 00:00:00
    10/21/15
    2015-10-21 00:00:00
    10/9/15
    2015-10-09 00:00:00
    8/13/15
    2015-08-13 00:00:00
    4/16/15
    2015-04-16 00:00:00
    9/12/15
    2015-09-12 00:00:00
    3/18/15
    2015-03-18 00:00:00
    7/9/15
    2015-07-09 00:00:00
    1/14/15
    2015-01-14 00:00:00
    10/8/15
    2015-10-08 00:00:00
    11/20/15
    2015-11-20 00:00:00
    9/24/15
    2015-09-24 00:00:00
    9/10/15
    2015-09-10 00:00:00
    12/24/15
    2015-12-24 00:00:00
    12/3/15
    2015-12-03 00:00:00
    10/2/15
    2015-10-02 00:00:00
    7/10/15
    2015-07-10 00:00:00
    1/21/15
    2015-01-21 00:00:00
    11/27/15
    2015-11-27 00:00:00
    1/13/15
    2015-01-13 00:00:00
    9/17/15
    2015-09-17 00:00:00
    8/5/15
    2015-08-05 00:00:00
    3/24/15
    2015-03-24 00:00:00
    5/6/15
    2015-05-06 00:00:00
    2/25/15
    2015-02-25 00:00:00
    8/20/15
    2015-08-20 00:00:00
    11/20/15
    2015-11-20 00:00:00
    5/21/15
    2015-05-21 00:00:00
    11/25/15
    2015-11-25 00:00:00
    7/17/15
    2015-07-17 00:00:00
    9/11/15
    2015-09-11 00:00:00
    9/9/15
    2015-09-09 00:00:00
    4/17/15
    2015-04-17 00:00:00
    9/24/15
    2015-09-24 00:00:00
    6/24/15
    2015-06-24 00:00:00
    6/12/15
    2015-06-12 00:00:00
    11/5/15
    2015-11-05 00:00:00
    11/12/15
    2015-11-12 00:00:00
    11/20/15
    2015-11-20 00:00:00
    6/26/15
    2015-06-26 00:00:00
    4/9/15
    2015-04-09 00:00:00
    9/21/15
    2015-09-21 00:00:00
    11/12/15
    2015-11-12 00:00:00
    9/30/15
    2015-09-30 00:00:00
    1/15/15
    2015-01-15 00:00:00
    7/28/15
    2015-07-28 00:00:00
    5/20/15
    2015-05-20 00:00:00
    5/8/15
    2015-05-08 00:00:00
    9/4/15
    2015-09-04 00:00:00
    3/15/15
    2015-03-15 00:00:00
    7/30/15
    2015-07-30 00:00:00
    12/25/15
    2015-12-25 00:00:00
    12/19/15
    2015-12-19 00:00:00
    1/16/15
    2015-01-16 00:00:00
    7/29/15
    2015-07-29 00:00:00
    12/17/15
    2015-12-17 00:00:00
    10/1/15
    2015-10-01 00:00:00
    2/20/15
    2015-02-20 00:00:00
    8/19/15
    2015-08-19 00:00:00
    4/16/15
    2015-04-16 00:00:00
    3/9/15
    2015-03-09 00:00:00
    9/26/15
    2015-09-26 00:00:00
    1/23/15
    2015-01-23 00:00:00
    4/11/15
    2015-04-11 00:00:00
    6/3/15
    2015-06-03 00:00:00
    1/16/15
    2015-01-16 00:00:00
    10/23/15
    2015-10-23 00:00:00
    9/10/15
    2015-09-10 00:00:00
    9/4/15
    2015-09-04 00:00:00
    11/13/15
    2015-11-13 00:00:00
    11/26/15
    2015-11-26 00:00:00
    3/31/15
    2015-03-31 00:00:00
    8/26/15
    2015-08-26 00:00:00
    6/19/15
    2015-06-19 00:00:00
    8/19/15
    2015-08-19 00:00:00
    4/10/15
    2015-04-10 00:00:00
    1/9/15
    2015-01-09 00:00:00
    12/18/15
    2015-12-18 00:00:00
    1/30/15
    2015-01-30 00:00:00
    6/26/15
    2015-06-26 00:00:00
    9/3/15
    2015-09-03 00:00:00
    10/13/15
    2015-10-13 00:00:00
    1/23/15
    2015-01-23 00:00:00
    7/1/15
    2015-07-01 00:00:00
    10/21/15
    2015-10-21 00:00:00
    4/16/15
    2015-04-16 00:00:00
    10/27/15
    2015-10-27 00:00:00
    10/14/15
    2015-10-14 00:00:00
    8/14/15
    2015-08-14 00:00:00
    2/5/15
    2015-02-05 00:00:00
    4/17/15
    2015-04-17 00:00:00
    6/26/15
    2015-06-26 00:00:00
    2/5/15
    2015-02-05 00:00:00
    5/18/15
    2015-05-18 00:00:00
    8/28/15
    2015-08-28 00:00:00
    2/16/15
    2015-02-16 00:00:00
    4/17/15
    2015-04-17 00:00:00
    5/8/15
    2015-05-08 00:00:00
    6/4/15
    2015-06-04 00:00:00
    8/13/15
    2015-08-13 00:00:00
    2/24/15
    2015-02-24 00:00:00
    11/13/15
    2015-11-13 00:00:00
    11/10/15
    2015-11-10 00:00:00
    12/24/15
    2015-12-24 00:00:00
    12/30/15
    2015-12-30 00:00:00
    10/23/15
    2015-10-23 00:00:00
    10/1/15
    2015-10-01 00:00:00
    9/16/15
    2015-09-16 00:00:00
    9/17/15
    2015-09-17 00:00:00
    8/1/15
    2015-08-01 00:00:00
    8/7/15
    2015-08-07 00:00:00
    9/16/15
    2015-09-16 00:00:00
    2/6/15
    2015-02-06 00:00:00
    4/3/15
    2015-04-03 00:00:00
    5/14/15
    2015-05-14 00:00:00
    7/17/15
    2015-07-17 00:00:00
    9/3/15
    2015-09-03 00:00:00
    7/10/15
    2015-07-10 00:00:00
    9/12/15
    2015-09-12 00:00:00
    5/28/15
    2015-05-28 00:00:00
    9/11/15
    2015-09-11 00:00:00
    11/26/15
    2015-11-26 00:00:00
    3/27/15
    2015-03-27 00:00:00
    3/26/15
    2015-03-26 00:00:00
    3/12/15
    2015-03-12 00:00:00
    12/11/15
    2015-12-11 00:00:00
    4/10/15
    2015-04-10 00:00:00
    8/6/15
    2015-08-06 00:00:00
    10/16/15
    2015-10-16 00:00:00
    8/28/15
    2015-08-28 00:00:00
    8/26/15
    2015-08-26 00:00:00
    4/30/15
    2015-04-30 00:00:00
    8/16/15
    2015-08-16 00:00:00
    12/18/15
    2015-12-18 00:00:00
    7/31/15
    2015-07-31 00:00:00
    7/31/15
    2015-07-31 00:00:00
    4/18/15
    2015-04-18 00:00:00
    5/1/15
    2015-05-01 00:00:00
    7/31/15
    2015-07-31 00:00:00
    2/26/15
    2015-02-26 00:00:00
    4/23/15
    2015-04-23 00:00:00
    2/10/15
    2015-02-10 00:00:00
    10/23/15
    2015-10-23 00:00:00
    5/27/15
    2015-05-27 00:00:00
    2/10/15
    2015-02-10 00:00:00
    10/27/15
    2015-10-27 00:00:00
    9/11/15
    2015-09-11 00:00:00
    6/3/15
    2015-06-03 00:00:00
    3/13/15
    2015-03-13 00:00:00
    4/17/15
    2015-04-17 00:00:00
    12/31/15
    2015-12-31 00:00:00
    8/7/15
    2015-08-07 00:00:00
    4/18/15
    2015-04-18 00:00:00
    6/26/15
    2015-06-26 00:00:00
    9/16/15
    2015-09-16 00:00:00
    7/3/15
    2015-07-03 00:00:00
    6/4/15
    2015-06-04 00:00:00
    11/6/15
    2015-11-06 00:00:00
    1/29/15
    2015-01-29 00:00:00
    10/1/15
    2015-10-01 00:00:00
    10/2/15
    2015-10-02 00:00:00
    5/29/15
    2015-05-29 00:00:00
    7/6/15
    2015-07-06 00:00:00
    9/12/15
    2015-09-12 00:00:00
    5/31/15
    2015-05-31 00:00:00
    3/20/15
    2015-03-20 00:00:00
    10/22/15
    2015-10-22 00:00:00
    10/1/15
    2015-10-01 00:00:00
    11/27/15
    2015-11-27 00:00:00
    7/24/15
    2015-07-24 00:00:00
    9/25/15
    2015-09-25 00:00:00
    2/20/15
    2015-02-20 00:00:00
    9/10/15
    2015-09-10 00:00:00
    6/3/15
    2015-06-03 00:00:00
    7/4/15
    2015-07-04 00:00:00
    10/29/15
    2015-10-29 00:00:00
    6/19/15
    2015-06-19 00:00:00
    8/14/15
    2015-08-14 00:00:00
    9/27/15
    2015-09-27 00:00:00
    9/15/15
    2015-09-15 00:00:00
    4/2/15
    2015-04-02 00:00:00
    8/7/15
    2015-08-07 00:00:00
    9/2/15
    2015-09-02 00:00:00
    9/2/15
    2015-09-02 00:00:00
    3/2/15
    2015-03-02 00:00:00
    1/29/15
    2015-01-29 00:00:00
    4/14/15
    2015-04-14 00:00:00
    5/6/15
    2015-05-06 00:00:00
    1/24/15
    2015-01-24 00:00:00
    3/5/15
    2015-03-05 00:00:00
    4/8/15
    2015-04-08 00:00:00
    3/23/15
    2015-03-23 00:00:00
    10/9/15
    2015-10-09 00:00:00
    11/20/15
    2015-11-20 00:00:00
    7/24/15
    2015-07-24 00:00:00
    2/20/15
    2015-02-20 00:00:00
    7/25/15
    2015-07-25 00:00:00
    3/13/15
    2015-03-13 00:00:00
    3/13/15
    2015-03-13 00:00:00
    4/9/15
    2015-04-09 00:00:00
    10/24/15
    2015-10-24 00:00:00
    5/29/15
    2015-05-29 00:00:00
    7/10/15
    2015-07-10 00:00:00
    9/4/15
    2015-09-04 00:00:00
    10/9/15
    2015-10-09 00:00:00
    10/16/15
    2015-10-16 00:00:00
    11/12/15
    2015-11-12 00:00:00
    10/16/15
    2015-10-16 00:00:00
    10/24/15
    2015-10-24 00:00:00
    8/28/15
    2015-08-28 00:00:00
    4/6/15
    2015-04-06 00:00:00
    6/12/15
    2015-06-12 00:00:00
    6/5/15
    2015-06-05 00:00:00
    6/4/15
    2015-06-04 00:00:00
    9/15/15
    2015-09-15 00:00:00
    10/2/15
    2015-10-02 00:00:00
    1/29/15
    2015-01-29 00:00:00
    3/10/15
    2015-03-10 00:00:00
    8/25/15
    2015-08-25 00:00:00
    11/27/15
    2015-11-27 00:00:00
    1/16/15
    2015-01-16 00:00:00
    1/9/15
    2015-01-09 00:00:00
    10/6/15
    2015-10-06 00:00:00
    11/3/15
    2015-11-03 00:00:00
    7/3/15
    2015-07-03 00:00:00
    3/5/15
    2015-03-05 00:00:00
    3/14/15
    2015-03-14 00:00:00
    12/4/15
    2015-12-04 00:00:00
    3/6/15
    2015-03-06 00:00:00
    1/23/15
    2015-01-23 00:00:00
    1/30/15
    2015-01-30 00:00:00
    3/27/15
    2015-03-27 00:00:00
    8/28/15
    2015-08-28 00:00:00
    7/28/15
    2015-07-28 00:00:00
    9/11/15
    2015-09-11 00:00:00
    4/9/15
    2015-04-09 00:00:00
    9/17/15
    2015-09-17 00:00:00
    8/1/15
    2015-08-01 00:00:00
    8/14/15
    2015-08-14 00:00:00
    11/3/15
    2015-11-03 00:00:00
    9/16/15
    2015-09-16 00:00:00
    7/31/15
    2015-07-31 00:00:00
    10/9/15
    2015-10-09 00:00:00
    3/17/15
    2015-03-17 00:00:00
    12/4/15
    2015-12-04 00:00:00
    11/6/15
    2015-11-06 00:00:00
    5/18/15
    2015-05-18 00:00:00
    10/22/15
    2015-10-22 00:00:00
    4/16/15
    2015-04-16 00:00:00
    2/27/15
    2015-02-27 00:00:00
    12/23/15
    2015-12-23 00:00:00
    9/11/15
    2015-09-11 00:00:00
    10/2/15
    2015-10-02 00:00:00
    12/4/15
    2015-12-04 00:00:00
    7/22/15
    2015-07-22 00:00:00
    11/26/15
    2015-11-26 00:00:00
    8/5/15
    2015-08-05 00:00:00
    2/12/15
    2015-02-12 00:00:00
    9/14/15
    2015-09-14 00:00:00
    7/24/15
    2015-07-24 00:00:00
    1/23/15
    2015-01-23 00:00:00
    1/6/15
    2015-01-06 00:00:00
    6/13/15
    2015-06-13 00:00:00
    3/17/15
    2015-03-17 00:00:00
    5/8/15
    2015-05-08 00:00:00
    11/22/15
    2015-11-22 00:00:00
    5/14/15
    2015-05-14 00:00:00
    3/1/15
    2015-03-01 00:00:00
    9/13/15
    2015-09-13 00:00:00
    4/10/15
    2015-04-10 00:00:00
    10/16/15
    2015-10-16 00:00:00
    10/5/15
    2015-10-05 00:00:00
    8/21/15
    2015-08-21 00:00:00
    5/22/15
    2015-05-22 00:00:00
    3/20/15
    2015-03-20 00:00:00
    2/24/15
    2015-02-24 00:00:00
    6/5/15
    2015-06-05 00:00:00
    3/2/15
    2015-03-02 00:00:00
    7/20/15
    2015-07-20 00:00:00
    10/8/15
    2015-10-08 00:00:00
    11/6/15
    2015-11-06 00:00:00
    6/15/15
    2015-06-15 00:00:00
    12/31/15
    2015-12-31 00:00:00
    1/31/15
    2015-01-31 00:00:00
    2/13/15
    2015-02-13 00:00:00
    5/15/15
    2015-05-15 00:00:00
    5/19/15
    2015-05-19 00:00:00
    8/7/15
    2015-08-07 00:00:00
    6/23/15
    2015-06-23 00:00:00
    1/25/15
    2015-01-25 00:00:00
    9/29/15
    2015-09-29 00:00:00
    7/1/15
    2015-07-01 00:00:00
    4/19/15
    2015-04-19 00:00:00
    1/31/15
    2015-01-31 00:00:00
    11/9/15
    2015-11-09 00:00:00
    1/25/15
    2015-01-25 00:00:00
    8/28/15
    2015-08-28 00:00:00
    8/3/15
    2015-08-03 00:00:00
    8/14/15
    2015-08-14 00:00:00
    8/13/15
    2015-08-13 00:00:00
    10/2/15
    2015-10-02 00:00:00
    4/17/15
    2015-04-17 00:00:00
    7/16/15
    2015-07-16 00:00:00
    11/4/15
    2015-11-04 00:00:00
    5/15/15
    2015-05-15 00:00:00
    10/18/15
    2015-10-18 00:00:00
    8/7/15
    2015-08-07 00:00:00
    2/21/15
    2015-02-21 00:00:00
    10/14/15
    2015-10-14 00:00:00
    10/6/15
    2015-10-06 00:00:00
    9/17/15
    2015-09-17 00:00:00
    1/30/15
    2015-01-30 00:00:00
    11/8/15
    2015-11-08 00:00:00
    9/19/15
    2015-09-19 00:00:00
    10/30/15
    2015-10-30 00:00:00
    6/26/15
    2015-06-26 00:00:00
    8/18/15
    2015-08-18 00:00:00
    2/13/15
    2015-02-13 00:00:00
    8/27/15
    2015-08-27 00:00:00
    8/7/15
    2015-08-07 00:00:00
    5/12/15
    2015-05-12 00:00:00
    8/11/15
    2015-08-11 00:00:00
    10/16/15
    2015-10-16 00:00:00
    7/17/15
    2015-07-17 00:00:00
    12/11/15
    2015-12-11 00:00:00
    5/16/15
    2015-05-16 00:00:00
    1/20/15
    2015-01-20 00:00:00
    5/21/15
    2015-05-21 00:00:00
    10/23/15
    2015-10-23 00:00:00
    9/17/15
    2015-09-17 00:00:00
    1/25/15
    2015-01-25 00:00:00
    11/25/15
    2015-11-25 00:00:00
    8/6/15
    2015-08-06 00:00:00
    12/4/15
    2015-12-04 00:00:00
    4/10/15
    2015-04-10 00:00:00
    7/22/15
    2015-07-22 00:00:00
    9/11/15
    2015-09-11 00:00:00
    9/2/15
    2015-09-02 00:00:00
    5/29/15
    2015-05-29 00:00:00
    5/2/15
    2015-05-02 00:00:00
    9/11/15
    2015-09-11 00:00:00
    8/8/15
    2015-08-08 00:00:00
    7/24/15
    2015-07-24 00:00:00
    1/27/15
    2015-01-27 00:00:00
    9/6/15
    2015-09-06 00:00:00
    10/16/15
    2015-10-16 00:00:00
    1/22/15
    2015-01-22 00:00:00
    3/20/15
    2015-03-20 00:00:00
    3/13/15
    2015-03-13 00:00:00
    1/24/15
    2015-01-24 00:00:00
    9/13/15
    2015-09-13 00:00:00
    4/15/15
    2015-04-15 00:00:00
    3/11/15
    2015-03-11 00:00:00
    8/29/15
    2015-08-29 00:00:00
    4/30/15
    2015-04-30 00:00:00
    9/14/15
    2015-09-14 00:00:00
    10/20/15
    2015-10-20 00:00:00
    3/14/15
    2015-03-14 00:00:00
    5/2/15
    2015-05-02 00:00:00
    10/1/15
    2015-10-01 00:00:00
    11/13/15
    2015-11-13 00:00:00
    3/15/15
    2015-03-15 00:00:00
    8/28/15
    2015-08-28 00:00:00
    8/21/15
    2015-08-21 00:00:00
    8/14/15
    2015-08-14 00:00:00
    4/14/15
    2015-04-14 00:00:00
    11/13/15
    2015-11-13 00:00:00
    3/14/15
    2015-03-14 00:00:00
    6/5/15
    2015-06-05 00:00:00
    6/4/15
    2015-06-04 00:00:00
    1/1/15
    2015-01-01 00:00:00
    8/30/15
    2015-08-30 00:00:00
    1/16/15
    2015-01-16 00:00:00
    6/12/15
    2015-06-12 00:00:00
    5/16/15
    2015-05-16 00:00:00
    8/16/15
    2015-08-16 00:00:00
    8/31/15
    2015-08-31 00:00:00
    5/7/15
    2015-05-07 00:00:00
    1/15/15
    2015-01-15 00:00:00
    4/14/15
    2015-04-14 00:00:00
    9/12/15
    2015-09-12 00:00:00
    4/18/15
    2015-04-18 00:00:00
    3/20/15
    2015-03-20 00:00:00
    7/11/15
    2015-07-11 00:00:00
    10/29/15
    2015-10-29 00:00:00
    6/26/15
    2015-06-26 00:00:00
    9/11/15
    2015-09-11 00:00:00
    9/25/15
    2015-09-25 00:00:00
    6/30/15
    2015-06-30 00:00:00
    2/10/15
    2015-02-10 00:00:00
    12/1/15
    2015-12-01 00:00:00
    12/11/15
    2015-12-11 00:00:00
    4/11/15
    2015-04-11 00:00:00
    8/26/15
    2015-08-26 00:00:00
    2/27/15
    2015-02-27 00:00:00
    12/1/15
    2015-12-01 00:00:00
    1/22/15
    2015-01-22 00:00:00
    4/17/15
    2015-04-17 00:00:00
    1/20/15
    2015-01-20 00:00:00
    8/14/15
    2015-08-14 00:00:00
    5/1/15
    2015-05-01 00:00:00
    10/9/15
    2015-10-09 00:00:00
    3/16/15
    2015-03-16 00:00:00
    9/14/15
    2015-09-14 00:00:00
    7/9/15
    2015-07-09 00:00:00
    10/9/15
    2015-10-09 00:00:00
    3/16/15
    2015-03-16 00:00:00
    8/6/15
    2015-08-06 00:00:00
    6/12/15
    2015-06-12 00:00:00
    5/8/15
    2015-05-08 00:00:00
    12/19/15
    2015-12-19 00:00:00
    3/15/15
    2015-03-15 00:00:00
    8/24/15
    2015-08-24 00:00:00
    5/8/15
    2015-05-08 00:00:00
    8/15/15
    2015-08-15 00:00:00
    12/20/15
    2015-12-20 00:00:00
    11/6/15
    2015-11-06 00:00:00
    2/26/15
    2015-02-26 00:00:00
    9/24/15
    2015-09-24 00:00:00
    3/15/15
    2015-03-15 00:00:00
    7/31/15
    2015-07-31 00:00:00
    2/3/15
    2015-02-03 00:00:00
    12/11/15
    2015-12-11 00:00:00
    9/8/15
    2015-09-08 00:00:00
    2/8/15
    2015-02-08 00:00:00
    1/24/15
    2015-01-24 00:00:00
    8/1/15
    2015-08-01 00:00:00
    6/19/15
    2015-06-19 00:00:00
    8/29/15
    2015-08-29 00:00:00
    7/24/15
    2015-07-24 00:00:00
    3/17/15
    2015-03-17 00:00:00
    9/30/15
    2015-09-30 00:00:00
    9/7/15
    2015-09-07 00:00:00
    11/5/15
    2015-11-05 00:00:00
    5/29/15
    2015-05-29 00:00:00
    9/1/15
    2015-09-01 00:00:00
    9/11/15
    2015-09-11 00:00:00
    5/6/15
    2015-05-06 00:00:00
    5/4/15
    2015-05-04 00:00:00
    12/11/15
    2015-12-11 00:00:00
    11/6/15
    2015-11-06 00:00:00
    1/9/15
    2015-01-09 00:00:00
    10/16/15
    2015-10-16 00:00:00
    10/16/15
    2015-10-16 00:00:00
    7/6/15
    2015-07-06 00:00:00
    9/1/15
    2015-09-01 00:00:00
    4/5/15
    2015-04-05 00:00:00
    3/16/15
    2015-03-16 00:00:00
    3/13/15
    2015-03-13 00:00:00
    1/15/15
    2015-01-15 00:00:00
    11/13/15
    2015-11-13 00:00:00
    10/23/15
    2015-10-23 00:00:00
    8/28/15
    2015-08-28 00:00:00
    6/20/15
    2015-06-20 00:00:00
    10/4/15
    2015-10-04 00:00:00
    8/13/15
    2015-08-13 00:00:00
    9/9/15
    2015-09-09 00:00:00
    10/22/15
    2015-10-22 00:00:00
    6/10/15
    2015-06-10 00:00:00
    10/13/15
    2015-10-13 00:00:00
    10/17/15
    2015-10-17 00:00:00
    4/17/15
    2015-04-17 00:00:00
    1/17/15
    2015-01-17 00:00:00
    5/29/15
    2015-05-29 00:00:00
    9/18/15
    2015-09-18 00:00:00
    3/3/15
    2015-03-03 00:00:00
    9/25/15
    2015-09-25 00:00:00
    8/30/15
    2015-08-30 00:00:00
    10/16/15
    2015-10-16 00:00:00
    4/1/15
    2015-04-01 00:00:00
    1/9/15
    2015-01-09 00:00:00
    8/18/15
    2015-08-18 00:00:00
    7/27/15
    2015-07-27 00:00:00
    4/15/15
    2015-04-15 00:00:00
    10/16/15
    2015-10-16 00:00:00
    12/23/15
    2015-12-23 00:00:00
    6/5/15
    2015-06-05 00:00:00
    7/10/15
    2015-07-10 00:00:00
    3/27/15
    2015-03-27 00:00:00
    1/22/15
    2015-01-22 00:00:00
    2/24/15
    2015-02-24 00:00:00
    3/14/15
    2015-03-14 00:00:00
    4/24/15
    2015-04-24 00:00:00
    5/29/15
    2015-05-29 00:00:00
    2/20/15
    2015-02-20 00:00:00
    10/9/15
    2015-10-09 00:00:00
    10/7/15
    2015-10-07 00:00:00
    6/22/15
    2015-06-22 00:00:00
    12/5/15
    2015-12-05 00:00:00
    8/30/15
    2015-08-30 00:00:00
    3/27/15
    2015-03-27 00:00:00
    5/1/15
    2015-05-01 00:00:00
    1/23/15
    2015-01-23 00:00:00
    3/27/15
    2015-03-27 00:00:00
    1/23/15
    2015-01-23 00:00:00
    9/4/15
    2015-09-04 00:00:00
    10/8/15
    2015-10-08 00:00:00
    8/14/15
    2015-08-14 00:00:00
    1/1/15
    2015-01-01 00:00:00
    6/9/15
    2015-06-09 00:00:00
    12/26/15
    2015-12-26 00:00:00
    9/4/15
    2015-09-04 00:00:00
    3/15/15
    2015-03-15 00:00:00
    3/30/15
    2015-03-30 00:00:00
    2/6/15
    2015-02-06 00:00:00
    7/29/15
    2015-07-29 00:00:00
    1/13/15
    2015-01-13 00:00:00
    4/3/15
    2015-04-03 00:00:00
    8/10/15
    2015-08-10 00:00:00
    8/14/15
    2015-08-14 00:00:00
    10/15/15
    2015-10-15 00:00:00
    3/27/15
    2015-03-27 00:00:00
    10/10/15
    2015-10-10 00:00:00
    12/25/15
    2015-12-25 00:00:00
    8/7/15
    2015-08-07 00:00:00
    8/24/15
    2015-08-24 00:00:00
    1/24/15
    2015-01-24 00:00:00
    8/14/15
    2015-08-14 00:00:00
    7/21/15
    2015-07-21 00:00:00
    12/14/15
    2015-12-14 00:00:00
    2/20/15
    2015-02-20 00:00:00
    9/18/15
    2015-09-18 00:00:00
    6/5/15
    2015-06-05 00:00:00
    10/17/15
    2015-10-17 00:00:00
    10/9/15
    2015-10-09 00:00:00
    7/23/15
    2015-07-23 00:00:00
    3/17/15
    2015-03-17 00:00:00
    7/10/15
    2015-07-10 00:00:00
    2/24/15
    2015-02-24 00:00:00
    12/27/15
    2015-12-27 00:00:00
    4/24/15
    2015-04-24 00:00:00
    9/25/15
    2015-09-25 00:00:00
    10/27/15
    2015-10-27 00:00:00
    1/27/15
    2015-01-27 00:00:00
    6/12/15
    2015-06-12 00:00:00
    3/26/15
    2015-03-26 00:00:00
    3/6/15
    2015-03-06 00:00:00
    4/16/15
    2015-04-16 00:00:00
    4/18/15
    2015-04-18 00:00:00
    2/6/15
    2015-02-06 00:00:00
    2/28/15
    2015-02-28 00:00:00
    8/19/15
    2015-08-19 00:00:00
    9/9/15
    2015-09-09 00:00:00
    9/18/15
    2015-09-18 00:00:00
    6/16/15
    2015-06-16 00:00:00
    7/5/15
    2015-07-05 00:00:00
    10/16/15
    2015-10-16 00:00:00
    2/27/15
    2015-02-27 00:00:00
    8/1/15
    2015-08-01 00:00:00
    4/23/15
    2015-04-23 00:00:00
    11/15/15
    2015-11-15 00:00:00
    11/27/15
    2015-11-27 00:00:00
    7/31/15
    2015-07-31 00:00:00
    6/8/15
    2015-06-08 00:00:00
    11/7/15
    2015-11-07 00:00:00
    1/1/15
    2015-01-01 00:00:00
    7/31/15
    2015-07-31 00:00:00
    6/12/15
    2015-06-12 00:00:00
    8/19/15
    2015-08-19 00:00:00
    6/12/15
    2015-06-12 00:00:00
    3/24/15
    2015-03-24 00:00:00
    4/14/15
    2015-04-14 00:00:00
    8/21/15
    2015-08-21 00:00:00
    9/3/15
    2015-09-03 00:00:00
    10/8/15
    2015-10-08 00:00:00
    3/20/15
    2015-03-20 00:00:00
    8/12/15
    2015-08-12 00:00:00
    7/15/15
    2015-07-15 00:00:00
    7/17/15
    2015-07-17 00:00:00
    2/8/15
    2015-02-08 00:00:00
    8/21/15
    2015-08-21 00:00:00
    3/31/15
    2015-03-31 00:00:00
    3/13/15
    2015-03-13 00:00:00
    4/1/15
    2015-04-01 00:00:00
    7/10/15
    2015-07-10 00:00:00
    3/21/15
    2015-03-21 00:00:00
    11/10/15
    2015-11-10 00:00:00
    3/6/15
    2015-03-06 00:00:00
    9/15/15
    2015-09-15 00:00:00
    3/20/15
    2015-03-20 00:00:00
    10/6/15
    2015-10-06 00:00:00
    4/7/15
    2015-04-07 00:00:00
    10/14/15
    2015-10-14 00:00:00
    9/27/15
    2015-09-27 00:00:00
    8/14/15
    2015-08-14 00:00:00
    11/13/15
    2015-11-13 00:00:00
    9/11/15
    2015-09-11 00:00:00
    3/15/15
    2015-03-15 00:00:00
    4/24/15
    2015-04-24 00:00:00
    11/5/14
    2014-11-05 00:00:00
    7/30/14
    2014-07-30 00:00:00
    3/20/14
    2014-03-20 00:00:00
    10/22/14
    2014-10-22 00:00:00
    11/18/14
    2014-11-18 00:00:00
    12/10/14
    2014-12-10 00:00:00
    10/24/14
    2014-10-24 00:00:00
    11/14/14
    2014-11-14 00:00:00
    9/10/14
    2014-09-10 00:00:00
    10/17/14
    2014-10-17 00:00:00
    10/1/14
    2014-10-01 00:00:00
    10/15/14
    2014-10-15 00:00:00
    12/17/14
    2014-12-17 00:00:00
    3/14/14
    2014-03-14 00:00:00
    5/15/14
    2014-05-15 00:00:00
    7/14/14
    2014-07-14 00:00:00
    8/7/14
    2014-08-07 00:00:00
    7/17/14
    2014-07-17 00:00:00
    10/23/14
    2014-10-23 00:00:00
    6/25/14
    2014-06-25 00:00:00
    2/26/14
    2014-02-26 00:00:00
    10/10/14
    2014-10-10 00:00:00
    6/26/14
    2014-06-26 00:00:00
    9/24/14
    2014-09-24 00:00:00
    11/26/14
    2014-11-26 00:00:00
    4/16/14
    2014-04-16 00:00:00
    5/27/14
    2014-05-27 00:00:00
    12/25/14
    2014-12-25 00:00:00
    12/11/14
    2014-12-11 00:00:00
    12/3/14
    2014-12-03 00:00:00
    5/14/14
    2014-05-14 00:00:00
    5/28/14
    2014-05-28 00:00:00
    8/27/14
    2014-08-27 00:00:00
    10/24/14
    2014-10-24 00:00:00
    6/12/14
    2014-06-12 00:00:00
    8/27/14
    2014-08-27 00:00:00
    8/4/14
    2014-08-04 00:00:00
    2/6/14
    2014-02-06 00:00:00
    10/14/14
    2014-10-14 00:00:00
    12/12/14
    2014-12-12 00:00:00
    11/26/14
    2014-11-26 00:00:00
    8/13/14
    2014-08-13 00:00:00
    10/1/14
    2014-10-01 00:00:00
    11/22/14
    2014-11-22 00:00:00
    12/25/14
    2014-12-25 00:00:00
    3/5/14
    2014-03-05 00:00:00
    8/11/14
    2014-08-11 00:00:00
    6/5/14
    2014-06-05 00:00:00
    9/18/14
    2014-09-18 00:00:00
    11/27/14
    2014-11-27 00:00:00
    9/7/14
    2014-09-07 00:00:00
    5/8/14
    2014-05-08 00:00:00
    7/17/14
    2014-07-17 00:00:00
    10/3/14
    2014-10-03 00:00:00
    6/5/14
    2014-06-05 00:00:00
    12/30/14
    2014-12-30 00:00:00
    7/23/14
    2014-07-23 00:00:00
    3/20/14
    2014-03-20 00:00:00
    10/8/14
    2014-10-08 00:00:00
    5/26/14
    2014-05-26 00:00:00
    5/21/14
    2014-05-21 00:00:00
    5/8/14
    2014-05-08 00:00:00
    8/21/14
    2014-08-21 00:00:00
    7/16/14
    2014-07-16 00:00:00
    7/2/14
    2014-07-02 00:00:00
    5/16/14
    2014-05-16 00:00:00
    7/18/14
    2014-07-18 00:00:00
    1/17/14
    2014-01-17 00:00:00
    6/10/14
    2014-06-10 00:00:00
    12/25/14
    2014-12-25 00:00:00
    1/15/14
    2014-01-15 00:00:00
    1/26/14
    2014-01-26 00:00:00
    8/28/14
    2014-08-28 00:00:00
    10/9/14
    2014-10-09 00:00:00
    12/8/14
    2014-12-08 00:00:00
    9/12/14
    2014-09-12 00:00:00
    1/30/14
    2014-01-30 00:00:00
    4/16/14
    2014-04-16 00:00:00
    4/16/14
    2014-04-16 00:00:00
    2/7/14
    2014-02-07 00:00:00
    11/7/14
    2014-11-07 00:00:00
    8/20/14
    2014-08-20 00:00:00
    12/12/14
    2014-12-12 00:00:00
    3/19/14
    2014-03-19 00:00:00
    11/12/14
    2014-11-12 00:00:00
    9/3/14
    2014-09-03 00:00:00
    1/24/14
    2014-01-24 00:00:00
    12/25/14
    2014-12-25 00:00:00
    12/25/14
    2014-12-25 00:00:00
    5/22/14
    2014-05-22 00:00:00
    10/11/14
    2014-10-11 00:00:00
    2/13/14
    2014-02-13 00:00:00
    12/5/14
    2014-12-05 00:00:00
    2/18/14
    2014-02-18 00:00:00
    10/16/14
    2014-10-16 00:00:00
    12/19/14
    2014-12-19 00:00:00
    9/11/14
    2014-09-11 00:00:00
    10/1/14
    2014-10-01 00:00:00
    10/9/14
    2014-10-09 00:00:00
    3/13/14
    2014-03-13 00:00:00
    9/26/14
    2014-09-26 00:00:00
    9/26/14
    2014-09-26 00:00:00
    8/6/14
    2014-08-06 00:00:00
    9/10/14
    2014-09-10 00:00:00
    12/24/14
    2014-12-24 00:00:00
    8/22/14
    2014-08-22 00:00:00
    12/5/14
    2014-12-05 00:00:00
    9/6/14
    2014-09-06 00:00:00
    2/4/14
    2014-02-04 00:00:00
    3/15/14
    2014-03-15 00:00:00
    8/14/14
    2014-08-14 00:00:00
    8/6/14
    2014-08-06 00:00:00
    9/26/14
    2014-09-26 00:00:00
    2/14/14
    2014-02-14 00:00:00
    7/17/14
    2014-07-17 00:00:00
    10/10/14
    2014-10-10 00:00:00
    3/28/14
    2014-03-28 00:00:00
    5/22/14
    2014-05-22 00:00:00
    9/10/14
    2014-09-10 00:00:00
    5/2/14
    2014-05-02 00:00:00
    10/17/14
    2014-10-17 00:00:00
    9/12/14
    2014-09-12 00:00:00
    12/5/14
    2014-12-05 00:00:00
    8/14/14
    2014-08-14 00:00:00
    5/19/14
    2014-05-19 00:00:00
    9/26/14
    2014-09-26 00:00:00
    2/12/14
    2014-02-12 00:00:00
    5/18/14
    2014-05-18 00:00:00
    6/19/14
    2014-06-19 00:00:00
    11/5/14
    2014-11-05 00:00:00
    5/16/14
    2014-05-16 00:00:00
    5/2/14
    2014-05-02 00:00:00
    3/14/14
    2014-03-14 00:00:00
    10/2/14
    2014-10-02 00:00:00
    7/5/14
    2014-07-05 00:00:00
    9/11/14
    2014-09-11 00:00:00
    1/22/14
    2014-01-22 00:00:00
    1/29/14
    2014-01-29 00:00:00
    5/9/14
    2014-05-09 00:00:00
    4/11/14
    2014-04-11 00:00:00
    4/22/14
    2014-04-22 00:00:00
    5/22/14
    2014-05-22 00:00:00
    7/2/14
    2014-07-02 00:00:00
    9/4/14
    2014-09-04 00:00:00
    10/12/14
    2014-10-12 00:00:00
    10/14/14
    2014-10-14 00:00:00
    9/11/14
    2014-09-11 00:00:00
    2/25/14
    2014-02-25 00:00:00
    9/12/14
    2014-09-12 00:00:00
    3/27/14
    2014-03-27 00:00:00
    8/5/14
    2014-08-05 00:00:00
    2/13/14
    2014-02-13 00:00:00
    9/12/14
    2014-09-12 00:00:00
    9/25/14
    2014-09-25 00:00:00
    9/17/14
    2014-09-17 00:00:00
    7/31/14
    2014-07-31 00:00:00
    9/11/14
    2014-09-11 00:00:00
    4/25/14
    2014-04-25 00:00:00
    11/6/14
    2014-11-06 00:00:00
    12/25/14
    2014-12-25 00:00:00
    9/4/14
    2014-09-04 00:00:00
    7/25/14
    2014-07-25 00:00:00
    6/13/14
    2014-06-13 00:00:00
    10/8/14
    2014-10-08 00:00:00
    9/10/14
    2014-09-10 00:00:00
    5/18/14
    2014-05-18 00:00:00
    10/16/14
    2014-10-16 00:00:00
    5/23/14
    2014-05-23 00:00:00
    4/4/14
    2014-04-04 00:00:00
    1/17/14
    2014-01-17 00:00:00
    11/7/14
    2014-11-07 00:00:00
    7/4/14
    2014-07-04 00:00:00
    9/5/14
    2014-09-05 00:00:00
    2/28/14
    2014-02-28 00:00:00
    6/5/14
    2014-06-05 00:00:00
    2/14/14
    2014-02-14 00:00:00
    5/9/14
    2014-05-09 00:00:00
    1/18/14
    2014-01-18 00:00:00
    1/18/14
    2014-01-18 00:00:00
    9/3/14
    2014-09-03 00:00:00
    9/8/14
    2014-09-08 00:00:00
    8/16/14
    2014-08-16 00:00:00
    9/11/14
    2014-09-11 00:00:00
    3/14/14
    2014-03-14 00:00:00
    3/13/14
    2014-03-13 00:00:00
    8/14/14
    2014-08-14 00:00:00
    2/28/14
    2014-02-28 00:00:00
    5/9/14
    2014-05-09 00:00:00
    8/12/14
    2014-08-12 00:00:00
    2/2/14
    2014-02-02 00:00:00
    2/2/14
    2014-02-02 00:00:00
    1/2/14
    2014-01-02 00:00:00
    9/4/14
    2014-09-04 00:00:00
    10/3/14
    2014-10-03 00:00:00
    8/22/14
    2014-08-22 00:00:00
    3/20/14
    2014-03-20 00:00:00
    12/2/14
    2014-12-02 00:00:00
    11/7/14
    2014-11-07 00:00:00
    9/1/14
    2014-09-01 00:00:00
    9/19/14
    2014-09-19 00:00:00
    9/19/14
    2014-09-19 00:00:00
    5/14/14
    2014-05-14 00:00:00
    5/2/14
    2014-05-02 00:00:00
    6/13/14
    2014-06-13 00:00:00
    4/20/14
    2014-04-20 00:00:00
    4/8/14
    2014-04-08 00:00:00
    2/14/14
    2014-02-14 00:00:00
    12/4/14
    2014-12-04 00:00:00
    4/1/14
    2014-04-01 00:00:00
    5/9/14
    2014-05-09 00:00:00
    7/25/14
    2014-07-25 00:00:00
    12/12/14
    2014-12-12 00:00:00
    8/1/14
    2014-08-01 00:00:00
    12/4/14
    2014-12-04 00:00:00
    7/1/14
    2014-07-01 00:00:00
    8/20/14
    2014-08-20 00:00:00
    10/9/14
    2014-10-09 00:00:00
    4/11/14
    2014-04-11 00:00:00
    5/24/14
    2014-05-24 00:00:00
    10/27/14
    2014-10-27 00:00:00
    6/19/14
    2014-06-19 00:00:00
    9/10/14
    2014-09-10 00:00:00
    9/12/14
    2014-09-12 00:00:00
    3/21/14
    2014-03-21 00:00:00
    9/19/14
    2014-09-19 00:00:00
    4/10/14
    2014-04-10 00:00:00
    7/5/14
    2014-07-05 00:00:00
    9/12/14
    2014-09-12 00:00:00
    9/5/14
    2014-09-05 00:00:00
    3/11/14
    2014-03-11 00:00:00
    5/23/14
    2014-05-23 00:00:00
    6/13/14
    2014-06-13 00:00:00
    12/30/14
    2014-12-30 00:00:00
    5/9/14
    2014-05-09 00:00:00
    8/29/14
    2014-08-29 00:00:00
    3/20/14
    2014-03-20 00:00:00
    10/10/14
    2014-10-10 00:00:00
    8/22/14
    2014-08-22 00:00:00
    10/30/14
    2014-10-30 00:00:00
    2/28/14
    2014-02-28 00:00:00
    10/21/14
    2014-10-21 00:00:00
    9/19/14
    2014-09-19 00:00:00
    10/17/14
    2014-10-17 00:00:00
    9/12/14
    2014-09-12 00:00:00
    11/22/14
    2014-11-22 00:00:00
    5/16/14
    2014-05-16 00:00:00
    9/26/14
    2014-09-26 00:00:00
    6/19/14
    2014-06-19 00:00:00
    10/14/14
    2014-10-14 00:00:00
    9/5/14
    2014-09-05 00:00:00
    4/17/14
    2014-04-17 00:00:00
    6/5/14
    2014-06-05 00:00:00
    4/16/14
    2014-04-16 00:00:00
    3/14/14
    2014-03-14 00:00:00
    4/4/14
    2014-04-04 00:00:00
    1/25/14
    2014-01-25 00:00:00
    10/9/14
    2014-10-09 00:00:00
    4/23/14
    2014-04-23 00:00:00
    9/7/14
    2014-09-07 00:00:00
    8/6/14
    2014-08-06 00:00:00
    8/30/14
    2014-08-30 00:00:00
    9/7/14
    2014-09-07 00:00:00
    10/10/14
    2014-10-10 00:00:00
    9/13/14
    2014-09-13 00:00:00
    4/17/14
    2014-04-17 00:00:00
    8/22/14
    2014-08-22 00:00:00
    10/16/14
    2014-10-16 00:00:00
    1/10/14
    2014-01-10 00:00:00
    5/21/14
    2014-05-21 00:00:00
    10/20/14
    2014-10-20 00:00:00
    4/18/14
    2014-04-18 00:00:00
    8/25/14
    2014-08-25 00:00:00
    7/20/14
    2014-07-20 00:00:00
    10/31/14
    2014-10-31 00:00:00
    5/2/14
    2014-05-02 00:00:00
    9/27/14
    2014-09-27 00:00:00
    8/28/14
    2014-08-28 00:00:00
    12/30/14
    2014-12-30 00:00:00
    9/4/14
    2014-09-04 00:00:00
    2/7/14
    2014-02-07 00:00:00
    5/14/14
    2014-05-14 00:00:00
    8/15/14
    2014-08-15 00:00:00
    8/27/14
    2014-08-27 00:00:00
    7/25/14
    2014-07-25 00:00:00
    1/17/14
    2014-01-17 00:00:00
    7/2/14
    2014-07-02 00:00:00
    9/9/14
    2014-09-09 00:00:00
    3/10/14
    2014-03-10 00:00:00
    10/16/14
    2014-10-16 00:00:00
    3/10/14
    2014-03-10 00:00:00
    10/2/14
    2014-10-02 00:00:00
    7/30/14
    2014-07-30 00:00:00
    2/7/14
    2014-02-07 00:00:00
    6/19/14
    2014-06-19 00:00:00
    4/4/14
    2014-04-04 00:00:00
    9/7/14
    2014-09-07 00:00:00
    4/2/14
    2014-04-02 00:00:00
    1/10/14
    2014-01-10 00:00:00
    4/14/14
    2014-04-14 00:00:00
    8/22/14
    2014-08-22 00:00:00
    9/5/14
    2014-09-05 00:00:00
    9/19/14
    2014-09-19 00:00:00
    4/24/14
    2014-04-24 00:00:00
    10/10/14
    2014-10-10 00:00:00
    9/11/14
    2014-09-11 00:00:00
    12/5/14
    2014-12-05 00:00:00
    1/17/14
    2014-01-17 00:00:00
    8/5/14
    2014-08-05 00:00:00
    1/17/14
    2014-01-17 00:00:00
    9/6/14
    2014-09-06 00:00:00
    3/14/14
    2014-03-14 00:00:00
    1/20/14
    2014-01-20 00:00:00
    7/19/14
    2014-07-19 00:00:00
    1/19/14
    2014-01-19 00:00:00
    10/11/14
    2014-10-11 00:00:00
    4/4/14
    2014-04-04 00:00:00
    2/25/14
    2014-02-25 00:00:00
    10/10/14
    2014-10-10 00:00:00
    9/10/14
    2014-09-10 00:00:00
    8/8/14
    2014-08-08 00:00:00
    11/21/14
    2014-11-21 00:00:00
    3/20/14
    2014-03-20 00:00:00
    9/19/14
    2014-09-19 00:00:00
    4/30/14
    2014-04-30 00:00:00
    9/5/14
    2014-09-05 00:00:00
    3/25/14
    2014-03-25 00:00:00
    4/20/14
    2014-04-20 00:00:00
    7/2/14
    2014-07-02 00:00:00
    4/20/14
    2014-04-20 00:00:00
    9/5/14
    2014-09-05 00:00:00
    6/20/14
    2014-06-20 00:00:00
    5/13/14
    2014-05-13 00:00:00
    8/22/14
    2014-08-22 00:00:00
    3/4/14
    2014-03-04 00:00:00
    1/1/14
    2014-01-01 00:00:00
    6/28/14
    2014-06-28 00:00:00
    10/10/14
    2014-10-10 00:00:00
    10/18/14
    2014-10-18 00:00:00
    1/3/14
    2014-01-03 00:00:00
    6/6/14
    2014-06-06 00:00:00
    9/27/14
    2014-09-27 00:00:00
    3/12/14
    2014-03-12 00:00:00
    4/3/14
    2014-04-03 00:00:00
    10/3/14
    2014-10-03 00:00:00
    4/18/14
    2014-04-18 00:00:00
    9/19/14
    2014-09-19 00:00:00
    1/31/14
    2014-01-31 00:00:00
    12/25/14
    2014-12-25 00:00:00
    5/27/14
    2014-05-27 00:00:00
    11/11/14
    2014-11-11 00:00:00
    9/28/14
    2014-09-28 00:00:00
    3/26/14
    2014-03-26 00:00:00
    4/30/14
    2014-04-30 00:00:00
    4/12/14
    2014-04-12 00:00:00
    5/30/14
    2014-05-30 00:00:00
    10/6/14
    2014-10-06 00:00:00
    4/10/14
    2014-04-10 00:00:00
    12/23/14
    2014-12-23 00:00:00
    10/17/14
    2014-10-17 00:00:00
    2/19/14
    2014-02-19 00:00:00
    11/2/14
    2014-11-02 00:00:00
    7/29/14
    2014-07-29 00:00:00
    8/29/14
    2014-08-29 00:00:00
    12/31/14
    2014-12-31 00:00:00
    5/16/14
    2014-05-16 00:00:00
    10/17/14
    2014-10-17 00:00:00
    4/19/14
    2014-04-19 00:00:00
    7/21/14
    2014-07-21 00:00:00
    10/3/14
    2014-10-03 00:00:00
    7/31/14
    2014-07-31 00:00:00
    2/7/14
    2014-02-07 00:00:00
    11/3/14
    2014-11-03 00:00:00
    1/18/14
    2014-01-18 00:00:00
    6/28/14
    2014-06-28 00:00:00
    9/26/14
    2014-09-26 00:00:00
    3/25/14
    2014-03-25 00:00:00
    11/3/14
    2014-11-03 00:00:00
    3/14/14
    2014-03-14 00:00:00
    2/5/14
    2014-02-05 00:00:00
    8/15/14
    2014-08-15 00:00:00
    10/10/14
    2014-10-10 00:00:00
    10/10/14
    2014-10-10 00:00:00
    10/22/14
    2014-10-22 00:00:00
    8/22/14
    2014-08-22 00:00:00
    2/7/14
    2014-02-07 00:00:00
    9/15/14
    2014-09-15 00:00:00
    4/18/14
    2014-04-18 00:00:00
    4/20/14
    2014-04-20 00:00:00
    3/7/14
    2014-03-07 00:00:00
    6/15/14
    2014-06-15 00:00:00
    3/21/14
    2014-03-21 00:00:00
    11/8/14
    2014-11-08 00:00:00
    7/17/14
    2014-07-17 00:00:00
    9/19/14
    2014-09-19 00:00:00
    3/21/14
    2014-03-21 00:00:00
    11/22/14
    2014-11-22 00:00:00
    7/31/14
    2014-07-31 00:00:00
    2/14/14
    2014-02-14 00:00:00
    9/18/14
    2014-09-18 00:00:00
    7/11/14
    2014-07-11 00:00:00
    11/4/14
    2014-11-04 00:00:00
    9/29/14
    2014-09-29 00:00:00
    9/7/14
    2014-09-07 00:00:00
    3/18/14
    2014-03-18 00:00:00
    10/23/14
    2014-10-23 00:00:00
    12/5/14
    2014-12-05 00:00:00
    8/8/14
    2014-08-08 00:00:00
    10/28/14
    2014-10-28 00:00:00
    10/10/14
    2014-10-10 00:00:00
    6/18/14
    2014-06-18 00:00:00
    2/21/14
    2014-02-21 00:00:00
    5/25/14
    2014-05-25 00:00:00
    3/6/14
    2014-03-06 00:00:00
    1/8/14
    2014-01-08 00:00:00
    8/6/14
    2014-08-06 00:00:00
    4/20/14
    2014-04-20 00:00:00
    9/25/14
    2014-09-25 00:00:00
    9/1/14
    2014-09-01 00:00:00
    12/5/14
    2014-12-05 00:00:00
    4/17/14
    2014-04-17 00:00:00
    4/24/14
    2014-04-24 00:00:00
    6/27/14
    2014-06-27 00:00:00
    3/15/14
    2014-03-15 00:00:00
    8/14/14
    2014-08-14 00:00:00
    11/12/14
    2014-11-12 00:00:00
    1/19/14
    2014-01-19 00:00:00
    12/12/14
    2014-12-12 00:00:00
    4/18/14
    2014-04-18 00:00:00
    10/17/14
    2014-10-17 00:00:00
    7/11/14
    2014-07-11 00:00:00
    8/28/14
    2014-08-28 00:00:00
    5/28/14
    2014-05-28 00:00:00
    8/6/14
    2014-08-06 00:00:00
    3/8/14
    2014-03-08 00:00:00
    6/3/14
    2014-06-03 00:00:00
    2/4/14
    2014-02-04 00:00:00
    3/25/14
    2014-03-25 00:00:00
    12/4/14
    2014-12-04 00:00:00
    7/18/14
    2014-07-18 00:00:00
    3/17/14
    2014-03-17 00:00:00
    9/9/14
    2014-09-09 00:00:00
    10/10/14
    2014-10-10 00:00:00
    6/5/14
    2014-06-05 00:00:00
    6/10/14
    2014-06-10 00:00:00
    10/17/14
    2014-10-17 00:00:00
    10/3/14
    2014-10-03 00:00:00
    10/2/14
    2014-10-02 00:00:00
    8/12/14
    2014-08-12 00:00:00
    9/1/14
    2014-09-01 00:00:00
    10/26/14
    2014-10-26 00:00:00
    3/26/14
    2014-03-26 00:00:00
    7/4/14
    2014-07-04 00:00:00
    12/5/14
    2014-12-05 00:00:00
    10/7/14
    2014-10-07 00:00:00
    11/21/14
    2014-11-21 00:00:00
    5/16/14
    2014-05-16 00:00:00
    10/7/14
    2014-10-07 00:00:00
    8/19/14
    2014-08-19 00:00:00
    9/5/14
    2014-09-05 00:00:00
    3/28/14
    2014-03-28 00:00:00
    4/6/14
    2014-04-06 00:00:00
    6/12/14
    2014-06-12 00:00:00
    8/15/14
    2014-08-15 00:00:00
    3/1/14
    2014-03-01 00:00:00
    3/28/14
    2014-03-28 00:00:00
    1/22/14
    2014-01-22 00:00:00
    6/6/14
    2014-06-06 00:00:00
    6/27/14
    2014-06-27 00:00:00
    2/1/14
    2014-02-01 00:00:00
    3/8/14
    2014-03-08 00:00:00
    7/3/14
    2014-07-03 00:00:00
    5/26/14
    2014-05-26 00:00:00
    7/29/14
    2014-07-29 00:00:00
    8/22/14
    2014-08-22 00:00:00
    7/25/14
    2014-07-25 00:00:00
    11/7/14
    2014-11-07 00:00:00
    11/5/14
    2014-11-05 00:00:00
    9/30/14
    2014-09-30 00:00:00
    4/29/14
    2014-04-29 00:00:00
    1/29/14
    2014-01-29 00:00:00
    4/4/14
    2014-04-04 00:00:00
    7/3/14
    2014-07-03 00:00:00
    11/21/14
    2014-11-21 00:00:00
    4/9/14
    2014-04-09 00:00:00
    3/15/14
    2014-03-15 00:00:00
    3/20/14
    2014-03-20 00:00:00
    10/11/14
    2014-10-11 00:00:00
    12/20/14
    2014-12-20 00:00:00
    11/4/14
    2014-11-04 00:00:00
    5/23/14
    2014-05-23 00:00:00
    6/23/14
    2014-06-23 00:00:00
    9/5/14
    2014-09-05 00:00:00
    9/12/14
    2014-09-12 00:00:00
    6/12/14
    2014-06-12 00:00:00
    8/30/14
    2014-08-30 00:00:00
    2/7/14
    2014-02-07 00:00:00
    10/9/14
    2014-10-09 00:00:00
    5/14/14
    2014-05-14 00:00:00
    9/28/14
    2014-09-28 00:00:00
    6/4/14
    2014-06-04 00:00:00
    6/21/14
    2014-06-21 00:00:00
    8/17/14
    2014-08-17 00:00:00
    2/12/14
    2014-02-12 00:00:00
    8/14/14
    2014-08-14 00:00:00
    7/22/14
    2014-07-22 00:00:00
    3/1/14
    2014-03-01 00:00:00
    4/19/14
    2014-04-19 00:00:00
    7/19/14
    2014-07-19 00:00:00
    5/17/14
    2014-05-17 00:00:00
    4/18/14
    2014-04-18 00:00:00
    6/30/14
    2014-06-30 00:00:00
    1/21/14
    2014-01-21 00:00:00
    3/25/14
    2014-03-25 00:00:00
    8/7/14
    2014-08-07 00:00:00
    3/17/14
    2014-03-17 00:00:00
    7/3/14
    2014-07-03 00:00:00
    10/20/14
    2014-10-20 00:00:00
    5/20/14
    2014-05-20 00:00:00
    12/17/14
    2014-12-17 00:00:00
    10/17/14
    2014-10-17 00:00:00
    12/25/14
    2014-12-25 00:00:00
    8/7/14
    2014-08-07 00:00:00
    10/31/14
    2014-10-31 00:00:00
    7/2/14
    2014-07-02 00:00:00
    9/1/14
    2014-09-01 00:00:00
    12/11/14
    2014-12-11 00:00:00
    4/20/14
    2014-04-20 00:00:00
    10/14/14
    2014-10-14 00:00:00
    4/8/14
    2014-04-08 00:00:00
    3/14/14
    2014-03-14 00:00:00
    6/3/14
    2014-06-03 00:00:00
    10/17/14
    2014-10-17 00:00:00
    7/8/14
    2014-07-08 00:00:00
    9/23/14
    2014-09-23 00:00:00
    5/23/14
    2014-05-23 00:00:00
    8/29/14
    2014-08-29 00:00:00
    12/16/14
    2014-12-16 00:00:00
    6/6/14
    2014-06-06 00:00:00
    3/9/14
    2014-03-09 00:00:00
    9/29/14
    2014-09-29 00:00:00
    3/7/14
    2014-03-07 00:00:00
    8/19/14
    2014-08-19 00:00:00
    8/25/14
    2014-08-25 00:00:00
    2/18/14
    2014-02-18 00:00:00
    4/20/14
    2014-04-20 00:00:00
    3/7/14
    2014-03-07 00:00:00
    1/19/14
    2014-01-19 00:00:00
    9/6/14
    2014-09-06 00:00:00
    5/8/14
    2014-05-08 00:00:00
    8/2/14
    2014-08-02 00:00:00
    6/27/14
    2014-06-27 00:00:00
    10/17/14
    2014-10-17 00:00:00
    4/27/14
    2014-04-27 00:00:00
    10/24/14
    2014-10-24 00:00:00
    12/15/14
    2014-12-15 00:00:00
    5/2/14
    2014-05-02 00:00:00
    8/13/14
    2014-08-13 00:00:00
    8/13/14
    2014-08-13 00:00:00
    1/18/14
    2014-01-18 00:00:00
    6/26/14
    2014-06-26 00:00:00
    11/7/14
    2014-11-07 00:00:00
    9/5/14
    2014-09-05 00:00:00
    8/15/14
    2014-08-15 00:00:00
    2/6/14
    2014-02-06 00:00:00
    6/18/14
    2014-06-18 00:00:00
    10/18/14
    2014-10-18 00:00:00
    4/18/14
    2014-04-18 00:00:00
    9/4/14
    2014-09-04 00:00:00
    9/4/14
    2014-09-04 00:00:00
    3/25/14
    2014-03-25 00:00:00
    10/5/14
    2014-10-05 00:00:00
    12/16/14
    2014-12-16 00:00:00
    6/17/14
    2014-06-17 00:00:00
    10/1/14
    2014-10-01 00:00:00
    3/19/14
    2014-03-19 00:00:00
    6/27/14
    2014-06-27 00:00:00
    8/22/14
    2014-08-22 00:00:00
    9/15/14
    2014-09-15 00:00:00
    6/13/14
    2014-06-13 00:00:00
    7/1/14
    2014-07-01 00:00:00
    7/1/14
    2014-07-01 00:00:00
    5/30/14
    2014-05-30 00:00:00
    9/1/14
    2014-09-01 00:00:00
    11/29/14
    2014-11-29 00:00:00
    5/9/14
    2014-05-09 00:00:00
    2/8/14
    2014-02-08 00:00:00
    3/1/14
    2014-03-01 00:00:00
    2/4/14
    2014-02-04 00:00:00
    11/3/14
    2014-11-03 00:00:00
    1/18/14
    2014-01-18 00:00:00
    2/14/14
    2014-02-14 00:00:00
    10/16/14
    2014-10-16 00:00:00
    9/18/14
    2014-09-18 00:00:00
    6/26/14
    2014-06-26 00:00:00
    1/6/14
    2014-01-06 00:00:00
    2/28/14
    2014-02-28 00:00:00
    9/28/14
    2014-09-28 00:00:00
    6/20/14
    2014-06-20 00:00:00
    11/7/14
    2014-11-07 00:00:00
    10/10/14
    2014-10-10 00:00:00
    6/13/14
    2014-06-13 00:00:00
    9/25/14
    2014-09-25 00:00:00
    7/8/14
    2014-07-08 00:00:00
    6/11/14
    2014-06-11 00:00:00
    3/21/14
    2014-03-21 00:00:00
    9/2/14
    2014-09-02 00:00:00
    1/20/14
    2014-01-20 00:00:00
    1/20/14
    2014-01-20 00:00:00
    5/20/14
    2014-05-20 00:00:00
    2/7/14
    2014-02-07 00:00:00
    10/10/14
    2014-10-10 00:00:00
    9/26/14
    2014-09-26 00:00:00
    1/1/14
    2014-01-01 00:00:00
    8/29/14
    2014-08-29 00:00:00
    12/5/14
    2014-12-05 00:00:00
    6/24/14
    2014-06-24 00:00:00
    1/23/14
    2014-01-23 00:00:00
    12/4/14
    2014-12-04 00:00:00
    6/7/14
    2014-06-07 00:00:00
    1/20/14
    2014-01-20 00:00:00
    2/14/14
    2014-02-14 00:00:00
    10/3/14
    2014-10-03 00:00:00
    9/26/14
    2014-09-26 00:00:00
    2/15/14
    2014-02-15 00:00:00
    4/12/14
    2014-04-12 00:00:00
    1/21/14
    2014-01-21 00:00:00
    11/21/14
    2014-11-21 00:00:00
    11/17/14
    2014-11-17 00:00:00
    9/23/14
    2014-09-23 00:00:00
    11/8/14
    2014-11-08 00:00:00
    4/23/14
    2014-04-23 00:00:00
    2/14/14
    2014-02-14 00:00:00
    3/12/14
    2014-03-12 00:00:00
    10/5/14
    2014-10-05 00:00:00
    11/19/14
    2014-11-19 00:00:00
    4/20/14
    2014-04-20 00:00:00
    8/22/14
    2014-08-22 00:00:00
    5/28/14
    2014-05-28 00:00:00
    10/11/14
    2014-10-11 00:00:00
    11/19/14
    2014-11-19 00:00:00
    4/4/14
    2014-04-04 00:00:00
    10/18/14
    2014-10-18 00:00:00
    7/25/14
    2014-07-25 00:00:00
    9/19/14
    2014-09-19 00:00:00
    10/17/14
    2014-10-17 00:00:00
    8/15/14
    2014-08-15 00:00:00
    5/9/14
    2014-05-09 00:00:00
    3/21/14
    2014-03-21 00:00:00
    5/14/14
    2014-05-14 00:00:00
    5/30/14
    2014-05-30 00:00:00
    7/2/14
    2014-07-02 00:00:00
    10/24/14
    2014-10-24 00:00:00
    6/5/14
    2014-06-05 00:00:00
    10/30/14
    2014-10-30 00:00:00
    10/29/14
    2014-10-29 00:00:00
    4/27/14
    2014-04-27 00:00:00
    10/10/14
    2014-10-10 00:00:00
    8/22/14
    2014-08-22 00:00:00
    4/4/14
    2014-04-04 00:00:00
    11/17/14
    2014-11-17 00:00:00
    10/20/14
    2014-10-20 00:00:00
    7/25/14
    2014-07-25 00:00:00
    9/6/14
    2014-09-06 00:00:00
    11/11/14
    2014-11-11 00:00:00
    10/30/14
    2014-10-30 00:00:00
    3/6/14
    2014-03-06 00:00:00
    4/19/14
    2014-04-19 00:00:00
    1/17/14
    2014-01-17 00:00:00
    4/19/14
    2014-04-19 00:00:00
    11/14/14
    2014-11-14 00:00:00
    2/6/14
    2014-02-06 00:00:00
    4/24/14
    2014-04-24 00:00:00
    9/4/14
    2014-09-04 00:00:00
    9/5/14
    2014-09-05 00:00:00
    1/16/14
    2014-01-16 00:00:00
    10/24/14
    2014-10-24 00:00:00
    12/3/14
    2014-12-03 00:00:00
    10/10/14
    2014-10-10 00:00:00
    12/27/14
    2014-12-27 00:00:00
    11/21/14
    2014-11-21 00:00:00
    9/18/14
    2014-09-18 00:00:00
    9/18/14
    2014-09-18 00:00:00
    1/17/14
    2014-01-17 00:00:00
    10/5/14
    2014-10-05 00:00:00
    8/20/14
    2014-08-20 00:00:00
    4/4/14
    2014-04-04 00:00:00
    11/14/14
    2014-11-14 00:00:00
    2/7/14
    2014-02-07 00:00:00
    8/24/14
    2014-08-24 00:00:00
    12/14/14
    2014-12-14 00:00:00
    4/6/14
    2014-04-06 00:00:00
    12/12/14
    2014-12-12 00:00:00
    9/12/14
    2014-09-12 00:00:00
    10/8/14
    2014-10-08 00:00:00
    5/27/14
    2014-05-27 00:00:00
    7/11/14
    2014-07-11 00:00:00
    7/28/14
    2014-07-28 00:00:00
    1/19/14
    2014-01-19 00:00:00
    2/17/14
    2014-02-17 00:00:00
    7/10/14
    2014-07-10 00:00:00
    3/20/77
    1977-03-20 00:00:00
    7/7/77
    1977-07-07 00:00:00
    6/22/77
    1977-06-22 00:00:00
    4/19/77
    1977-04-19 00:00:00
    11/3/77
    1977-11-03 00:00:00
    11/16/77
    1977-11-16 00:00:00
    6/15/77
    1977-06-15 00:00:00
    3/11/77
    1977-03-11 00:00:00
    6/24/77
    1977-06-24 00:00:00
    7/15/77
    1977-07-15 00:00:00
    9/8/77
    1977-09-08 00:00:00
    12/16/77
    1977-12-16 00:00:00
    12/25/77
    1977-12-25 00:00:00
    5/26/77
    1977-05-26 00:00:00
    1/18/77
    1977-01-18 00:00:00
    12/21/77
    1977-12-21 00:00:00
    1/29/77
    1977-01-29 00:00:00
    6/21/77
    1977-06-21 00:00:00
    2/25/77
    1977-02-25 00:00:00
    8/31/77
    1977-08-31 00:00:00
    10/2/77
    1977-10-02 00:00:00
    4/6/77
    1977-04-06 00:00:00
    6/17/77
    1977-06-17 00:00:00
    3/11/77
    1977-03-11 00:00:00
    3/28/77
    1977-03-28 00:00:00
    3/19/77
    1977-03-19 00:00:00
    10/14/77
    1977-10-14 00:00:00
    3/9/77
    1977-03-09 00:00:00
    11/27/77
    1977-11-27 00:00:00
    7/7/77
    1977-07-07 00:00:00
    3/11/77
    1977-03-11 00:00:00
    12/16/77
    1977-12-16 00:00:00
    10/28/77
    1977-10-28 00:00:00
    5/1/77
    1977-05-01 00:00:00
    7/22/77
    1977-07-22 00:00:00
    6/24/77
    1977-06-24 00:00:00
    1/1/77
    1977-01-01 00:00:00
    10/7/77
    1977-10-07 00:00:00
    4/3/77
    1977-04-03 00:00:00
    8/24/77
    1977-08-24 00:00:00
    5/12/77
    1977-05-12 00:00:00
    2/9/77
    1977-02-09 00:00:00
    6/10/77
    1977-06-10 00:00:00
    7/22/77
    1977-07-22 00:00:00
    4/7/77
    1977-04-07 00:00:00
    8/10/77
    1977-08-10 00:00:00
    7/13/77
    1977-07-13 00:00:00
    5/23/77
    1977-05-23 00:00:00
    6/17/77
    1977-06-17 00:00:00
    6/29/77
    1977-06-29 00:00:00
    5/13/77
    1977-05-13 00:00:00
    1/7/77
    1977-01-07 00:00:00
    11/30/77
    1977-11-30 00:00:00
    12/22/77
    1977-12-22 00:00:00
    4/8/77
    1977-04-08 00:00:00
    7/15/77
    1977-07-15 00:00:00
    1/1/77
    1977-01-01 00:00:00
    12/10/09
    2009-12-10 00:00:00
    8/18/09
    2009-08-18 00:00:00
    1/22/09
    2009-01-22 00:00:00
    7/7/09
    2009-07-07 00:00:00
    5/13/09
    2009-05-13 00:00:00
    5/6/09
    2009-05-06 00:00:00
    5/20/09
    2009-05-20 00:00:00
    8/5/09
    2009-08-05 00:00:00
    1/7/09
    2009-01-07 00:00:00
    7/17/09
    2009-07-17 00:00:00
    6/29/09
    2009-06-29 00:00:00
    6/5/09
    2009-06-05 00:00:00
    3/5/09
    2009-03-05 00:00:00
    10/10/09
    2009-10-10 00:00:00
    8/19/09
    2009-08-19 00:00:00
    5/13/09
    2009-05-13 00:00:00
    12/8/09
    2009-12-08 00:00:00
    11/20/09
    2009-11-20 00:00:00
    10/7/09
    2009-10-07 00:00:00
    3/15/09
    2009-03-15 00:00:00
    2/6/09
    2009-02-06 00:00:00
    8/4/09
    2009-08-04 00:00:00
    6/1/09
    2009-06-01 00:00:00
    2/6/09
    2009-02-06 00:00:00
    7/24/09
    2009-07-24 00:00:00
    9/17/09
    2009-09-17 00:00:00
    10/23/09
    2009-10-23 00:00:00
    2/5/09
    2009-02-05 00:00:00
    5/20/09
    2009-05-20 00:00:00
    3/11/09
    2009-03-11 00:00:00
    11/4/09
    2009-11-04 00:00:00
    9/16/09
    2009-09-16 00:00:00
    12/26/09
    2009-12-26 00:00:00
    1/15/09
    2009-01-15 00:00:00
    6/13/09
    2009-06-13 00:00:00
    10/15/09
    2009-10-15 00:00:00
    9/17/09
    2009-09-17 00:00:00
    4/16/09
    2009-04-16 00:00:00
    9/8/09
    2009-09-08 00:00:00
    9/3/09
    2009-09-03 00:00:00
    3/15/09
    2009-03-15 00:00:00
    10/6/09
    2009-10-06 00:00:00
    3/19/09
    2009-03-19 00:00:00
    6/12/09
    2009-06-12 00:00:00
    7/21/09
    2009-07-21 00:00:00
    6/2/09
    2009-06-02 00:00:00
    6/11/09
    2009-06-11 00:00:00
    9/13/09
    2009-09-13 00:00:00
    2/5/09
    2009-02-05 00:00:00
    5/1/09
    2009-05-01 00:00:00
    9/3/09
    2009-09-03 00:00:00
    2/27/09
    2009-02-27 00:00:00
    9/11/09
    2009-09-11 00:00:00
    9/19/09
    2009-09-19 00:00:00
    1/27/09
    2009-01-27 00:00:00
    11/25/09
    2009-11-25 00:00:00
    3/19/09
    2009-03-19 00:00:00
    5/14/09
    2009-05-14 00:00:00
    8/6/09
    2009-08-06 00:00:00
    3/25/09
    2009-03-25 00:00:00
    7/1/09
    2009-07-01 00:00:00
    11/16/09
    2009-11-16 00:00:00
    9/18/09
    2009-09-18 00:00:00
    7/24/09
    2009-07-24 00:00:00
    9/4/09
    2009-09-04 00:00:00
    8/26/09
    2009-08-26 00:00:00
    12/21/09
    2009-12-21 00:00:00
    11/7/09
    2009-11-07 00:00:00
    1/9/09
    2009-01-09 00:00:00
    11/6/09
    2009-11-06 00:00:00
    4/17/09
    2009-04-17 00:00:00
    1/9/09
    2009-01-09 00:00:00
    10/16/09
    2009-10-16 00:00:00
    3/20/09
    2009-03-20 00:00:00
    9/18/09
    2009-09-18 00:00:00
    3/12/09
    2009-03-12 00:00:00
    8/13/09
    2009-08-13 00:00:00
    4/1/09
    2009-04-01 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/6/09
    2009-01-06 00:00:00
    8/14/09
    2009-08-14 00:00:00
    2/6/09
    2009-02-06 00:00:00
    5/18/09
    2009-05-18 00:00:00
    1/1/09
    2009-01-01 00:00:00
    9/18/09
    2009-09-18 00:00:00
    2/6/09
    2009-02-06 00:00:00
    1/16/09
    2009-01-16 00:00:00
    10/22/09
    2009-10-22 00:00:00
    3/4/09
    2009-03-04 00:00:00
    10/2/09
    2009-10-02 00:00:00
    6/8/09
    2009-06-08 00:00:00
    1/16/09
    2009-01-16 00:00:00
    9/9/09
    2009-09-09 00:00:00
    1/30/09
    2009-01-30 00:00:00
    5/8/09
    2009-05-08 00:00:00
    1/18/09
    2009-01-18 00:00:00
    5/5/09
    2009-05-05 00:00:00
    10/19/09
    2009-10-19 00:00:00
    3/19/09
    2009-03-19 00:00:00
    3/5/09
    2009-03-05 00:00:00
    12/23/09
    2009-12-23 00:00:00
    6/19/09
    2009-06-19 00:00:00
    10/17/09
    2009-10-17 00:00:00
    6/19/09
    2009-06-19 00:00:00
    9/29/09
    2009-09-29 00:00:00
    8/6/09
    2009-08-06 00:00:00
    10/16/09
    2009-10-16 00:00:00
    12/23/09
    2009-12-23 00:00:00
    1/15/09
    2009-01-15 00:00:00
    1/9/09
    2009-01-09 00:00:00
    7/10/09
    2009-07-10 00:00:00
    9/11/09
    2009-09-11 00:00:00
    6/5/09
    2009-06-05 00:00:00
    12/10/09
    2009-12-10 00:00:00
    11/19/09
    2009-11-19 00:00:00
    9/9/09
    2009-09-09 00:00:00
    3/27/09
    2009-03-27 00:00:00
    6/18/09
    2009-06-18 00:00:00
    7/28/09
    2009-07-28 00:00:00
    5/1/09
    2009-05-01 00:00:00
    6/18/09
    2009-06-18 00:00:00
    10/10/09
    2009-10-10 00:00:00
    9/14/09
    2009-09-14 00:00:00
    10/17/09
    2009-10-17 00:00:00
    1/16/09
    2009-01-16 00:00:00
    11/26/09
    2009-11-26 00:00:00
    10/27/09
    2009-10-27 00:00:00
    6/3/09
    2009-06-03 00:00:00
    11/24/09
    2009-11-24 00:00:00
    5/16/09
    2009-05-16 00:00:00
    5/17/09
    2009-05-17 00:00:00
    2/9/09
    2009-02-09 00:00:00
    11/24/09
    2009-11-24 00:00:00
    9/11/09
    2009-09-11 00:00:00
    10/1/09
    2009-10-01 00:00:00
    2/6/09
    2009-02-06 00:00:00
    12/4/09
    2009-12-04 00:00:00
    9/4/09
    2009-09-04 00:00:00
    5/17/09
    2009-05-17 00:00:00
    9/12/09
    2009-09-12 00:00:00
    4/19/09
    2009-04-19 00:00:00
    10/23/09
    2009-10-23 00:00:00
    1/27/09
    2009-01-27 00:00:00
    9/13/09
    2009-09-13 00:00:00
    10/29/09
    2009-10-29 00:00:00
    7/31/09
    2009-07-31 00:00:00
    6/15/09
    2009-06-15 00:00:00
    9/5/09
    2009-09-05 00:00:00
    10/15/09
    2009-10-15 00:00:00
    2/3/09
    2009-02-03 00:00:00
    10/16/09
    2009-10-16 00:00:00
    9/9/09
    2009-09-09 00:00:00
    10/1/09
    2009-10-01 00:00:00
    4/2/09
    2009-04-02 00:00:00
    9/11/09
    2009-09-11 00:00:00
    8/14/09
    2009-08-14 00:00:00
    9/6/09
    2009-09-06 00:00:00
    4/18/09
    2009-04-18 00:00:00
    1/9/09
    2009-01-09 00:00:00
    1/24/09
    2009-01-24 00:00:00
    8/28/09
    2009-08-28 00:00:00
    10/9/09
    2009-10-09 00:00:00
    4/10/09
    2009-04-10 00:00:00
    5/17/09
    2009-05-17 00:00:00
    4/10/09
    2009-04-10 00:00:00
    4/24/09
    2009-04-24 00:00:00
    2/18/09
    2009-02-18 00:00:00
    6/9/09
    2009-06-09 00:00:00
    7/31/09
    2009-07-31 00:00:00
    4/24/09
    2009-04-24 00:00:00
    7/29/09
    2009-07-29 00:00:00
    10/14/09
    2009-10-14 00:00:00
    7/9/09
    2009-07-09 00:00:00
    10/22/09
    2009-10-22 00:00:00
    6/13/09
    2009-06-13 00:00:00
    11/13/09
    2009-11-13 00:00:00
    11/6/09
    2009-11-06 00:00:00
    11/20/09
    2009-11-20 00:00:00
    9/7/09
    2009-09-07 00:00:00
    12/4/09
    2009-12-04 00:00:00
    8/14/09
    2009-08-14 00:00:00
    9/16/09
    2009-09-16 00:00:00
    9/12/09
    2009-09-12 00:00:00
    3/19/09
    2009-03-19 00:00:00
    2/7/09
    2009-02-07 00:00:00
    9/29/09
    2009-09-29 00:00:00
    8/11/09
    2009-08-11 00:00:00
    6/8/09
    2009-06-08 00:00:00
    2/20/09
    2009-02-20 00:00:00
    2/2/09
    2009-02-02 00:00:00
    3/3/09
    2009-03-03 00:00:00
    11/5/09
    2009-11-05 00:00:00
    7/21/09
    2009-07-21 00:00:00
    10/25/09
    2009-10-25 00:00:00
    10/22/09
    2009-10-22 00:00:00
    11/19/09
    2009-11-19 00:00:00
    4/3/09
    2009-04-03 00:00:00
    9/3/09
    2009-09-03 00:00:00
    9/12/09
    2009-09-12 00:00:00
    9/12/09
    2009-09-12 00:00:00
    11/4/09
    2009-11-04 00:00:00
    10/3/09
    2009-10-03 00:00:00
    12/3/09
    2009-12-03 00:00:00
    2/28/09
    2009-02-28 00:00:00
    11/13/09
    2009-11-13 00:00:00
    10/16/09
    2009-10-16 00:00:00
    1/13/09
    2009-01-13 00:00:00
    6/5/09
    2009-06-05 00:00:00
    2/6/09
    2009-02-06 00:00:00
    6/26/09
    2009-06-26 00:00:00
    3/11/09
    2009-03-11 00:00:00
    8/14/09
    2009-08-14 00:00:00
    7/31/09
    2009-07-31 00:00:00
    4/24/09
    2009-04-24 00:00:00
    10/17/09
    2009-10-17 00:00:00
    2/11/09
    2009-02-11 00:00:00
    1/17/09
    2009-01-17 00:00:00
    2/27/09
    2009-02-27 00:00:00
    3/13/09
    2009-03-13 00:00:00
    11/7/09
    2009-11-07 00:00:00
    11/21/09
    2009-11-21 00:00:00
    11/11/09
    2009-11-11 00:00:00
    9/24/09
    2009-09-24 00:00:00
    8/6/09
    2009-08-06 00:00:00
    2/7/09
    2009-02-07 00:00:00
    7/19/09
    2009-07-19 00:00:00
    2/27/09
    2009-02-27 00:00:00
    1/16/09
    2009-01-16 00:00:00
    9/25/09
    2009-09-25 00:00:00
    5/16/09
    2009-05-16 00:00:00
    2/24/09
    2009-02-24 00:00:00
    6/19/09
    2009-06-19 00:00:00
    4/24/09
    2009-04-24 00:00:00
    6/11/09
    2009-06-11 00:00:00
    9/26/09
    2009-09-26 00:00:00
    10/3/09
    2009-10-03 00:00:00
    3/16/09
    2009-03-16 00:00:00
    10/8/09
    2009-10-08 00:00:00
    2/7/09
    2009-02-07 00:00:00
    9/13/09
    2009-09-13 00:00:00
    9/11/09
    2009-09-11 00:00:00
    10/21/09
    2009-10-21 00:00:00
    1/22/09
    2009-01-22 00:00:00
    10/28/09
    2009-10-28 00:00:00
    1/17/09
    2009-01-17 00:00:00
    9/4/09
    2009-09-04 00:00:00
    9/23/09
    2009-09-23 00:00:00
    10/23/09
    2009-10-23 00:00:00
    1/1/09
    2009-01-01 00:00:00
    5/1/09
    2009-05-01 00:00:00
    11/25/09
    2009-11-25 00:00:00
    9/16/09
    2009-09-16 00:00:00
    1/1/09
    2009-01-01 00:00:00
    9/9/09
    2009-09-09 00:00:00
    5/22/09
    2009-05-22 00:00:00
    5/26/09
    2009-05-26 00:00:00
    2/28/09
    2009-02-28 00:00:00
    9/15/09
    2009-09-15 00:00:00
    4/28/09
    2009-04-28 00:00:00
    1/13/09
    2009-01-13 00:00:00
    10/28/09
    2009-10-28 00:00:00
    1/19/09
    2009-01-19 00:00:00
    4/28/09
    2009-04-28 00:00:00
    5/26/09
    2009-05-26 00:00:00
    3/24/09
    2009-03-24 00:00:00
    6/26/09
    2009-06-26 00:00:00
    9/14/09
    2009-09-14 00:00:00
    10/16/09
    2009-10-16 00:00:00
    1/1/09
    2009-01-01 00:00:00
    3/6/09
    2009-03-06 00:00:00
    12/25/09
    2009-12-25 00:00:00
    2/9/09
    2009-02-09 00:00:00
    10/22/09
    2009-10-22 00:00:00
    2/17/09
    2009-02-17 00:00:00
    9/24/09
    2009-09-24 00:00:00
    5/28/09
    2009-05-28 00:00:00
    2/9/09
    2009-02-09 00:00:00
    1/25/09
    2009-01-25 00:00:00
    6/17/09
    2009-06-17 00:00:00
    2/27/09
    2009-02-27 00:00:00
    2/5/09
    2009-02-05 00:00:00
    11/5/09
    2009-11-05 00:00:00
    7/16/09
    2009-07-16 00:00:00
    11/27/09
    2009-11-27 00:00:00
    9/2/09
    2009-09-02 00:00:00
    11/9/09
    2009-11-09 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/21/09
    2009-01-21 00:00:00
    1/30/09
    2009-01-30 00:00:00
    11/25/09
    2009-11-25 00:00:00
    8/21/09
    2009-08-21 00:00:00
    9/6/09
    2009-09-06 00:00:00
    12/6/09
    2009-12-06 00:00:00
    3/13/09
    2009-03-13 00:00:00
    3/13/09
    2009-03-13 00:00:00
    1/19/09
    2009-01-19 00:00:00
    3/21/09
    2009-03-21 00:00:00
    6/27/09
    2009-06-27 00:00:00
    7/7/09
    2009-07-07 00:00:00
    6/19/09
    2009-06-19 00:00:00
    2/6/09
    2009-02-06 00:00:00
    1/1/09
    2009-01-01 00:00:00
    3/31/09
    2009-03-31 00:00:00
    10/1/09
    2009-10-01 00:00:00
    12/6/09
    2009-12-06 00:00:00
    12/8/09
    2009-12-08 00:00:00
    9/30/09
    2009-09-30 00:00:00
    2/20/09
    2009-02-20 00:00:00
    4/26/09
    2009-04-26 00:00:00
    7/14/09
    2009-07-14 00:00:00
    4/1/09
    2009-04-01 00:00:00
    2/9/09
    2009-02-09 00:00:00
    4/2/09
    2009-04-02 00:00:00
    9/9/09
    2009-09-09 00:00:00
    11/3/09
    2009-11-03 00:00:00
    5/9/09
    2009-05-09 00:00:00
    1/1/09
    2009-01-01 00:00:00
    7/24/09
    2009-07-24 00:00:00
    1/16/09
    2009-01-16 00:00:00
    4/25/09
    2009-04-25 00:00:00
    4/28/09
    2009-04-28 00:00:00
    8/16/09
    2009-08-16 00:00:00
    6/25/09
    2009-06-25 00:00:00
    8/28/09
    2009-08-28 00:00:00
    12/18/09
    2009-12-18 00:00:00
    12/25/09
    2009-12-25 00:00:00
    2/27/09
    2009-02-27 00:00:00
    12/18/09
    2009-12-18 00:00:00
    4/23/09
    2009-04-23 00:00:00
    1/6/09
    2009-01-06 00:00:00
    10/22/09
    2009-10-22 00:00:00
    2/17/09
    2009-02-17 00:00:00
    9/19/09
    2009-09-19 00:00:00
    2/7/09
    2009-02-07 00:00:00
    9/30/09
    2009-09-30 00:00:00
    6/18/09
    2009-06-18 00:00:00
    2/10/09
    2009-02-10 00:00:00
    3/24/09
    2009-03-24 00:00:00
    9/13/09
    2009-09-13 00:00:00
    1/18/09
    2009-01-18 00:00:00
    1/28/09
    2009-01-28 00:00:00
    12/5/09
    2009-12-05 00:00:00
    2/20/09
    2009-02-20 00:00:00
    2/8/09
    2009-02-08 00:00:00
    8/21/09
    2009-08-21 00:00:00
    9/18/09
    2009-09-18 00:00:00
    12/4/09
    2009-12-04 00:00:00
    9/18/09
    2009-09-18 00:00:00
    8/20/09
    2009-08-20 00:00:00
    8/11/09
    2009-08-11 00:00:00
    1/20/09
    2009-01-20 00:00:00
    8/30/09
    2009-08-30 00:00:00
    1/18/09
    2009-01-18 00:00:00
    12/4/09
    2009-12-04 00:00:00
    10/5/09
    2009-10-05 00:00:00
    11/2/09
    2009-11-02 00:00:00
    8/7/09
    2009-08-07 00:00:00
    4/21/09
    2009-04-21 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/17/09
    2009-01-17 00:00:00
    1/14/09
    2009-01-14 00:00:00
    10/5/09
    2009-10-05 00:00:00
    1/1/09
    2009-01-01 00:00:00
    12/20/09
    2009-12-20 00:00:00
    1/27/09
    2009-01-27 00:00:00
    12/5/09
    2009-12-05 00:00:00
    2/13/09
    2009-02-13 00:00:00
    5/31/09
    2009-05-31 00:00:00
    4/24/09
    2009-04-24 00:00:00
    10/7/09
    2009-10-07 00:00:00
    8/21/09
    2009-08-21 00:00:00
    8/14/09
    2009-08-14 00:00:00
    7/31/09
    2009-07-31 00:00:00
    8/27/09
    2009-08-27 00:00:00
    12/29/09
    2009-12-29 00:00:00
    1/1/09
    2009-01-01 00:00:00
    6/11/09
    2009-06-11 00:00:00
    9/29/09
    2009-09-29 00:00:00
    5/27/09
    2009-05-27 00:00:00
    9/3/09
    2009-09-03 00:00:00
    11/10/09
    2009-11-10 00:00:00
    4/28/09
    2009-04-28 00:00:00
    9/21/09
    2009-09-21 00:00:00
    1/16/09
    2009-01-16 00:00:00
    9/13/09
    2009-09-13 00:00:00
    12/4/09
    2009-12-04 00:00:00
    8/24/09
    2009-08-24 00:00:00
    1/1/09
    2009-01-01 00:00:00
    12/16/09
    2009-12-16 00:00:00
    3/19/09
    2009-03-19 00:00:00
    5/22/09
    2009-05-22 00:00:00
    8/14/09
    2009-08-14 00:00:00
    1/1/09
    2009-01-01 00:00:00
    2/11/09
    2009-02-11 00:00:00
    10/10/09
    2009-10-10 00:00:00
    6/27/09
    2009-06-27 00:00:00
    5/1/09
    2009-05-01 00:00:00
    1/1/09
    2009-01-01 00:00:00
    4/19/09
    2009-04-19 00:00:00
    2/5/09
    2009-02-05 00:00:00
    7/29/09
    2009-07-29 00:00:00
    9/11/09
    2009-09-11 00:00:00
    9/27/09
    2009-09-27 00:00:00
    9/16/09
    2009-09-16 00:00:00
    9/12/09
    2009-09-12 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/1/09
    2009-01-01 00:00:00
    9/5/09
    2009-09-05 00:00:00
    5/14/09
    2009-05-14 00:00:00
    8/28/09
    2009-08-28 00:00:00
    2/17/09
    2009-02-17 00:00:00
    2/5/09
    2009-02-05 00:00:00
    6/17/09
    2009-06-17 00:00:00
    7/17/09
    2009-07-17 00:00:00
    9/25/09
    2009-09-25 00:00:00
    1/17/09
    2009-01-17 00:00:00
    10/11/09
    2009-10-11 00:00:00
    10/10/09
    2009-10-10 00:00:00
    5/19/09
    2009-05-19 00:00:00
    6/17/09
    2009-06-17 00:00:00
    8/14/09
    2009-08-14 00:00:00
    2/27/09
    2009-02-27 00:00:00
    7/8/09
    2009-07-08 00:00:00
    8/28/09
    2009-08-28 00:00:00
    4/23/09
    2009-04-23 00:00:00
    11/15/09
    2009-11-15 00:00:00
    1/1/09
    2009-01-01 00:00:00
    2/23/09
    2009-02-23 00:00:00
    11/8/09
    2009-11-08 00:00:00
    11/4/09
    2009-11-04 00:00:00
    8/28/09
    2009-08-28 00:00:00
    5/4/09
    2009-05-04 00:00:00
    4/27/09
    2009-04-27 00:00:00
    10/9/09
    2009-10-09 00:00:00
    3/3/09
    2009-03-03 00:00:00
    1/2/09
    2009-01-02 00:00:00
    11/15/09
    2009-11-15 00:00:00
    4/15/09
    2009-04-15 00:00:00
    2/5/09
    2009-02-05 00:00:00
    2/6/09
    2009-02-06 00:00:00
    4/25/09
    2009-04-25 00:00:00
    10/23/09
    2009-10-23 00:00:00
    1/1/09
    2009-01-01 00:00:00
    12/17/09
    2009-12-17 00:00:00
    8/5/09
    2009-08-05 00:00:00
    5/12/09
    2009-05-12 00:00:00
    2/7/09
    2009-02-07 00:00:00
    7/26/09
    2009-07-26 00:00:00
    2/3/09
    2009-02-03 00:00:00
    7/21/09
    2009-07-21 00:00:00
    11/25/09
    2009-11-25 00:00:00
    11/1/09
    2009-11-01 00:00:00
    11/5/09
    2009-11-05 00:00:00
    9/9/09
    2009-09-09 00:00:00
    10/9/09
    2009-10-09 00:00:00
    1/23/09
    2009-01-23 00:00:00
    1/13/09
    2009-01-13 00:00:00
    9/25/09
    2009-09-25 00:00:00
    10/30/09
    2009-10-30 00:00:00
    7/10/09
    2009-07-10 00:00:00
    4/24/09
    2009-04-24 00:00:00
    7/21/09
    2009-07-21 00:00:00
    2/1/09
    2009-02-01 00:00:00
    2/16/09
    2009-02-16 00:00:00
    1/1/09
    2009-01-01 00:00:00
    10/27/09
    2009-10-27 00:00:00
    1/9/09
    2009-01-09 00:00:00
    1/18/09
    2009-01-18 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/1/09
    2009-01-01 00:00:00
    8/21/09
    2009-08-21 00:00:00
    9/11/09
    2009-09-11 00:00:00
    11/21/09
    2009-11-21 00:00:00
    5/29/09
    2009-05-29 00:00:00
    9/13/09
    2009-09-13 00:00:00
    4/24/09
    2009-04-24 00:00:00
    9/21/09
    2009-09-21 00:00:00
    4/21/09
    2009-04-21 00:00:00
    12/14/09
    2009-12-14 00:00:00
    10/1/09
    2009-10-01 00:00:00
    4/2/09
    2009-04-02 00:00:00
    1/1/09
    2009-01-01 00:00:00
    1/1/09
    2009-01-01 00:00:00
    9/22/09
    2009-09-22 00:00:00
    9/1/09
    2009-09-01 00:00:00
    12/4/09
    2009-12-04 00:00:00
    4/9/09
    2009-04-09 00:00:00
    12/22/09
    2009-12-22 00:00:00
    9/8/09
    2009-09-08 00:00:00
    8/21/09
    2009-08-21 00:00:00
    3/27/09
    2009-03-27 00:00:00
    9/12/09
    2009-09-12 00:00:00
    12/26/09
    2009-12-26 00:00:00
    4/16/09
    2009-04-16 00:00:00
    5/30/09
    2009-05-30 00:00:00
    10/28/09
    2009-10-28 00:00:00
    2/16/09
    2009-02-16 00:00:00
    2/18/09
    2009-02-18 00:00:00
    4/24/09
    2009-04-24 00:00:00
    4/1/09
    2009-04-01 00:00:00
    3/7/09
    2009-03-07 00:00:00
    1/14/09
    2009-01-14 00:00:00
    3/14/09
    2009-03-14 00:00:00
    12/26/09
    2009-12-26 00:00:00
    8/6/09
    2009-08-06 00:00:00
    1/19/09
    2009-01-19 00:00:00
    4/3/09
    2009-04-03 00:00:00
    7/31/09
    2009-07-31 00:00:00
    12/19/09
    2009-12-19 00:00:00
    1/1/09
    2009-01-01 00:00:00
    2/27/09
    2009-02-27 00:00:00
    12/9/09
    2009-12-09 00:00:00
    1/1/09
    2009-01-01 00:00:00
    9/5/09
    2009-09-05 00:00:00
    11/20/09
    2009-11-20 00:00:00
    1/18/09
    2009-01-18 00:00:00
    1/1/09
    2009-01-01 00:00:00
    10/14/09
    2009-10-14 00:00:00
    6/12/09
    2009-06-12 00:00:00
    1/1/09
    2009-01-01 00:00:00
    7/28/09
    2009-07-28 00:00:00
    8/14/09
    2009-08-14 00:00:00
    12/2/09
    2009-12-02 00:00:00
    2/1/09
    2009-02-01 00:00:00
    2/3/09
    2009-02-03 00:00:00
    7/28/09
    2009-07-28 00:00:00
    11/27/09
    2009-11-27 00:00:00
    6/26/09
    2009-06-26 00:00:00
    11/17/09
    2009-11-17 00:00:00
    5/16/09
    2009-05-16 00:00:00
    7/22/09
    2009-07-22 00:00:00
    10/27/09
    2009-10-27 00:00:00
    7/14/10
    2010-07-14 00:00:00
    4/28/10
    2010-04-28 00:00:00
    3/3/10
    2010-03-03 00:00:00
    12/2/10
    2010-12-02 00:00:00
    10/17/10
    2010-10-17 00:00:00
    7/8/10
    2010-07-08 00:00:00
    8/3/10
    2010-08-03 00:00:00
    3/5/10
    2010-03-05 00:00:00
    2/18/10
    2010-02-18 00:00:00
    12/10/10
    2010-12-10 00:00:00
    11/24/10
    2010-11-24 00:00:00
    6/16/10
    2010-06-16 00:00:00
    7/21/10
    2010-07-21 00:00:00
    11/5/10
    2010-11-05 00:00:00
    9/10/10
    2010-09-10 00:00:00
    8/13/10
    2010-08-13 00:00:00
    3/22/10
    2010-03-22 00:00:00
    12/8/10
    2010-12-08 00:00:00
    5/19/10
    2010-05-19 00:00:00
    5/12/10
    2010-05-12 00:00:00
    12/22/10
    2010-12-22 00:00:00
    4/1/10
    2010-04-01 00:00:00
    7/13/10
    2010-07-13 00:00:00
    5/14/10
    2010-05-14 00:00:00
    9/30/10
    2010-09-30 00:00:00
    7/27/10
    2010-07-27 00:00:00
    12/21/10
    2010-12-21 00:00:00
    7/3/10
    2010-07-03 00:00:00
    2/1/10
    2010-02-01 00:00:00
    11/4/10
    2010-11-04 00:00:00
    5/16/10
    2010-05-16 00:00:00
    2/4/10
    2010-02-04 00:00:00
    6/10/10
    2010-06-10 00:00:00
    10/13/10
    2010-10-13 00:00:00
    9/1/10
    2010-09-01 00:00:00
    6/24/10
    2010-06-24 00:00:00
    1/14/10
    2010-01-14 00:00:00
    4/8/10
    2010-04-08 00:00:00
    9/15/10
    2010-09-15 00:00:00
    3/12/10
    2010-03-12 00:00:00
    6/23/10
    2010-06-23 00:00:00
    10/28/10
    2010-10-28 00:00:00
    6/30/10
    2010-06-30 00:00:00
    8/6/10
    2010-08-06 00:00:00
    10/4/10
    2010-10-04 00:00:00
    9/13/10
    2010-09-13 00:00:00
    8/4/10
    2010-08-04 00:00:00
    9/6/10
    2010-09-06 00:00:00
    11/22/10
    2010-11-22 00:00:00
    9/3/10
    2010-09-03 00:00:00
    3/18/10
    2010-03-18 00:00:00
    5/11/10
    2010-05-11 00:00:00
    6/15/10
    2010-06-15 00:00:00
    5/22/10
    2010-05-22 00:00:00
    8/12/10
    2010-08-12 00:00:00
    12/27/10
    2010-12-27 00:00:00
    3/16/10
    2010-03-16 00:00:00
    1/8/10
    2010-01-08 00:00:00
    3/11/10
    2010-03-11 00:00:00
    12/25/10
    2010-12-25 00:00:00
    4/23/10
    2010-04-23 00:00:00
    5/1/10
    2010-05-01 00:00:00
    5/11/10
    2010-05-11 00:00:00
    3/19/10
    2010-03-19 00:00:00
    2/10/10
    2010-02-10 00:00:00
    11/6/10
    2010-11-06 00:00:00
    8/13/10
    2010-08-13 00:00:00
    10/21/10
    2010-10-21 00:00:00
    6/10/10
    2010-06-10 00:00:00
    11/4/10
    2010-11-04 00:00:00
    6/17/10
    2010-06-17 00:00:00
    2/5/10
    2010-02-05 00:00:00
    2/12/10
    2010-02-12 00:00:00
    7/6/10
    2010-07-06 00:00:00
    11/18/10
    2010-11-18 00:00:00
    4/30/10
    2010-04-30 00:00:00
    4/23/10
    2010-04-23 00:00:00
    12/17/10
    2010-12-17 00:00:00
    6/4/10
    2010-06-04 00:00:00
    1/22/10
    2010-01-22 00:00:00
    6/18/10
    2010-06-18 00:00:00
    8/22/10
    2010-08-22 00:00:00
    10/10/10
    2010-10-10 00:00:00
    2/12/10
    2010-02-12 00:00:00
    8/18/10
    2010-08-18 00:00:00
    10/8/10
    2010-10-08 00:00:00
    6/4/10
    2010-06-04 00:00:00
    1/21/10
    2010-01-21 00:00:00
    3/12/10
    2010-03-12 00:00:00
    5/26/10
    2010-05-26 00:00:00
    9/24/10
    2010-09-24 00:00:00
    3/26/10
    2010-03-26 00:00:00
    3/26/10
    2010-03-26 00:00:00
    7/15/10
    2010-07-15 00:00:00
    10/1/10
    2010-10-01 00:00:00
    8/6/10
    2010-08-06 00:00:00
    7/10/10
    2010-07-10 00:00:00
    8/26/10
    2010-08-26 00:00:00
    11/23/10
    2010-11-23 00:00:00
    10/21/10
    2010-10-21 00:00:00
    9/1/10
    2010-09-01 00:00:00
    11/23/10
    2010-11-23 00:00:00
    3/10/10
    2010-03-10 00:00:00
    10/20/10
    2010-10-20 00:00:00
    9/15/10
    2010-09-15 00:00:00
    7/27/10
    2010-07-27 00:00:00
    8/31/10
    2010-08-31 00:00:00
    11/26/10
    2010-11-26 00:00:00
    2/26/10
    2010-02-26 00:00:00
    9/23/10
    2010-09-23 00:00:00
    6/18/10
    2010-06-18 00:00:00
    9/10/10
    2010-09-10 00:00:00
    10/8/10
    2010-10-08 00:00:00
    2/15/10
    2010-02-15 00:00:00
    9/11/10
    2010-09-11 00:00:00
    12/13/10
    2010-12-13 00:00:00
    5/17/10
    2010-05-17 00:00:00
    3/27/10
    2010-03-27 00:00:00
    10/7/10
    2010-10-07 00:00:00
    3/19/10
    2010-03-19 00:00:00
    6/17/10
    2010-06-17 00:00:00
    8/20/10
    2010-08-20 00:00:00
    1/12/10
    2010-01-12 00:00:00
    9/11/10
    2010-09-11 00:00:00
    11/24/10
    2010-11-24 00:00:00
    8/27/10
    2010-08-27 00:00:00
    1/14/10
    2010-01-14 00:00:00
    4/18/10
    2010-04-18 00:00:00
    5/20/10
    2010-05-20 00:00:00
    9/13/10
    2010-09-13 00:00:00
    2/2/10
    2010-02-02 00:00:00
    9/24/10
    2010-09-24 00:00:00
    7/30/10
    2010-07-30 00:00:00
    1/29/10
    2010-01-29 00:00:00
    11/27/10
    2010-11-27 00:00:00
    2/10/10
    2010-02-10 00:00:00
    7/9/10
    2010-07-09 00:00:00
    10/12/10
    2010-10-12 00:00:00
    11/11/10
    2010-11-11 00:00:00
    4/11/10
    2010-04-11 00:00:00
    8/3/10
    2010-08-03 00:00:00
    7/30/10
    2010-07-30 00:00:00
    6/14/10
    2010-06-14 00:00:00
    6/11/10
    2010-06-11 00:00:00
    1/20/10
    2010-01-20 00:00:00
    1/1/10
    2010-01-01 00:00:00
    7/30/10
    2010-07-30 00:00:00
    6/6/10
    2010-06-06 00:00:00
    5/21/10
    2010-05-21 00:00:00
    7/23/10
    2010-07-23 00:00:00
    2/11/10
    2010-02-11 00:00:00
    3/26/10
    2010-03-26 00:00:00
    12/17/10
    2010-12-17 00:00:00
    10/15/10
    2010-10-15 00:00:00
    12/13/10
    2010-12-13 00:00:00
    11/12/10
    2010-11-12 00:00:00
    10/7/10
    2010-10-07 00:00:00
    9/24/10
    2010-09-24 00:00:00
    7/9/10
    2010-07-09 00:00:00
    9/14/10
    2010-09-14 00:00:00
    1/15/10
    2010-01-15 00:00:00
    3/11/10
    2010-03-11 00:00:00
    6/7/10
    2010-06-07 00:00:00
    12/3/10
    2010-12-03 00:00:00
    1/14/10
    2010-01-14 00:00:00
    2/19/10
    2010-02-19 00:00:00
    11/10/10
    2010-11-10 00:00:00
    10/15/10
    2010-10-15 00:00:00
    6/10/10
    2010-06-10 00:00:00
    10/8/10
    2010-10-08 00:00:00
    3/20/10
    2010-03-20 00:00:00
    3/20/10
    2010-03-20 00:00:00
    9/10/10
    2010-09-10 00:00:00
    4/8/10
    2010-04-08 00:00:00
    4/15/10
    2010-04-15 00:00:00
    10/18/10
    2010-10-18 00:00:00
    1/8/10
    2010-01-08 00:00:00
    1/22/10
    2010-01-22 00:00:00
    6/3/10
    2010-06-03 00:00:00
    6/4/10
    2010-06-04 00:00:00
    9/10/10
    2010-09-10 00:00:00
    9/23/10
    2010-09-23 00:00:00
    11/5/10
    2010-11-05 00:00:00
    2/5/10
    2010-02-05 00:00:00
    9/2/10
    2010-09-02 00:00:00
    9/13/10
    2010-09-13 00:00:00
    10/15/10
    2010-10-15 00:00:00
    1/22/10
    2010-01-22 00:00:00
    5/19/10
    2010-05-19 00:00:00
    4/20/10
    2010-04-20 00:00:00
    6/10/10
    2010-06-10 00:00:00
    9/15/10
    2010-09-15 00:00:00
    12/4/10
    2010-12-04 00:00:00
    6/15/10
    2010-06-15 00:00:00
    3/31/10
    2010-03-31 00:00:00
    8/21/10
    2010-08-21 00:00:00
    11/5/10
    2010-11-05 00:00:00
    9/14/10
    2010-09-14 00:00:00
    10/15/10
    2010-10-15 00:00:00
    1/26/10
    2010-01-26 00:00:00
    2/26/10
    2010-02-26 00:00:00
    8/11/10
    2010-08-11 00:00:00
    7/19/10
    2010-07-19 00:00:00
    6/16/10
    2010-06-16 00:00:00
    12/16/10
    2010-12-16 00:00:00
    9/3/10
    2010-09-03 00:00:00
    9/17/10
    2010-09-17 00:00:00
    5/23/10
    2010-05-23 00:00:00
    4/14/10
    2010-04-14 00:00:00
    10/15/10
    2010-10-15 00:00:00
    1/25/10
    2010-01-25 00:00:00
    4/24/10
    2010-04-24 00:00:00
    6/25/10
    2010-06-25 00:00:00
    2/14/10
    2010-02-14 00:00:00
    7/20/10
    2010-07-20 00:00:00
    12/11/10
    2010-12-11 00:00:00
    5/4/10
    2010-05-04 00:00:00
    1/13/10
    2010-01-13 00:00:00
    5/19/10
    2010-05-19 00:00:00
    10/28/10
    2010-10-28 00:00:00
    6/8/10
    2010-06-08 00:00:00
    5/14/10
    2010-05-14 00:00:00
    9/28/10
    2010-09-28 00:00:00
    9/9/10
    2010-09-09 00:00:00
    12/10/10
    2010-12-10 00:00:00
    3/10/10
    2010-03-10 00:00:00
    4/23/10
    2010-04-23 00:00:00
    12/25/10
    2010-12-25 00:00:00
    6/23/10
    2010-06-23 00:00:00
    7/15/10
    2010-07-15 00:00:00
    12/7/10
    2010-12-07 00:00:00
    4/22/10
    2010-04-22 00:00:00
    1/14/10
    2010-01-14 00:00:00
    9/30/10
    2010-09-30 00:00:00
    11/5/10
    2010-11-05 00:00:00
    11/24/10
    2010-11-24 00:00:00
    5/30/10
    2010-05-30 00:00:00
    11/9/10
    2010-11-09 00:00:00
    8/7/10
    2010-08-07 00:00:00
    12/22/10
    2010-12-22 00:00:00
    1/23/10
    2010-01-23 00:00:00
    2/23/10
    2010-02-23 00:00:00
    1/24/10
    2010-01-24 00:00:00
    10/7/10
    2010-10-07 00:00:00
    12/16/10
    2010-12-16 00:00:00
    8/8/10
    2010-08-08 00:00:00
    10/31/10
    2010-10-31 00:00:00
    4/24/10
    2010-04-24 00:00:00
    1/30/10
    2010-01-30 00:00:00
    2/25/10
    2010-02-25 00:00:00
    8/20/10
    2010-08-20 00:00:00
    7/23/10
    2010-07-23 00:00:00
    1/21/10
    2010-01-21 00:00:00
    10/22/10
    2010-10-22 00:00:00
    1/13/10
    2010-01-13 00:00:00
    8/22/10
    2010-08-22 00:00:00
    10/29/10
    2010-10-29 00:00:00
    8/11/10
    2010-08-11 00:00:00
    8/26/10
    2010-08-26 00:00:00
    12/6/10
    2010-12-06 00:00:00
    6/17/10
    2010-06-17 00:00:00
    6/18/10
    2010-06-18 00:00:00
    1/19/10
    2010-01-19 00:00:00
    9/10/10
    2010-09-10 00:00:00
    10/15/10
    2010-10-15 00:00:00
    1/1/10
    2010-01-01 00:00:00
    12/7/10
    2010-12-07 00:00:00
    9/3/10
    2010-09-03 00:00:00
    3/12/10
    2010-03-12 00:00:00
    2/4/10
    2010-02-04 00:00:00
    12/10/10
    2010-12-10 00:00:00
    7/20/10
    2010-07-20 00:00:00
    2/16/10
    2010-02-16 00:00:00
    3/23/10
    2010-03-23 00:00:00
    10/26/10
    2010-10-26 00:00:00
    5/17/10
    2010-05-17 00:00:00
    3/14/10
    2010-03-14 00:00:00
    1/29/10
    2010-01-29 00:00:00
    10/10/10
    2010-10-10 00:00:00
    5/15/10
    2010-05-15 00:00:00
    10/10/10
    2010-10-10 00:00:00
    3/19/10
    2010-03-19 00:00:00
    6/2/10
    2010-06-02 00:00:00
    9/12/10
    2010-09-12 00:00:00
    5/23/10
    2010-05-23 00:00:00
    7/16/10
    2010-07-16 00:00:00
    12/13/10
    2010-12-13 00:00:00
    9/3/10
    2010-09-03 00:00:00
    9/17/10
    2010-09-17 00:00:00
    4/2/10
    2010-04-02 00:00:00
    9/16/10
    2010-09-16 00:00:00
    5/21/10
    2010-05-21 00:00:00
    2/24/10
    2010-02-24 00:00:00
    8/26/10
    2010-08-26 00:00:00
    9/4/10
    2010-09-04 00:00:00
    5/16/10
    2010-05-16 00:00:00
    8/11/10
    2010-08-11 00:00:00
    8/21/10
    2010-08-21 00:00:00
    5/17/10
    2010-05-17 00:00:00
    1/1/10
    2010-01-01 00:00:00
    1/1/10
    2010-01-01 00:00:00
    10/26/10
    2010-10-26 00:00:00
    8/10/10
    2010-08-10 00:00:00
    5/1/10
    2010-05-01 00:00:00
    5/28/10
    2010-05-28 00:00:00
    1/30/10
    2010-01-30 00:00:00
    6/1/10
    2010-06-01 00:00:00
    10/17/10
    2010-10-17 00:00:00
    8/26/10
    2010-08-26 00:00:00
    10/1/10
    2010-10-01 00:00:00
    11/12/10
    2010-11-12 00:00:00
    4/1/10
    2010-04-01 00:00:00
    1/20/10
    2010-01-20 00:00:00
    12/25/10
    2010-12-25 00:00:00
    3/23/10
    2010-03-23 00:00:00
    5/9/10
    2010-05-09 00:00:00
    12/31/10
    2010-12-31 00:00:00
    9/4/10
    2010-09-04 00:00:00
    9/30/10
    2010-09-30 00:00:00
    7/27/10
    2010-07-27 00:00:00
    1/1/10
    2010-01-01 00:00:00
    9/18/10
    2010-09-18 00:00:00
    10/8/10
    2010-10-08 00:00:00
    1/23/10
    2010-01-23 00:00:00
    6/25/10
    2010-06-25 00:00:00
    12/2/10
    2010-12-02 00:00:00
    3/19/10
    2010-03-19 00:00:00
    10/29/10
    2010-10-29 00:00:00
    8/24/10
    2010-08-24 00:00:00
    9/12/10
    2010-09-12 00:00:00
    2/16/10
    2010-02-16 00:00:00
    5/26/10
    2010-05-26 00:00:00
    5/12/10
    2010-05-12 00:00:00
    3/14/10
    2010-03-14 00:00:00
    10/1/10
    2010-10-01 00:00:00
    8/27/10
    2010-08-27 00:00:00
    5/7/10
    2010-05-07 00:00:00
    6/22/10
    2010-06-22 00:00:00
    4/15/10
    2010-04-15 00:00:00
    11/23/10
    2010-11-23 00:00:00
    9/13/10
    2010-09-13 00:00:00
    12/31/10
    2010-12-31 00:00:00
    12/29/10
    2010-12-29 00:00:00
    4/14/10
    2010-04-14 00:00:00
    1/19/10
    2010-01-19 00:00:00
    4/2/10
    2010-04-02 00:00:00
    7/9/10
    2010-07-09 00:00:00
    1/5/10
    2010-01-05 00:00:00
    10/8/10
    2010-10-08 00:00:00
    10/3/10
    2010-10-03 00:00:00
    10/2/10
    2010-10-02 00:00:00
    12/19/10
    2010-12-19 00:00:00
    1/1/10
    2010-01-01 00:00:00
    9/10/10
    2010-09-10 00:00:00
    10/6/10
    2010-10-06 00:00:00
    2/14/10
    2010-02-14 00:00:00
    1/24/10
    2010-01-24 00:00:00
    12/12/10
    2010-12-12 00:00:00
    8/1/10
    2010-08-01 00:00:00
    7/27/10
    2010-07-27 00:00:00
    1/28/10
    2010-01-28 00:00:00
    12/21/10
    2010-12-21 00:00:00
    9/1/10
    2010-09-01 00:00:00
    4/18/10
    2010-04-18 00:00:00
    9/10/10
    2010-09-10 00:00:00
    1/1/10
    2010-01-01 00:00:00
    8/4/10
    2010-08-04 00:00:00
    10/19/10
    2010-10-19 00:00:00
    12/19/10
    2010-12-19 00:00:00
    1/1/10
    2010-01-01 00:00:00
    10/6/10
    2010-10-06 00:00:00
    12/28/10
    2010-12-28 00:00:00
    5/30/10
    2010-05-30 00:00:00
    9/14/10
    2010-09-14 00:00:00
    11/6/10
    2010-11-06 00:00:00
    4/9/10
    2010-04-09 00:00:00
    11/5/10
    2010-11-05 00:00:00
    3/14/10
    2010-03-14 00:00:00
    10/19/10
    2010-10-19 00:00:00
    7/27/10
    2010-07-27 00:00:00
    10/8/10
    2010-10-08 00:00:00
    9/24/10
    2010-09-24 00:00:00
    9/1/10
    2010-09-01 00:00:00
    8/4/10
    2010-08-04 00:00:00
    12/17/10
    2010-12-17 00:00:00
    4/28/10
    2010-04-28 00:00:00
    10/14/10
    2010-10-14 00:00:00
    3/11/10
    2010-03-11 00:00:00
    9/7/10
    2010-09-07 00:00:00
    8/26/10
    2010-08-26 00:00:00
    6/25/10
    2010-06-25 00:00:00
    2/6/10
    2010-02-06 00:00:00
    9/15/10
    2010-09-15 00:00:00
    9/25/10
    2010-09-25 00:00:00
    6/10/10
    2010-06-10 00:00:00
    3/9/10
    2010-03-09 00:00:00
    8/14/10
    2010-08-14 00:00:00
    2/5/10
    2010-02-05 00:00:00
    9/5/10
    2010-09-05 00:00:00
    1/26/10
    2010-01-26 00:00:00
    10/8/10
    2010-10-08 00:00:00
    8/12/10
    2010-08-12 00:00:00
    8/21/10
    2010-08-21 00:00:00
    5/10/10
    2010-05-10 00:00:00
    9/11/10
    2010-09-11 00:00:00
    8/30/10
    2010-08-30 00:00:00
    4/10/10
    2010-04-10 00:00:00
    12/7/10
    2010-12-07 00:00:00
    7/30/10
    2010-07-30 00:00:00
    6/8/10
    2010-06-08 00:00:00
    2/9/10
    2010-02-09 00:00:00
    10/15/10
    2010-10-15 00:00:00
    10/1/10
    2010-10-01 00:00:00
    9/6/10
    2010-09-06 00:00:00
    1/15/10
    2010-01-15 00:00:00
    6/29/10
    2010-06-29 00:00:00
    7/1/10
    2010-07-01 00:00:00
    5/10/10
    2010-05-10 00:00:00
    9/9/10
    2010-09-09 00:00:00
    4/2/10
    2010-04-02 00:00:00
    10/22/10
    2010-10-22 00:00:00
    2/10/10
    2010-02-10 00:00:00
    10/1/10
    2010-10-01 00:00:00
    12/30/10
    2010-12-30 00:00:00
    9/9/10
    2010-09-09 00:00:00
    1/29/10
    2010-01-29 00:00:00
    1/22/10
    2010-01-22 00:00:00
    7/18/10
    2010-07-18 00:00:00
    6/3/10
    2010-06-03 00:00:00
    12/3/10
    2010-12-03 00:00:00
    3/13/10
    2010-03-13 00:00:00
    4/20/10
    2010-04-20 00:00:00
    5/7/10
    2010-05-07 00:00:00
    7/22/10
    2010-07-22 00:00:00
    4/9/10
    2010-04-09 00:00:00
    7/3/10
    2010-07-03 00:00:00
    4/30/10
    2010-04-30 00:00:00
    8/21/10
    2010-08-21 00:00:00
    11/19/10
    2010-11-19 00:00:00
    9/1/10
    2010-09-01 00:00:00
    9/11/10
    2010-09-11 00:00:00
    10/29/10
    2010-10-29 00:00:00
    11/25/10
    2010-11-25 00:00:00
    12/16/10
    2010-12-16 00:00:00
    10/5/10
    2010-10-05 00:00:00
    5/1/10
    2010-05-01 00:00:00
    5/21/10
    2010-05-21 00:00:00
    11/26/10
    2010-11-26 00:00:00
    1/24/10
    2010-01-24 00:00:00
    4/16/10
    2010-04-16 00:00:00
    6/22/10
    2010-06-22 00:00:00
    6/5/10
    2010-06-05 00:00:00
    3/7/10
    2010-03-07 00:00:00
    5/3/10
    2010-05-03 00:00:00
    2/27/10
    2010-02-27 00:00:00
    7/20/10
    2010-07-20 00:00:00
    6/8/10
    2010-06-08 00:00:00
    12/25/10
    2010-12-25 00:00:00
    7/28/10
    2010-07-28 00:00:00
    11/5/10
    2010-11-05 00:00:00
    1/22/10
    2010-01-22 00:00:00
    12/12/10
    2010-12-12 00:00:00
    7/30/10
    2010-07-30 00:00:00
    1/12/10
    2010-01-12 00:00:00
    4/20/10
    2010-04-20 00:00:00
    9/18/10
    2010-09-18 00:00:00
    9/24/10
    2010-09-24 00:00:00
    3/15/10
    2010-03-15 00:00:00
    8/20/10
    2010-08-20 00:00:00
    8/20/10
    2010-08-20 00:00:00
    3/24/10
    2010-03-24 00:00:00
    9/28/10
    2010-09-28 00:00:00
    7/2/10
    2010-07-02 00:00:00
    10/1/10
    2010-10-01 00:00:00
    5/4/10
    2010-05-04 00:00:00
    1/26/10
    2010-01-26 00:00:00
    7/24/10
    2010-07-24 00:00:00
    1/1/10
    2010-01-01 00:00:00
    7/31/10
    2010-07-31 00:00:00
    2/14/10
    2010-02-14 00:00:00
    4/29/10
    2010-04-29 00:00:00
    10/27/10
    2010-10-27 00:00:00
    9/21/10
    2010-09-21 00:00:00
    3/1/10
    2010-03-01 00:00:00
    1/23/10
    2010-01-23 00:00:00
    9/14/10
    2010-09-14 00:00:00
    11/12/10
    2010-11-12 00:00:00
    5/22/10
    2010-05-22 00:00:00
    1/25/10
    2010-01-25 00:00:00
    1/7/10
    2010-01-07 00:00:00
    10/14/99
    1999-10-14 00:00:00
    3/30/99
    1999-03-30 00:00:00
    9/15/99
    1999-09-15 00:00:00
    5/19/99
    1999-05-19 00:00:00
    5/6/99
    1999-05-06 00:00:00
    12/10/99
    1999-12-10 00:00:00
    8/2/99
    1999-08-02 00:00:00
    11/8/99
    1999-11-08 00:00:00
    3/5/99
    1999-03-05 00:00:00
    11/18/99
    1999-11-18 00:00:00
    6/18/99
    1999-06-18 00:00:00
    7/9/99
    1999-07-09 00:00:00
    4/30/99
    1999-04-30 00:00:00
    10/30/99
    1999-10-30 00:00:00
    3/30/99
    1999-03-30 00:00:00
    5/13/99
    1999-05-13 00:00:00
    6/8/99
    1999-06-08 00:00:00
    9/2/99
    1999-09-02 00:00:00
    10/28/99
    1999-10-28 00:00:00
    12/23/99
    1999-12-23 00:00:00
    7/14/99
    1999-07-14 00:00:00
    9/30/99
    1999-09-30 00:00:00
    3/5/99
    1999-03-05 00:00:00
    12/17/99
    1999-12-17 00:00:00
    11/1/99
    1999-11-01 00:00:00
    6/29/99
    1999-06-29 00:00:00
    8/27/99
    1999-08-27 00:00:00
    11/23/99
    1999-11-23 00:00:00
    12/17/99
    1999-12-17 00:00:00
    8/6/99
    1999-08-06 00:00:00
    7/27/99
    1999-07-27 00:00:00
    6/30/99
    1999-06-30 00:00:00
    2/4/99
    1999-02-04 00:00:00
    7/30/99
    1999-07-30 00:00:00
    1/29/99
    1999-01-29 00:00:00
    9/24/99
    1999-09-24 00:00:00
    1/22/99
    1999-01-22 00:00:00
    2/19/99
    1999-02-19 00:00:00
    4/21/99
    1999-04-21 00:00:00
    12/17/99
    1999-12-17 00:00:00
    7/14/99
    1999-07-14 00:00:00
    8/24/99
    1999-08-24 00:00:00
    9/17/99
    1999-09-17 00:00:00
    4/16/99
    1999-04-16 00:00:00
    6/25/99
    1999-06-25 00:00:00
    10/22/99
    1999-10-22 00:00:00
    2/25/99
    1999-02-25 00:00:00
    11/12/99
    1999-11-12 00:00:00
    12/21/99
    1999-12-21 00:00:00
    12/7/99
    1999-12-07 00:00:00
    7/23/99
    1999-07-23 00:00:00
    3/12/99
    1999-03-12 00:00:00
    2/12/99
    1999-02-12 00:00:00
    3/26/99
    1999-03-26 00:00:00
    2/22/99
    1999-02-22 00:00:00
    6/18/99
    1999-06-18 00:00:00
    9/10/99
    1999-09-10 00:00:00
    9/10/99
    1999-09-10 00:00:00
    4/23/99
    1999-04-23 00:00:00
    7/23/99
    1999-07-23 00:00:00
    12/8/99
    1999-12-08 00:00:00
    11/5/99
    1999-11-05 00:00:00
    1/14/99
    1999-01-14 00:00:00
    12/25/99
    1999-12-25 00:00:00
    7/15/99
    1999-07-15 00:00:00
    8/13/99
    1999-08-13 00:00:00
    8/6/99
    1999-08-06 00:00:00
    9/13/99
    1999-09-13 00:00:00
    12/31/99
    1999-12-31 00:00:00
    4/9/99
    1999-04-09 00:00:00
    4/9/99
    1999-04-09 00:00:00
    6/4/99
    1999-06-04 00:00:00
    2/19/99
    1999-02-19 00:00:00
    9/27/99
    1999-09-27 00:00:00
    12/22/99
    1999-12-22 00:00:00
    8/13/99
    1999-08-13 00:00:00
    11/7/99
    1999-11-07 00:00:00
    10/29/99
    1999-10-29 00:00:00
    3/16/99
    1999-03-16 00:00:00
    4/9/99
    1999-04-09 00:00:00
    12/5/99
    1999-12-05 00:00:00
    8/6/99
    1999-08-06 00:00:00
    10/8/99
    1999-10-08 00:00:00
    4/14/99
    1999-04-14 00:00:00
    12/3/99
    1999-12-03 00:00:00
    9/17/99
    1999-09-17 00:00:00
    3/11/99
    1999-03-11 00:00:00
    3/19/99
    1999-03-19 00:00:00
    9/4/99
    1999-09-04 00:00:00
    8/21/99
    1999-08-21 00:00:00
    10/22/99
    1999-10-22 00:00:00
    3/12/99
    1999-03-12 00:00:00
    12/16/99
    1999-12-16 00:00:00
    12/25/99
    1999-12-25 00:00:00
    12/7/99
    1999-12-07 00:00:00
    8/5/99
    1999-08-05 00:00:00
    8/16/99
    1999-08-16 00:00:00
    7/9/99
    1999-07-09 00:00:00
    9/3/99
    1999-09-03 00:00:00
    5/16/99
    1999-05-16 00:00:00
    4/23/99
    1999-04-23 00:00:00
    10/30/99
    1999-10-30 00:00:00
    9/1/99
    1999-09-01 00:00:00
    9/13/99
    1999-09-13 00:00:00
    4/16/99
    1999-04-16 00:00:00
    11/12/99
    1999-11-12 00:00:00
    8/26/99
    1999-08-26 00:00:00
    2/11/99
    1999-02-11 00:00:00
    4/25/99
    1999-04-25 00:00:00
    5/15/99
    1999-05-15 00:00:00
    12/25/99
    1999-12-25 00:00:00
    10/13/99
    1999-10-13 00:00:00
    10/1/99
    1999-10-01 00:00:00
    10/7/99
    1999-10-07 00:00:00
    2/4/99
    1999-02-04 00:00:00
    7/14/99
    1999-07-14 00:00:00
    10/15/99
    1999-10-15 00:00:00
    4/15/99
    1999-04-15 00:00:00
    6/18/99
    1999-06-18 00:00:00
    9/16/99
    1999-09-16 00:00:00
    12/10/99
    1999-12-10 00:00:00
    4/17/99
    1999-04-17 00:00:00
    4/30/99
    1999-04-30 00:00:00
    12/16/99
    1999-12-16 00:00:00
    3/12/99
    1999-03-12 00:00:00
    9/25/99
    1999-09-25 00:00:00
    6/4/99
    1999-06-04 00:00:00
    8/25/99
    1999-08-25 00:00:00
    8/12/99
    1999-08-12 00:00:00
    8/11/99
    1999-08-11 00:00:00
    9/4/99
    1999-09-04 00:00:00
    7/2/99
    1999-07-02 00:00:00
    1/15/99
    1999-01-15 00:00:00
    1/1/99
    1999-01-01 00:00:00
    3/11/99
    1999-03-11 00:00:00
    6/11/99
    1999-06-11 00:00:00
    9/12/99
    1999-09-12 00:00:00
    1/15/99
    1999-01-15 00:00:00
    10/8/99
    1999-10-08 00:00:00
    3/19/99
    1999-03-19 00:00:00
    5/10/99
    1999-05-10 00:00:00
    7/23/99
    1999-07-23 00:00:00
    12/3/99
    1999-12-03 00:00:00
    12/22/99
    1999-12-22 00:00:00
    10/22/99
    1999-10-22 00:00:00
    2/5/99
    1999-02-05 00:00:00
    10/29/99
    1999-10-29 00:00:00
    11/15/99
    1999-11-15 00:00:00
    7/29/99
    1999-07-29 00:00:00
    2/17/99
    1999-02-17 00:00:00
    9/17/99
    1999-09-17 00:00:00
    10/1/99
    1999-10-01 00:00:00
    3/26/99
    1999-03-26 00:00:00
    2/28/99
    1999-02-28 00:00:00
    7/16/99
    1999-07-16 00:00:00
    11/12/99
    1999-11-12 00:00:00
    2/26/99
    1999-02-26 00:00:00
    3/16/99
    1999-03-16 00:00:00
    4/2/99
    1999-04-02 00:00:00
    1/22/99
    1999-01-22 00:00:00
    11/5/99
    1999-11-05 00:00:00
    9/4/99
    1999-09-04 00:00:00
    10/18/99
    1999-10-18 00:00:00
    2/26/99
    1999-02-26 00:00:00
    10/9/99
    1999-10-09 00:00:00
    6/26/99
    1999-06-26 00:00:00
    3/28/99
    1999-03-28 00:00:00
    4/1/99
    1999-04-01 00:00:00
    12/25/99
    1999-12-25 00:00:00
    10/3/99
    1999-10-03 00:00:00
    11/7/99
    1999-11-07 00:00:00
    12/3/99
    1999-12-03 00:00:00
    7/7/99
    1999-07-07 00:00:00
    6/14/99
    1999-06-14 00:00:00
    9/24/99
    1999-09-24 00:00:00
    11/24/99
    1999-11-24 00:00:00
    2/14/99
    1999-02-14 00:00:00
    1/30/99
    1999-01-30 00:00:00
    10/22/99
    1999-10-22 00:00:00
    9/3/99
    1999-09-03 00:00:00
    8/4/99
    1999-08-04 00:00:00
    11/26/99
    1999-11-26 00:00:00
    3/19/99
    1999-03-19 00:00:00
    1/3/99
    1999-01-03 00:00:00
    1/29/99
    1999-01-29 00:00:00
    6/5/99
    1999-06-05 00:00:00
    5/1/99
    1999-05-01 00:00:00
    8/16/99
    1999-08-16 00:00:00
    11/9/99
    1999-11-09 00:00:00
    6/18/99
    1999-06-18 00:00:00
    10/19/99
    1999-10-19 00:00:00
    2/13/99
    1999-02-13 00:00:00
    1/15/99
    1999-01-15 00:00:00
    10/7/99
    1999-10-07 00:00:00
    2/8/99
    1999-02-08 00:00:00
    12/6/99
    1999-12-06 00:00:00
    3/12/99
    1999-03-12 00:00:00
    8/27/99
    1999-08-27 00:00:00
    3/26/99
    1999-03-26 00:00:00
    11/12/99
    1999-11-12 00:00:00
    7/10/99
    1999-07-10 00:00:00
    7/23/99
    1999-07-23 00:00:00
    3/12/99
    1999-03-12 00:00:00
    1/1/99
    1999-01-01 00:00:00
    1/22/99
    1999-01-22 00:00:00
    12/25/99
    1999-12-25 00:00:00
    1/10/99
    1999-01-10 00:00:00
    5/15/99
    1999-05-15 00:00:00
    5/14/99
    1999-05-14 00:00:00
    4/23/99
    1999-04-23 00:00:00
    10/4/99
    1999-10-04 00:00:00
    7/24/99
    1999-07-24 00:00:00
    12/12/99
    1999-12-12 00:00:00
    9/6/99
    1999-09-06 00:00:00
    12/3/99
    1999-12-03 00:00:00
    3/25/99
    1999-03-25 00:00:00
    1/1/99
    1999-01-01 00:00:00
    2/6/99
    1999-02-06 00:00:00
    10/1/99
    1999-10-01 00:00:00
    3/30/99
    1999-03-30 00:00:00
    11/5/99
    1999-11-05 00:00:00
    2/14/99
    1999-02-14 00:00:00
    1/1/99
    1999-01-01 00:00:00
    4/30/99
    1999-04-30 00:00:00
    12/18/01
    2001-12-18 00:00:00
    11/16/01
    2001-11-16 00:00:00
    5/16/01
    2001-05-16 00:00:00
    1/18/01
    2001-01-18 00:00:00
    11/1/01
    2001-11-01 00:00:00
    6/29/01
    2001-06-29 00:00:00
    5/16/01
    2001-05-16 00:00:00
    4/13/01
    2001-04-13 00:00:00
    12/7/01
    2001-12-07 00:00:00
    5/21/01
    2001-05-21 00:00:00
    4/28/01
    2001-04-28 00:00:00
    6/2/01
    2001-06-02 00:00:00
    6/11/01
    2001-06-11 00:00:00
    12/11/01
    2001-12-11 00:00:00
    8/10/01
    2001-08-10 00:00:00
    8/3/01
    2001-08-03 00:00:00
    3/9/01
    2001-03-09 00:00:00
    7/25/01
    2001-07-25 00:00:00
    10/5/01
    2001-10-05 00:00:00
    4/4/01
    2001-04-04 00:00:00
    3/13/01
    2001-03-13 00:00:00
    12/28/01
    2001-12-28 00:00:00
    7/13/01
    2001-07-13 00:00:00
    5/11/01
    2001-05-11 00:00:00
    9/28/01
    2001-09-28 00:00:00
    12/7/01
    2001-12-07 00:00:00
    7/4/01
    2001-07-04 00:00:00
    8/3/01
    2001-08-03 00:00:00
    10/5/01
    2001-10-05 00:00:00
    8/2/01
    2001-08-02 00:00:00
    12/21/01
    2001-12-21 00:00:00
    8/22/01
    2001-08-22 00:00:00
    10/5/01
    2001-10-05 00:00:00
    7/4/01
    2001-07-04 00:00:00
    12/10/01
    2001-12-10 00:00:00
    12/31/01
    2001-12-31 00:00:00
    7/6/01
    2001-07-06 00:00:00
    11/18/01
    2001-11-18 00:00:00
    6/7/01
    2001-06-07 00:00:00
    11/1/01
    2001-11-01 00:00:00
    9/30/01
    2001-09-30 00:00:00
    2/18/01
    2001-02-18 00:00:00
    1/12/01
    2001-01-12 00:00:00
    7/2/01
    2001-07-02 00:00:00
    1/23/01
    2001-01-23 00:00:00
    8/3/01
    2001-08-03 00:00:00
    6/22/01
    2001-06-22 00:00:00
    11/17/01
    2001-11-17 00:00:00
    7/1/01
    2001-07-01 00:00:00
    12/25/01
    2001-12-25 00:00:00
    7/17/01
    2001-07-17 00:00:00
    10/19/01
    2001-10-19 00:00:00
    10/19/01
    2001-10-19 00:00:00
    4/6/01
    2001-04-06 00:00:00
    1/18/01
    2001-01-18 00:00:00
    1/26/01
    2001-01-26 00:00:00
    7/30/01
    2001-07-30 00:00:00
    11/7/01
    2001-11-07 00:00:00
    7/24/01
    2001-07-24 00:00:00
    3/1/01
    2001-03-01 00:00:00
    10/5/01
    2001-10-05 00:00:00
    12/28/01
    2001-12-28 00:00:00
    4/20/01
    2001-04-20 00:00:00
    5/11/01
    2001-05-11 00:00:00
    10/2/01
    2001-10-02 00:00:00
    4/19/01
    2001-04-19 00:00:00
    9/4/01
    2001-09-04 00:00:00
    11/15/01
    2001-11-15 00:00:00
    10/22/01
    2001-10-22 00:00:00
    3/30/01
    2001-03-30 00:00:00
    3/1/01
    2001-03-01 00:00:00
    9/28/01
    2001-09-28 00:00:00
    10/26/01
    2001-10-26 00:00:00
    4/12/01
    2001-04-12 00:00:00
    11/17/01
    2001-11-17 00:00:00
    11/6/01
    2001-11-06 00:00:00
    5/15/01
    2001-05-15 00:00:00
    10/25/01
    2001-10-25 00:00:00
    12/18/01
    2001-12-18 00:00:00
    9/7/01
    2001-09-07 00:00:00
    7/20/01
    2001-07-20 00:00:00
    9/14/01
    2001-09-14 00:00:00
    3/1/01
    2001-03-01 00:00:00
    8/31/01
    2001-08-31 00:00:00
    9/14/01
    2001-09-14 00:00:00
    6/8/01
    2001-06-08 00:00:00
    3/8/01
    2001-03-08 00:00:00
    3/23/01
    2001-03-23 00:00:00
    12/11/01
    2001-12-11 00:00:00
    9/9/01
    2001-09-09 00:00:00
    8/10/01
    2001-08-10 00:00:00
    7/13/01
    2001-07-13 00:00:00
    11/11/01
    2001-11-11 00:00:00
    4/10/01
    2001-04-10 00:00:00
    12/4/01
    2001-12-04 00:00:00
    3/13/01
    2001-03-13 00:00:00
    2/11/01
    2001-02-11 00:00:00
    2/16/01
    2001-02-16 00:00:00
    12/21/01
    2001-12-21 00:00:00
    2/16/01
    2001-02-16 00:00:00
    3/30/01
    2001-03-30 00:00:00
    3/16/01
    2001-03-16 00:00:00
    12/21/01
    2001-12-21 00:00:00
    2/23/01
    2001-02-23 00:00:00
    7/15/01
    2001-07-15 00:00:00
    1/26/01
    2001-01-26 00:00:00
    10/23/01
    2001-10-23 00:00:00
    5/31/01
    2001-05-31 00:00:00
    7/24/01
    2001-07-24 00:00:00
    8/24/01
    2001-08-24 00:00:00
    2/9/01
    2001-02-09 00:00:00
    8/5/01
    2001-08-05 00:00:00
    10/19/01
    2001-10-19 00:00:00
    7/17/01
    2001-07-17 00:00:00
    8/7/01
    2001-08-07 00:00:00
    5/13/01
    2001-05-13 00:00:00
    11/2/01
    2001-11-02 00:00:00
    9/8/01
    2001-09-08 00:00:00
    2/2/01
    2001-02-02 00:00:00
    1/22/01
    2001-01-22 00:00:00
    6/22/01
    2001-06-22 00:00:00
    5/18/01
    2001-05-18 00:00:00
    1/12/01
    2001-01-12 00:00:00
    11/20/01
    2001-11-20 00:00:00
    6/15/01
    2001-06-15 00:00:00
    10/4/01
    2001-10-04 00:00:00
    11/23/01
    2001-11-23 00:00:00
    3/18/01
    2001-03-18 00:00:00
    1/9/01
    2001-01-09 00:00:00
    7/20/01
    2001-07-20 00:00:00
    12/26/01
    2001-12-26 00:00:00
    10/12/01
    2001-10-12 00:00:00
    6/1/01
    2001-06-01 00:00:00
    9/7/01
    2001-09-07 00:00:00
    11/19/01
    2001-11-19 00:00:00
    4/6/01
    2001-04-06 00:00:00
    6/29/01
    2001-06-29 00:00:00
    9/8/01
    2001-09-08 00:00:00
    5/11/01
    2001-05-11 00:00:00
    4/27/01
    2001-04-27 00:00:00
    11/21/01
    2001-11-21 00:00:00
    1/20/01
    2001-01-20 00:00:00
    11/1/01
    2001-11-01 00:00:00
    3/11/01
    2001-03-11 00:00:00
    8/17/01
    2001-08-17 00:00:00
    2/8/01
    2001-02-08 00:00:00
    10/30/01
    2001-10-30 00:00:00
    9/14/01
    2001-09-14 00:00:00
    9/6/01
    2001-09-06 00:00:00
    1/27/01
    2001-01-27 00:00:00
    11/1/01
    2001-11-01 00:00:00
    12/3/01
    2001-12-03 00:00:00
    6/12/01
    2001-06-12 00:00:00
    6/29/01
    2001-06-29 00:00:00
    10/4/01
    2001-10-04 00:00:00
    6/22/01
    2001-06-22 00:00:00
    11/30/01
    2001-11-30 00:00:00
    7/13/01
    2001-07-13 00:00:00
    12/17/01
    2001-12-17 00:00:00
    12/4/01
    2001-12-04 00:00:00
    8/22/01
    2001-08-22 00:00:00
    7/27/01
    2001-07-27 00:00:00
    2/1/01
    2001-02-01 00:00:00
    1/1/01
    2001-01-01 00:00:00
    11/9/01
    2001-11-09 00:00:00
    4/4/01
    2001-04-04 00:00:00
    12/2/01
    2001-12-02 00:00:00
    1/19/01
    2001-01-19 00:00:00
    10/5/01
    2001-10-05 00:00:00
    2/23/01
    2001-02-23 00:00:00
    3/2/01
    2001-03-02 00:00:00
    11/8/01
    2001-11-08 00:00:00
    8/31/01
    2001-08-31 00:00:00
    4/28/01
    2001-04-28 00:00:00
    1/1/01
    2001-01-01 00:00:00
    6/29/01
    2001-06-29 00:00:00
    8/15/01
    2001-08-15 00:00:00
    10/12/01
    2001-10-12 00:00:00
    12/21/01
    2001-12-21 00:00:00
    9/2/01
    2001-09-02 00:00:00
    10/28/01
    2001-10-28 00:00:00
    12/14/01
    2001-12-14 00:00:00
    1/1/01
    2001-01-01 00:00:00
    11/21/01
    2001-11-21 00:00:00
    10/25/01
    2001-10-25 00:00:00
    2/9/01
    2001-02-09 00:00:00
    10/9/01
    2001-10-09 00:00:00
    7/18/01
    2001-07-18 00:00:00
    9/13/01
    2001-09-13 00:00:00
    4/6/01
    2001-04-06 00:00:00
    8/17/01
    2001-08-17 00:00:00
    4/13/01
    2001-04-13 00:00:00
    12/10/01
    2001-12-10 00:00:00
    11/15/01
    2001-11-15 00:00:00
    8/4/01
    2001-08-04 00:00:00
    4/18/01
    2001-04-18 00:00:00
    11/24/01
    2001-11-24 00:00:00
    1/1/01
    2001-01-01 00:00:00
    1/12/01
    2001-01-12 00:00:00
    4/21/01
    2001-04-21 00:00:00
    5/2/01
    2001-05-02 00:00:00
    2/16/01
    2001-02-16 00:00:00
    9/14/01
    2001-09-14 00:00:00
    9/7/01
    2001-09-07 00:00:00
    8/3/01
    2001-08-03 00:00:00
    4/27/01
    2001-04-27 00:00:00
    12/2/01
    2001-12-02 00:00:00
    1/20/01
    2001-01-20 00:00:00
    3/12/01
    2001-03-12 00:00:00
    8/24/01
    2001-08-24 00:00:00
    9/12/01
    2001-09-12 00:00:00
    5/19/01
    2001-05-19 00:00:00
    11/23/01
    2001-11-23 00:00:00
    1/24/01
    2001-01-24 00:00:00
    3/27/01
    2001-03-27 00:00:00
    10/14/01
    2001-10-14 00:00:00
    10/24/01
    2001-10-24 00:00:00
    1/18/01
    2001-01-18 00:00:00
    5/12/01
    2001-05-12 00:00:00
    5/15/01
    2001-05-15 00:00:00
    2/2/01
    2001-02-02 00:00:00
    11/16/01
    2001-11-16 00:00:00
    5/11/01
    2001-05-11 00:00:00
    7/13/01
    2001-07-13 00:00:00
    9/6/01
    2001-09-06 00:00:00
    10/12/01
    2001-10-12 00:00:00
    10/21/01
    2001-10-21 00:00:00
    10/12/01
    2001-10-12 00:00:00
    10/9/01
    2001-10-09 00:00:00
    11/10/01
    2001-11-10 00:00:00
    4/4/01
    2001-04-04 00:00:00
    1/21/01
    2001-01-21 00:00:00
    6/27/01
    2001-06-27 00:00:00
    12/25/01
    2001-12-25 00:00:00
    1/27/01
    2001-01-27 00:00:00
    3/9/01
    2001-03-09 00:00:00
    3/9/01
    2001-03-09 00:00:00
    11/21/01
    2001-11-21 00:00:00
    1/12/01
    2001-01-12 00:00:00
    2/13/01
    2001-02-13 00:00:00
    10/17/01
    2001-10-17 00:00:00
    1/1/01
    2001-01-01 00:00:00
    7/16/08
    2008-07-16 00:00:00
    6/22/08
    2008-06-22 00:00:00
    4/30/08
    2008-04-30 00:00:00
    2/18/08
    2008-02-18 00:00:00
    5/21/08
    2008-05-21 00:00:00
    11/20/08
    2008-11-20 00:00:00
    10/30/08
    2008-10-30 00:00:00
    6/4/08
    2008-06-04 00:00:00
    10/16/08
    2008-10-16 00:00:00
    6/30/08
    2008-06-30 00:00:00
    11/24/08
    2008-11-24 00:00:00
    7/1/08
    2008-07-01 00:00:00
    6/12/08
    2008-06-12 00:00:00
    12/9/08
    2008-12-09 00:00:00
    10/30/08
    2008-10-30 00:00:00
    7/11/08
    2008-07-11 00:00:00
    2/10/08
    2008-02-10 00:00:00
    7/10/08
    2008-07-10 00:00:00
    6/12/08
    2008-06-12 00:00:00
    10/27/08
    2008-10-27 00:00:00
    5/12/08
    2008-05-12 00:00:00
    11/26/08
    2008-11-26 00:00:00
    12/25/08
    2008-12-25 00:00:00
    2/22/08
    2008-02-22 00:00:00
    7/1/08
    2008-07-01 00:00:00
    10/10/08
    2008-10-10 00:00:00
    5/7/08
    2008-05-07 00:00:00
    5/15/08
    2008-05-15 00:00:00
    8/9/08
    2008-08-09 00:00:00
    9/25/08
    2008-09-25 00:00:00
    5/7/08
    2008-05-07 00:00:00
    1/24/08
    2008-01-24 00:00:00
    8/22/08
    2008-08-22 00:00:00
    2/14/08
    2008-02-14 00:00:00
    12/10/08
    2008-12-10 00:00:00
    7/19/08
    2008-07-19 00:00:00
    12/18/08
    2008-12-18 00:00:00
    1/15/08
    2008-01-15 00:00:00
    11/20/08
    2008-11-20 00:00:00
    4/17/08
    2008-04-17 00:00:00
    11/18/08
    2008-11-18 00:00:00
    12/9/08
    2008-12-09 00:00:00
    12/25/08
    2008-12-25 00:00:00
    10/10/08
    2008-10-10 00:00:00
    9/5/08
    2008-09-05 00:00:00
    1/10/08
    2008-01-10 00:00:00
    12/11/08
    2008-12-11 00:00:00
    3/27/08
    2008-03-27 00:00:00
    2/28/08
    2008-02-28 00:00:00
    7/25/08
    2008-07-25 00:00:00
    2/8/08
    2008-02-08 00:00:00
    2/7/08
    2008-02-07 00:00:00
    3/4/08
    2008-03-04 00:00:00
    8/6/08
    2008-08-06 00:00:00
    9/7/08
    2008-09-07 00:00:00
    12/24/08
    2008-12-24 00:00:00
    12/5/08
    2008-12-05 00:00:00
    10/16/08
    2008-10-16 00:00:00
    8/15/08
    2008-08-15 00:00:00
    2/1/08
    2008-02-01 00:00:00
    9/18/08
    2008-09-18 00:00:00
    6/19/08
    2008-06-19 00:00:00
    2/3/08
    2008-02-03 00:00:00
    7/19/08
    2008-07-19 00:00:00
    10/15/08
    2008-10-15 00:00:00
    10/22/08
    2008-10-22 00:00:00
    12/10/08
    2008-12-10 00:00:00
    8/20/08
    2008-08-20 00:00:00
    10/14/08
    2008-10-14 00:00:00
    2/8/08
    2008-02-08 00:00:00
    1/30/08
    2008-01-30 00:00:00
    2/14/08
    2008-02-14 00:00:00
    4/25/08
    2008-04-25 00:00:00
    2/29/08
    2008-02-29 00:00:00
    1/1/08
    2008-01-01 00:00:00
    7/17/08
    2008-07-17 00:00:00
    2/5/08
    2008-02-05 00:00:00
    11/8/08
    2008-11-08 00:00:00
    4/10/08
    2008-04-10 00:00:00
    10/3/08
    2008-10-03 00:00:00
    11/26/08
    2008-11-26 00:00:00
    9/16/08
    2008-09-16 00:00:00
    3/11/08
    2008-03-11 00:00:00
    8/22/08
    2008-08-22 00:00:00
    6/18/08
    2008-06-18 00:00:00
    8/5/08
    2008-08-05 00:00:00
    10/23/08
    2008-10-23 00:00:00
    6/20/08
    2008-06-20 00:00:00
    5/9/08
    2008-05-09 00:00:00
    8/15/08
    2008-08-15 00:00:00
    3/3/08
    2008-03-03 00:00:00
    9/5/08
    2008-09-05 00:00:00
    4/18/08
    2008-04-18 00:00:00
    4/18/08
    2008-04-18 00:00:00
    9/5/08
    2008-09-05 00:00:00
    11/17/08
    2008-11-17 00:00:00
    9/11/08
    2008-09-11 00:00:00
    1/18/08
    2008-01-18 00:00:00
    8/23/08
    2008-08-23 00:00:00
    6/5/08
    2008-06-05 00:00:00
    12/31/08
    2008-12-31 00:00:00
    9/7/08
    2008-09-07 00:00:00
    1/22/08
    2008-01-22 00:00:00
    1/19/08
    2008-01-19 00:00:00
    6/6/08
    2008-06-06 00:00:00
    5/29/08
    2008-05-29 00:00:00
    7/29/08
    2008-07-29 00:00:00
    11/26/08
    2008-11-26 00:00:00
    4/24/08
    2008-04-24 00:00:00
    7/24/08
    2008-07-24 00:00:00
    11/27/08
    2008-11-27 00:00:00
    10/3/08
    2008-10-03 00:00:00
    12/3/08
    2008-12-03 00:00:00
    10/7/08
    2008-10-07 00:00:00
    5/2/08
    2008-05-02 00:00:00
    2/28/08
    2008-02-28 00:00:00
    10/21/08
    2008-10-21 00:00:00
    4/24/08
    2008-04-24 00:00:00
    9/4/08
    2008-09-04 00:00:00
    2/1/08
    2008-02-01 00:00:00
    8/8/08
    2008-08-08 00:00:00
    8/21/08
    2008-08-21 00:00:00
    1/24/08
    2008-01-24 00:00:00
    3/28/08
    2008-03-28 00:00:00
    3/20/08
    2008-03-20 00:00:00
    10/24/08
    2008-10-24 00:00:00
    5/26/08
    2008-05-26 00:00:00
    5/14/08
    2008-05-14 00:00:00
    12/25/08
    2008-12-25 00:00:00
    1/14/08
    2008-01-14 00:00:00
    12/19/08
    2008-12-19 00:00:00
    9/12/08
    2008-09-12 00:00:00
    9/26/08
    2008-09-26 00:00:00
    11/5/08
    2008-11-05 00:00:00
    6/11/08
    2008-06-11 00:00:00
    3/14/08
    2008-03-14 00:00:00
    9/6/08
    2008-09-06 00:00:00
    10/7/08
    2008-10-07 00:00:00
    10/9/08
    2008-10-09 00:00:00
    6/4/08
    2008-06-04 00:00:00
    12/18/08
    2008-12-18 00:00:00
    3/4/08
    2008-03-04 00:00:00
    5/12/08
    2008-05-12 00:00:00
    8/20/08
    2008-08-20 00:00:00
    2/8/08
    2008-02-08 00:00:00
    10/29/08
    2008-10-29 00:00:00
    4/18/08
    2008-04-18 00:00:00
    9/22/08
    2008-09-22 00:00:00
    3/27/08
    2008-03-27 00:00:00
    9/17/08
    2008-09-17 00:00:00
    7/22/08
    2008-07-22 00:00:00
    12/25/08
    2008-12-25 00:00:00
    10/3/08
    2008-10-03 00:00:00
    7/11/08
    2008-07-11 00:00:00
    12/19/08
    2008-12-19 00:00:00
    1/20/08
    2008-01-20 00:00:00
    1/18/08
    2008-01-18 00:00:00
    6/24/08
    2008-06-24 00:00:00
    5/10/08
    2008-05-10 00:00:00
    4/2/08
    2008-04-02 00:00:00
    6/20/08
    2008-06-20 00:00:00
    9/19/08
    2008-09-19 00:00:00
    5/15/08
    2008-05-15 00:00:00
    9/10/08
    2008-09-10 00:00:00
    10/27/08
    2008-10-27 00:00:00
    8/12/08
    2008-08-12 00:00:00
    11/3/08
    2008-11-03 00:00:00
    12/4/08
    2008-12-04 00:00:00
    3/7/08
    2008-03-07 00:00:00
    9/9/08
    2008-09-09 00:00:00
    10/31/08
    2008-10-31 00:00:00
    11/16/08
    2008-11-16 00:00:00
    9/5/08
    2008-09-05 00:00:00
    4/23/08
    2008-04-23 00:00:00
    3/23/08
    2008-03-23 00:00:00
    8/28/08
    2008-08-28 00:00:00
    9/7/08
    2008-09-07 00:00:00
    2/10/08
    2008-02-10 00:00:00
    7/25/08
    2008-07-25 00:00:00
    3/21/08
    2008-03-21 00:00:00
    3/7/08
    2008-03-07 00:00:00
    9/9/08
    2008-09-09 00:00:00
    7/8/08
    2008-07-08 00:00:00
    7/29/08
    2008-07-29 00:00:00
    10/10/08
    2008-10-10 00:00:00
    3/14/08
    2008-03-14 00:00:00
    2/26/08
    2008-02-26 00:00:00
    4/11/08
    2008-04-11 00:00:00
    1/4/08
    2008-01-04 00:00:00
    9/6/08
    2008-09-06 00:00:00
    7/25/08
    2008-07-25 00:00:00
    9/12/08
    2008-09-12 00:00:00
    6/20/08
    2008-06-20 00:00:00
    4/13/08
    2008-04-13 00:00:00
    1/17/08
    2008-01-17 00:00:00
    1/1/08
    2008-01-01 00:00:00
    8/24/08
    2008-08-24 00:00:00
    9/24/08
    2008-09-24 00:00:00
    2/7/08
    2008-02-07 00:00:00
    7/3/08
    2008-07-03 00:00:00
    11/19/08
    2008-11-19 00:00:00
    9/26/08
    2008-09-26 00:00:00
    8/6/08
    2008-08-06 00:00:00
    10/10/08
    2008-10-10 00:00:00
    2/1/08
    2008-02-01 00:00:00
    2/8/08
    2008-02-08 00:00:00
    9/12/08
    2008-09-12 00:00:00
    2/7/08
    2008-02-07 00:00:00
    7/13/08
    2008-07-13 00:00:00
    11/13/08
    2008-11-13 00:00:00
    6/6/08
    2008-06-06 00:00:00
    9/21/08
    2008-09-21 00:00:00
    9/26/08
    2008-09-26 00:00:00
    2/1/08
    2008-02-01 00:00:00
    7/17/08
    2008-07-17 00:00:00
    6/6/08
    2008-06-06 00:00:00
    9/19/08
    2008-09-19 00:00:00
    12/6/08
    2008-12-06 00:00:00
    8/1/08
    2008-08-01 00:00:00
    11/7/08
    2008-11-07 00:00:00
    5/25/08
    2008-05-25 00:00:00
    10/9/08
    2008-10-09 00:00:00
    4/10/08
    2008-04-10 00:00:00
    4/8/08
    2008-04-08 00:00:00
    10/18/08
    2008-10-18 00:00:00
    7/4/08
    2008-07-04 00:00:00
    9/30/08
    2008-09-30 00:00:00
    7/29/08
    2008-07-29 00:00:00
    7/1/08
    2008-07-01 00:00:00
    1/1/08
    2008-01-01 00:00:00
    1/18/08
    2008-01-18 00:00:00
    9/2/08
    2008-09-02 00:00:00
    1/1/08
    2008-01-01 00:00:00
    8/19/08
    2008-08-19 00:00:00
    2/11/08
    2008-02-11 00:00:00
    2/15/08
    2008-02-15 00:00:00
    4/10/08
    2008-04-10 00:00:00
    1/1/08
    2008-01-01 00:00:00
    10/2/08
    2008-10-02 00:00:00
    9/26/08
    2008-09-26 00:00:00
    8/30/08
    2008-08-30 00:00:00
    4/28/08
    2008-04-28 00:00:00
    7/3/08
    2008-07-03 00:00:00
    10/17/08
    2008-10-17 00:00:00
    8/8/08
    2008-08-08 00:00:00
    12/12/08
    2008-12-12 00:00:00
    12/7/08
    2008-12-07 00:00:00
    12/3/08
    2008-12-03 00:00:00
    10/16/08
    2008-10-16 00:00:00
    6/20/08
    2008-06-20 00:00:00
    1/1/08
    2008-01-01 00:00:00
    5/28/08
    2008-05-28 00:00:00
    1/11/08
    2008-01-11 00:00:00
    10/1/08
    2008-10-01 00:00:00
    12/5/08
    2008-12-05 00:00:00
    12/14/08
    2008-12-14 00:00:00
    3/21/08
    2008-03-21 00:00:00
    2/12/08
    2008-02-12 00:00:00
    1/17/08
    2008-01-17 00:00:00
    8/29/08
    2008-08-29 00:00:00
    7/25/08
    2008-07-25 00:00:00
    5/31/08
    2008-05-31 00:00:00
    9/15/08
    2008-09-15 00:00:00
    10/28/08
    2008-10-28 00:00:00
    9/7/08
    2008-09-07 00:00:00
    3/4/08
    2008-03-04 00:00:00
    7/13/08
    2008-07-13 00:00:00
    10/9/08
    2008-10-09 00:00:00
    9/19/08
    2008-09-19 00:00:00
    1/19/08
    2008-01-19 00:00:00
    12/5/08
    2008-12-05 00:00:00
    8/25/08
    2008-08-25 00:00:00
    8/7/08
    2008-08-07 00:00:00
    7/18/08
    2008-07-18 00:00:00
    8/7/08
    2008-08-07 00:00:00
    4/28/08
    2008-04-28 00:00:00
    4/3/08
    2008-04-03 00:00:00
    11/29/08
    2008-11-29 00:00:00
    1/1/08
    2008-01-01 00:00:00
    6/1/08
    2008-06-01 00:00:00
    11/7/08
    2008-11-07 00:00:00
    5/1/08
    2008-05-01 00:00:00
    10/1/08
    2008-10-01 00:00:00
    8/1/08
    2008-08-01 00:00:00
    10/17/08
    2008-10-17 00:00:00
    2/15/08
    2008-02-15 00:00:00
    1/1/08
    2008-01-01 00:00:00
    7/15/08
    2008-07-15 00:00:00
    10/14/08
    2008-10-14 00:00:00
    9/26/08
    2008-09-26 00:00:00
    1/1/08
    2008-01-01 00:00:00
    6/24/08
    2008-06-24 00:00:00
    9/7/08
    2008-09-07 00:00:00
    9/6/08
    2008-09-06 00:00:00
    10/17/08
    2008-10-17 00:00:00
    7/16/08
    2008-07-16 00:00:00
    1/1/08
    2008-01-01 00:00:00
    8/15/08
    2008-08-15 00:00:00
    10/24/08
    2008-10-24 00:00:00
    2/8/08
    2008-02-08 00:00:00
    9/9/08
    2008-09-09 00:00:00
    10/17/08
    2008-10-17 00:00:00
    4/30/08
    2008-04-30 00:00:00
    5/15/08
    2008-05-15 00:00:00
    8/6/08
    2008-08-06 00:00:00
    9/26/08
    2008-09-26 00:00:00
    10/29/08
    2008-10-29 00:00:00
    9/27/08
    2008-09-27 00:00:00
    4/11/08
    2008-04-11 00:00:00
    10/3/08
    2008-10-03 00:00:00
    9/26/08
    2008-09-26 00:00:00
    3/5/08
    2008-03-05 00:00:00
    2/22/08
    2008-02-22 00:00:00
    2/9/08
    2008-02-09 00:00:00
    8/15/08
    2008-08-15 00:00:00
    4/7/08
    2008-04-07 00:00:00
    5/20/08
    2008-05-20 00:00:00
    4/30/08
    2008-04-30 00:00:00
    4/3/08
    2008-04-03 00:00:00
    10/24/08
    2008-10-24 00:00:00
    4/30/08
    2008-04-30 00:00:00
    8/8/08
    2008-08-08 00:00:00
    11/6/08
    2008-11-06 00:00:00
    9/19/08
    2008-09-19 00:00:00
    12/5/08
    2008-12-05 00:00:00
    2/7/08
    2008-02-07 00:00:00
    1/1/08
    2008-01-01 00:00:00
    6/16/08
    2008-06-16 00:00:00
    2/28/08
    2008-02-28 00:00:00
    10/31/08
    2008-10-31 00:00:00
    12/25/08
    2008-12-25 00:00:00
    10/2/08
    2008-10-02 00:00:00
    2/2/08
    2008-02-02 00:00:00
    10/24/08
    2008-10-24 00:00:00
    11/6/08
    2008-11-06 00:00:00
    5/28/08
    2008-05-28 00:00:00
    8/7/08
    2008-08-07 00:00:00
    9/7/08
    2008-09-07 00:00:00
    1/1/08
    2008-01-01 00:00:00
    10/16/08
    2008-10-16 00:00:00
    11/17/08
    2008-11-17 00:00:00
    10/4/08
    2008-10-04 00:00:00
    10/27/08
    2008-10-27 00:00:00
    5/22/08
    2008-05-22 00:00:00
    1/18/08
    2008-01-18 00:00:00
    11/23/08
    2008-11-23 00:00:00
    12/23/08
    2008-12-23 00:00:00
    1/18/08
    2008-01-18 00:00:00
    8/20/08
    2008-08-20 00:00:00
    7/1/08
    2008-07-01 00:00:00
    2/8/08
    2008-02-08 00:00:00
    9/8/08
    2008-09-08 00:00:00
    11/16/08
    2008-11-16 00:00:00
    2/6/08
    2008-02-06 00:00:00
    3/16/08
    2008-03-16 00:00:00
    3/7/08
    2008-03-07 00:00:00
    10/30/08
    2008-10-30 00:00:00
    7/2/08
    2008-07-02 00:00:00
    1/25/08
    2008-01-25 00:00:00
    9/25/08
    2008-09-25 00:00:00
    12/25/08
    2008-12-25 00:00:00
    9/24/08
    2008-09-24 00:00:00
    5/16/08
    2008-05-16 00:00:00
    10/2/08
    2008-10-02 00:00:00
    7/15/08
    2008-07-15 00:00:00
    10/6/08
    2008-10-06 00:00:00
    9/4/08
    2008-09-04 00:00:00
    5/14/08
    2008-05-14 00:00:00
    7/4/08
    2008-07-04 00:00:00
    1/11/08
    2008-01-11 00:00:00
    2/8/08
    2008-02-08 00:00:00
    7/15/08
    2008-07-15 00:00:00
    1/1/08
    2008-01-01 00:00:00
    5/31/08
    2008-05-31 00:00:00
    6/4/08
    2008-06-04 00:00:00
    1/1/08
    2008-01-01 00:00:00
    5/18/08
    2008-05-18 00:00:00
    2/11/08
    2008-02-11 00:00:00
    10/7/08
    2008-10-07 00:00:00
    1/1/08
    2008-01-01 00:00:00
    3/28/08
    2008-03-28 00:00:00
    8/24/08
    2008-08-24 00:00:00
    8/6/08
    2008-08-06 00:00:00
    11/5/08
    2008-11-05 00:00:00
    10/6/08
    2008-10-06 00:00:00
    9/13/08
    2008-09-13 00:00:00
    1/21/08
    2008-01-21 00:00:00
    7/1/08
    2008-07-01 00:00:00
    9/20/08
    2008-09-20 00:00:00
    1/19/08
    2008-01-19 00:00:00
    9/24/08
    2008-09-24 00:00:00
    10/1/08
    2008-10-01 00:00:00
    7/18/08
    2008-07-18 00:00:00
    12/5/08
    2008-12-05 00:00:00
    11/21/08
    2008-11-21 00:00:00
    6/20/08
    2008-06-20 00:00:00
    2/8/08
    2008-02-08 00:00:00
    2/9/08
    2008-02-09 00:00:00
    1/21/08
    2008-01-21 00:00:00
    8/22/08
    2008-08-22 00:00:00
    7/4/08
    2008-07-04 00:00:00
    7/18/08
    2008-07-18 00:00:00
    11/9/08
    2008-11-09 00:00:00
    1/18/08
    2008-01-18 00:00:00
    8/13/08
    2008-08-13 00:00:00
    3/8/08
    2008-03-08 00:00:00
    2/3/08
    2008-02-03 00:00:00
    12/18/08
    2008-12-18 00:00:00
    2/14/08
    2008-02-14 00:00:00
    6/12/08
    2008-06-12 00:00:00
    8/26/08
    2008-08-26 00:00:00
    6/13/08
    2008-06-13 00:00:00
    3/20/08
    2008-03-20 00:00:00
    2/7/08
    2008-02-07 00:00:00
    4/30/08
    2008-04-30 00:00:00
    12/25/08
    2008-12-25 00:00:00
    12/10/08
    2008-12-10 00:00:00
    3/10/08
    2008-03-10 00:00:00
    11/11/08
    2008-11-11 00:00:00
    1/1/08
    2008-01-01 00:00:00
    9/1/08
    2008-09-01 00:00:00
    12/17/08
    2008-12-17 00:00:00
    2/22/08
    2008-02-22 00:00:00
    10/7/08
    2008-10-07 00:00:00
    1/1/08
    2008-01-01 00:00:00
    8/27/08
    2008-08-27 00:00:00
    10/17/08
    2008-10-17 00:00:00
    2/11/08
    2008-02-11 00:00:00
    10/1/08
    2008-10-01 00:00:00
    5/30/08
    2008-05-30 00:00:00
    1/21/08
    2008-01-21 00:00:00
    1/1/08
    2008-01-01 00:00:00
    10/7/08
    2008-10-07 00:00:00
    3/20/08
    2008-03-20 00:00:00
    11/11/08
    2008-11-11 00:00:00
    10/31/08
    2008-10-31 00:00:00
    8/15/08
    2008-08-15 00:00:00
    9/18/08
    2008-09-18 00:00:00
    8/13/08
    2008-08-13 00:00:00
    11/12/08
    2008-11-12 00:00:00
    3/25/08
    2008-03-25 00:00:00
    1/29/08
    2008-01-29 00:00:00
    3/20/08
    2008-03-20 00:00:00
    2/8/08
    2008-02-08 00:00:00
    12/11/08
    2008-12-11 00:00:00
    7/13/08
    2008-07-13 00:00:00
    8/22/08
    2008-08-22 00:00:00
    1/20/08
    2008-01-20 00:00:00
    12/5/08
    2008-12-05 00:00:00
    10/1/08
    2008-10-01 00:00:00
    10/27/08
    2008-10-27 00:00:00
    9/9/08
    2008-09-09 00:00:00
    10/3/08
    2008-10-03 00:00:00
    6/24/08
    2008-06-24 00:00:00
    3/7/08
    2008-03-07 00:00:00
    8/28/08
    2008-08-28 00:00:00
    9/12/08
    2008-09-12 00:00:00
    1/14/08
    2008-01-14 00:00:00
    9/13/08
    2008-09-13 00:00:00
    4/20/08
    2008-04-20 00:00:00
    8/5/08
    2008-08-05 00:00:00
    8/2/08
    2008-08-02 00:00:00
    11/8/08
    2008-11-08 00:00:00
    1/1/08
    2008-01-01 00:00:00
    10/2/08
    2008-10-02 00:00:00
    4/18/08
    2008-04-18 00:00:00
    2/9/08
    2008-02-09 00:00:00
    8/22/08
    2008-08-22 00:00:00
    12/6/08
    2008-12-06 00:00:00
    6/17/08
    2008-06-17 00:00:00
    3/1/08
    2008-03-01 00:00:00
    7/10/08
    2008-07-10 00:00:00
    10/17/08
    2008-10-17 00:00:00
    5/20/08
    2008-05-20 00:00:00
    1/1/08
    2008-01-01 00:00:00
    1/1/08
    2008-01-01 00:00:00
    9/8/08
    2008-09-08 00:00:00
    1/4/08
    2008-01-04 00:00:00
    10/15/08
    2008-10-15 00:00:00
    3/11/08
    2008-03-11 00:00:00
    12/12/08
    2008-12-12 00:00:00
    3/16/08
    2008-03-16 00:00:00
    9/23/08
    2008-09-23 00:00:00
    2/7/08
    2008-02-07 00:00:00
    8/22/08
    2008-08-22 00:00:00
    9/6/08
    2008-09-06 00:00:00
    7/25/08
    2008-07-25 00:00:00
    8/26/08
    2008-08-26 00:00:00
    1/2/08
    2008-01-02 00:00:00
    10/31/08
    2008-10-31 00:00:00
    8/25/08
    2008-08-25 00:00:00
    9/6/08
    2008-09-06 00:00:00
    2/1/08
    2008-02-01 00:00:00
    1/17/08
    2008-01-17 00:00:00
    2/11/08
    2008-02-11 00:00:00
    4/11/08
    2008-04-11 00:00:00
    10/19/11
    2011-10-19 00:00:00
    7/22/11
    2011-07-22 00:00:00
    1/10/11
    2011-01-10 00:00:00
    7/7/11
    2011-07-07 00:00:00
    5/11/11
    2011-05-11 00:00:00
    8/3/11
    2011-08-03 00:00:00
    4/21/11
    2011-04-21 00:00:00
    10/27/11
    2011-10-27 00:00:00
    2/24/11
    2011-02-24 00:00:00
    1/13/11
    2011-01-13 00:00:00
    3/30/11
    2011-03-30 00:00:00
    12/7/11
    2011-12-07 00:00:00
    7/29/11
    2011-07-29 00:00:00
    12/14/11
    2011-12-14 00:00:00
    3/8/11
    2011-03-08 00:00:00
    5/25/11
    2011-05-25 00:00:00
    2/10/11
    2011-02-10 00:00:00
    7/8/11
    2011-07-08 00:00:00
    9/28/11
    2011-09-28 00:00:00
    7/21/11
    2011-07-21 00:00:00
    5/16/11
    2011-05-16 00:00:00
    11/22/11
    2011-11-22 00:00:00
    6/11/11
    2011-06-11 00:00:00
    6/16/11
    2011-06-16 00:00:00
    5/25/11
    2011-05-25 00:00:00
    1/12/11
    2011-01-12 00:00:00
    10/25/11
    2011-10-25 00:00:00
    4/28/11
    2011-04-28 00:00:00
    2/25/11
    2011-02-25 00:00:00
    6/8/11
    2011-06-08 00:00:00
    3/24/11
    2011-03-24 00:00:00
    3/11/11
    2011-03-11 00:00:00
    3/15/11
    2011-03-15 00:00:00
    10/2/11
    2011-10-02 00:00:00
    11/24/11
    2011-11-24 00:00:00
    6/17/11
    2011-06-17 00:00:00
    5/18/11
    2011-05-18 00:00:00
    2/18/11
    2011-02-18 00:00:00
    4/3/11
    2011-04-03 00:00:00
    5/15/11
    2011-05-15 00:00:00
    7/6/11
    2011-07-06 00:00:00
    7/29/11
    2011-07-29 00:00:00
    12/24/11
    2011-12-24 00:00:00
    12/25/11
    2011-12-25 00:00:00
    9/30/11
    2011-09-30 00:00:00
    3/2/11
    2011-03-02 00:00:00
    8/5/11
    2011-08-05 00:00:00
    11/22/11
    2011-11-22 00:00:00
    3/17/11
    2011-03-17 00:00:00
    12/10/11
    2011-12-10 00:00:00
    3/2/11
    2011-03-02 00:00:00
    9/23/11
    2011-09-23 00:00:00
    1/28/11
    2011-01-28 00:00:00
    4/13/11
    2011-04-13 00:00:00
    8/31/11
    2011-08-31 00:00:00
    8/28/11
    2011-08-28 00:00:00
    2/2/11
    2011-02-02 00:00:00
    10/17/11
    2011-10-17 00:00:00
    5/11/11
    2011-05-11 00:00:00
    5/5/11
    2011-05-05 00:00:00
    2/22/11
    2011-02-22 00:00:00
    12/16/11
    2011-12-16 00:00:00
    9/9/11
    2011-09-09 00:00:00
    9/22/11
    2011-09-22 00:00:00
    2/16/11
    2011-02-16 00:00:00
    3/3/11
    2011-03-03 00:00:00
    11/10/11
    2011-11-10 00:00:00
    10/28/11
    2011-10-28 00:00:00
    6/30/11
    2011-06-30 00:00:00
    2/14/11
    2011-02-14 00:00:00
    12/8/11
    2011-12-08 00:00:00
    8/19/11
    2011-08-19 00:00:00
    7/27/11
    2011-07-27 00:00:00
    3/8/11
    2011-03-08 00:00:00
    8/12/11
    2011-08-12 00:00:00
    9/23/11
    2011-09-23 00:00:00
    11/11/11
    2011-11-11 00:00:00
    9/30/11
    2011-09-30 00:00:00
    1/7/11
    2011-01-07 00:00:00
    10/17/11
    2011-10-17 00:00:00
    12/27/11
    2011-12-27 00:00:00
    12/22/11
    2011-12-22 00:00:00
    9/12/11
    2011-09-12 00:00:00
    12/25/11
    2011-12-25 00:00:00
    8/9/11
    2011-08-09 00:00:00
    12/11/11
    2011-12-11 00:00:00
    9/22/11
    2011-09-22 00:00:00
    9/28/11
    2011-09-28 00:00:00
    2/4/11
    2011-02-04 00:00:00
    10/28/11
    2011-10-28 00:00:00
    9/16/11
    2011-09-16 00:00:00
    1/12/11
    2011-01-12 00:00:00
    5/5/11
    2011-05-05 00:00:00
    8/17/11
    2011-08-17 00:00:00
    9/5/11
    2011-09-05 00:00:00
    9/15/11
    2011-09-15 00:00:00
    9/8/11
    2011-09-08 00:00:00
    11/15/11
    2011-11-15 00:00:00
    10/14/11
    2011-10-14 00:00:00
    1/21/11
    2011-01-21 00:00:00
    11/4/11
    2011-11-04 00:00:00
    10/12/11
    2011-10-12 00:00:00
    11/2/11
    2011-11-02 00:00:00
    12/8/11
    2011-12-08 00:00:00
    4/8/11
    2011-04-08 00:00:00
    2/22/11
    2011-02-22 00:00:00
    9/16/11
    2011-09-16 00:00:00
    8/23/11
    2011-08-23 00:00:00
    6/23/11
    2011-06-23 00:00:00
    7/20/11
    2011-07-20 00:00:00
    8/18/11
    2011-08-18 00:00:00
    11/17/11
    2011-11-17 00:00:00
    4/11/11
    2011-04-11 00:00:00
    3/9/11
    2011-03-09 00:00:00
    5/12/11
    2011-05-12 00:00:00
    3/25/11
    2011-03-25 00:00:00
    12/8/11
    2011-12-08 00:00:00
    12/24/11
    2011-12-24 00:00:00
    3/10/11
    2011-03-10 00:00:00
    12/14/11
    2011-12-14 00:00:00
    2/11/11
    2011-02-11 00:00:00
    9/16/11
    2011-09-16 00:00:00
    11/28/11
    2011-11-28 00:00:00
    7/29/11
    2011-07-29 00:00:00
    1/24/11
    2011-01-24 00:00:00
    4/7/11
    2011-04-07 00:00:00
    8/4/11
    2011-08-04 00:00:00
    9/23/11
    2011-09-23 00:00:00
    10/14/11
    2011-10-14 00:00:00
    6/10/11
    2011-06-10 00:00:00
    3/18/11
    2011-03-18 00:00:00
    8/19/11
    2011-08-19 00:00:00
    9/28/11
    2011-09-28 00:00:00
    1/13/11
    2011-01-13 00:00:00
    5/20/11
    2011-05-20 00:00:00
    2/11/11
    2011-02-11 00:00:00
    9/9/11
    2011-09-09 00:00:00
    3/30/11
    2011-03-30 00:00:00
    10/13/11
    2011-10-13 00:00:00
    3/16/11
    2011-03-16 00:00:00
    7/7/11
    2011-07-07 00:00:00
    4/26/11
    2011-04-26 00:00:00
    8/26/11
    2011-08-26 00:00:00
    5/6/11
    2011-05-06 00:00:00
    10/6/11
    2011-10-06 00:00:00
    2/3/11
    2011-02-03 00:00:00
    10/14/11
    2011-10-14 00:00:00
    9/16/11
    2011-09-16 00:00:00
    7/22/11
    2011-07-22 00:00:00
    9/24/11
    2011-09-24 00:00:00
    2/1/11
    2011-02-01 00:00:00
    6/28/11
    2011-06-28 00:00:00
    10/19/11
    2011-10-19 00:00:00
    9/12/11
    2011-09-12 00:00:00
    1/13/11
    2011-01-13 00:00:00
    3/4/11
    2011-03-04 00:00:00
    6/6/11
    2011-06-06 00:00:00
    5/6/11
    2011-05-06 00:00:00
    9/10/11
    2011-09-10 00:00:00
    4/9/11
    2011-04-09 00:00:00
    11/22/11
    2011-11-22 00:00:00
    8/18/11
    2011-08-18 00:00:00
    4/1/11
    2011-04-01 00:00:00
    11/4/11
    2011-11-04 00:00:00
    9/2/11
    2011-09-02 00:00:00
    3/11/11
    2011-03-11 00:00:00
    9/2/11
    2011-09-02 00:00:00
    3/1/11
    2011-03-01 00:00:00
    4/8/11
    2011-04-08 00:00:00
    11/26/11
    2011-11-26 00:00:00
    6/1/11
    2011-06-01 00:00:00
    8/3/11
    2011-08-03 00:00:00
    2/8/11
    2011-02-08 00:00:00
    12/9/11
    2011-12-09 00:00:00
    2/2/11
    2011-02-02 00:00:00
    9/15/11
    2011-09-15 00:00:00
    9/1/11
    2011-09-01 00:00:00
    1/1/11
    2011-01-01 00:00:00
    9/3/11
    2011-09-03 00:00:00
    11/11/11
    2011-11-11 00:00:00
    11/11/11
    2011-11-11 00:00:00
    4/2/11
    2011-04-02 00:00:00
    12/1/11
    2011-12-01 00:00:00
    9/16/11
    2011-09-16 00:00:00
    9/14/11
    2011-09-14 00:00:00
    12/3/11
    2011-12-03 00:00:00
    2/14/11
    2011-02-14 00:00:00
    11/9/11
    2011-11-09 00:00:00
    2/3/11
    2011-02-03 00:00:00
    2/12/11
    2011-02-12 00:00:00
    9/16/11
    2011-09-16 00:00:00
    9/30/11
    2011-09-30 00:00:00
    4/8/11
    2011-04-08 00:00:00
    8/11/11
    2011-08-11 00:00:00
    8/17/11
    2011-08-17 00:00:00
    4/29/11
    2011-04-29 00:00:00
    4/24/11
    2011-04-24 00:00:00
    11/22/11
    2011-11-22 00:00:00
    2/10/11
    2011-02-10 00:00:00
    9/27/11
    2011-09-27 00:00:00
    6/7/11
    2011-06-07 00:00:00
    9/10/11
    2011-09-10 00:00:00
    4/14/11
    2011-04-14 00:00:00
    6/27/11
    2011-06-27 00:00:00
    7/28/11
    2011-07-28 00:00:00
    3/15/11
    2011-03-15 00:00:00
    9/30/11
    2011-09-30 00:00:00
    6/24/11
    2011-06-24 00:00:00
    9/23/11
    2011-09-23 00:00:00
    9/24/11
    2011-09-24 00:00:00
    6/3/11
    2011-06-03 00:00:00
    4/1/11
    2011-04-01 00:00:00
    12/10/11
    2011-12-10 00:00:00
    10/21/11
    2011-10-21 00:00:00
    11/25/11
    2011-11-25 00:00:00
    4/29/11
    2011-04-29 00:00:00
    10/7/11
    2011-10-07 00:00:00
    5/16/11
    2011-05-16 00:00:00
    2/24/11
    2011-02-24 00:00:00
    12/3/11
    2011-12-03 00:00:00
    8/4/11
    2011-08-04 00:00:00
    9/15/11
    2011-09-15 00:00:00
    2/25/11
    2011-02-25 00:00:00
    10/14/11
    2011-10-14 00:00:00
    9/30/11
    2011-09-30 00:00:00
    10/29/11
    2011-10-29 00:00:00
    1/22/11
    2011-01-22 00:00:00
    12/21/11
    2011-12-21 00:00:00
    3/19/11
    2011-03-19 00:00:00
    11/8/11
    2011-11-08 00:00:00
    9/10/11
    2011-09-10 00:00:00
    11/23/11
    2011-11-23 00:00:00
    3/14/11
    2011-03-14 00:00:00
    2/16/11
    2011-02-16 00:00:00
    9/9/11
    2011-09-09 00:00:00
    10/18/11
    2011-10-18 00:00:00
    4/9/11
    2011-04-09 00:00:00
    1/28/11
    2011-01-28 00:00:00
    10/14/11
    2011-10-14 00:00:00
    9/23/11
    2011-09-23 00:00:00
    4/19/11
    2011-04-19 00:00:00
    1/23/11
    2011-01-23 00:00:00
    9/10/11
    2011-09-10 00:00:00
    9/9/11
    2011-09-09 00:00:00
    1/20/11
    2011-01-20 00:00:00
    5/19/11
    2011-05-19 00:00:00
    10/14/11
    2011-10-14 00:00:00
    11/3/11
    2011-11-03 00:00:00
    4/20/11
    2011-04-20 00:00:00
    4/1/11
    2011-04-01 00:00:00
    11/14/11
    2011-11-14 00:00:00
    4/3/11
    2011-04-03 00:00:00
    12/23/11
    2011-12-23 00:00:00
    8/5/11
    2011-08-05 00:00:00
    11/18/11
    2011-11-18 00:00:00
    8/26/11
    2011-08-26 00:00:00
    9/19/11
    2011-09-19 00:00:00
    5/6/11
    2011-05-06 00:00:00
    4/19/11
    2011-04-19 00:00:00
    4/15/11
    2011-04-15 00:00:00
    1/21/11
    2011-01-21 00:00:00
    3/10/11
    2011-03-10 00:00:00
    9/8/11
    2011-09-08 00:00:00
    5/22/11
    2011-05-22 00:00:00
    10/21/11
    2011-10-21 00:00:00
    9/5/11
    2011-09-05 00:00:00
    4/29/11
    2011-04-29 00:00:00
    4/22/11
    2011-04-22 00:00:00
    9/11/11
    2011-09-11 00:00:00
    1/1/11
    2011-01-01 00:00:00
    1/1/11
    2011-01-01 00:00:00
    7/1/11
    2011-07-01 00:00:00
    3/18/11
    2011-03-18 00:00:00
    6/25/11
    2011-06-25 00:00:00
    11/4/11
    2011-11-04 00:00:00
    9/5/11
    2011-09-05 00:00:00
    9/20/11
    2011-09-20 00:00:00
    2/10/11
    2011-02-10 00:00:00
    9/30/11
    2011-09-30 00:00:00
    8/31/11
    2011-08-31 00:00:00
    7/29/11
    2011-07-29 00:00:00
    4/1/11
    2011-04-01 00:00:00
    8/11/11
    2011-08-11 00:00:00
    10/27/11
    2011-10-27 00:00:00
    6/20/11
    2011-06-20 00:00:00
    9/12/11
    2011-09-12 00:00:00
    5/19/11
    2011-05-19 00:00:00
    8/25/11
    2011-08-25 00:00:00
    6/18/11
    2011-06-18 00:00:00
    7/1/11
    2011-07-01 00:00:00
    12/30/11
    2011-12-30 00:00:00
    7/1/11
    2011-07-01 00:00:00
    9/15/11
    2011-09-15 00:00:00
    8/19/11
    2011-08-19 00:00:00
    12/5/11
    2011-12-05 00:00:00
    2/11/11
    2011-02-11 00:00:00
    9/10/11
    2011-09-10 00:00:00
    9/9/11
    2011-09-09 00:00:00
    4/23/11
    2011-04-23 00:00:00
    11/27/11
    2011-11-27 00:00:00
    8/9/11
    2011-08-09 00:00:00
    9/7/11
    2011-09-07 00:00:00
    3/3/11
    2011-03-03 00:00:00
    3/18/11
    2011-03-18 00:00:00
    1/28/11
    2011-01-28 00:00:00
    6/12/11
    2011-06-12 00:00:00
    9/9/11
    2011-09-09 00:00:00
    9/13/11
    2011-09-13 00:00:00
    5/6/11
    2011-05-06 00:00:00
    9/2/11
    2011-09-02 00:00:00
    8/7/11
    2011-08-07 00:00:00
    1/2/11
    2011-01-02 00:00:00
    3/2/11
    2011-03-02 00:00:00
    2/15/11
    2011-02-15 00:00:00
    5/6/11
    2011-05-06 00:00:00
    4/22/11
    2011-04-22 00:00:00
    6/6/11
    2011-06-06 00:00:00
    3/1/11
    2011-03-01 00:00:00
    4/13/11
    2011-04-13 00:00:00
    9/8/11
    2011-09-08 00:00:00
    4/14/11
    2011-04-14 00:00:00
    9/6/11
    2011-09-06 00:00:00
    10/14/11
    2011-10-14 00:00:00
    10/21/11
    2011-10-21 00:00:00
    10/25/11
    2011-10-25 00:00:00
    10/29/11
    2011-10-29 00:00:00
    11/14/11
    2011-11-14 00:00:00
    10/1/11
    2011-10-01 00:00:00
    4/19/11
    2011-04-19 00:00:00
    6/24/11
    2011-06-24 00:00:00
    6/7/11
    2011-06-07 00:00:00
    8/4/11
    2011-08-04 00:00:00
    2/26/11
    2011-02-26 00:00:00
    11/16/11
    2011-11-16 00:00:00
    1/23/11
    2011-01-23 00:00:00
    4/21/11
    2011-04-21 00:00:00
    1/1/11
    2011-01-01 00:00:00
    9/27/11
    2011-09-27 00:00:00
    11/15/11
    2011-11-15 00:00:00
    10/24/11
    2011-10-24 00:00:00
    3/25/11
    2011-03-25 00:00:00
    8/13/11
    2011-08-13 00:00:00
    10/18/11
    2011-10-18 00:00:00
    9/7/11
    2011-09-07 00:00:00
    10/11/11
    2011-10-11 00:00:00
    10/11/11
    2011-10-11 00:00:00
    12/21/11
    2011-12-21 00:00:00
    4/22/11
    2011-04-22 00:00:00
    1/17/11
    2011-01-17 00:00:00
    10/14/11
    2011-10-14 00:00:00
    9/2/11
    2011-09-02 00:00:00
    9/30/11
    2011-09-30 00:00:00
    10/21/11
    2011-10-21 00:00:00
    5/10/11
    2011-05-10 00:00:00
    4/1/11
    2011-04-01 00:00:00
    3/7/11
    2011-03-07 00:00:00
    10/18/11
    2011-10-18 00:00:00
    7/19/11
    2011-07-19 00:00:00
    1/1/11
    2011-01-01 00:00:00
    11/15/11
    2011-11-15 00:00:00
    3/5/11
    2011-03-05 00:00:00
    6/17/11
    2011-06-17 00:00:00
    7/15/11
    2011-07-15 00:00:00
    12/9/11
    2011-12-09 00:00:00
    11/12/11
    2011-11-12 00:00:00
    3/3/11
    2011-03-03 00:00:00
    11/21/11
    2011-11-21 00:00:00
    2/2/11
    2011-02-02 00:00:00
    12/27/11
    2011-12-27 00:00:00
    11/30/11
    2011-11-30 00:00:00
    9/6/11
    2011-09-06 00:00:00
    8/17/11
    2011-08-17 00:00:00
    2/15/11
    2011-02-15 00:00:00
    7/16/11
    2011-07-16 00:00:00
    3/29/11
    2011-03-29 00:00:00
    2/25/11
    2011-02-25 00:00:00
    6/30/11
    2011-06-30 00:00:00
    9/20/11
    2011-09-20 00:00:00
    9/9/11
    2011-09-09 00:00:00
    10/22/11
    2011-10-22 00:00:00
    10/22/11
    2011-10-22 00:00:00
    1/1/11
    2011-01-01 00:00:00
    7/15/11
    2011-07-15 00:00:00
    9/14/11
    2011-09-14 00:00:00
    9/22/11
    2011-09-22 00:00:00
    1/1/11
    2011-01-01 00:00:00
    4/17/11
    2011-04-17 00:00:00
    9/2/11
    2011-09-02 00:00:00
    1/15/11
    2011-01-15 00:00:00
    4/9/11
    2011-04-09 00:00:00
    11/23/11
    2011-11-23 00:00:00
    2/11/11
    2011-02-11 00:00:00
    6/4/11
    2011-06-04 00:00:00
    3/25/11
    2011-03-25 00:00:00
    5/4/11
    2011-05-04 00:00:00
    2/11/11
    2011-02-11 00:00:00
    7/13/11
    2011-07-13 00:00:00
    8/23/11
    2011-08-23 00:00:00
    5/1/11
    2011-05-01 00:00:00
    9/1/11
    2011-09-01 00:00:00
    12/10/11
    2011-12-10 00:00:00
    3/6/11
    2011-03-06 00:00:00
    4/8/11
    2011-04-08 00:00:00
    9/20/11
    2011-09-20 00:00:00
    2/28/11
    2011-02-28 00:00:00
    6/3/11
    2011-06-03 00:00:00
    12/5/11
    2011-12-05 00:00:00
    11/20/11
    2011-11-20 00:00:00
    9/27/11
    2011-09-27 00:00:00
    5/10/11
    2011-05-10 00:00:00
    7/22/11
    2011-07-22 00:00:00
    8/27/11
    2011-08-27 00:00:00
    6/6/11
    2011-06-06 00:00:00
    9/1/11
    2011-09-01 00:00:00
    4/16/11
    2011-04-16 00:00:00
    9/13/11
    2011-09-13 00:00:00
    4/13/11
    2011-04-13 00:00:00
    3/13/11
    2011-03-13 00:00:00
    10/9/11
    2011-10-09 00:00:00
    7/18/11
    2011-07-18 00:00:00
    9/11/11
    2011-09-11 00:00:00
    9/20/11
    2011-09-20 00:00:00
    3/25/11
    2011-03-25 00:00:00
    2/22/11
    2011-02-22 00:00:00
    10/1/11
    2011-10-01 00:00:00
    7/8/11
    2011-07-08 00:00:00
    1/30/11
    2011-01-30 00:00:00
    1/27/11
    2011-01-27 00:00:00
    4/25/11
    2011-04-25 00:00:00
    10/14/11
    2011-10-14 00:00:00
    5/24/11
    2011-05-24 00:00:00
    4/29/11
    2011-04-29 00:00:00
    9/30/11
    2011-09-30 00:00:00
    3/18/11
    2011-03-18 00:00:00
    3/19/11
    2011-03-19 00:00:00
    1/24/11
    2011-01-24 00:00:00
    10/14/11
    2011-10-14 00:00:00
    9/10/11
    2011-09-10 00:00:00
    11/23/11
    2011-11-23 00:00:00
    12/2/11
    2011-12-02 00:00:00
    1/17/11
    2011-01-17 00:00:00
    3/21/11
    2011-03-21 00:00:00
    7/19/11
    2011-07-19 00:00:00
    6/11/11
    2011-06-11 00:00:00
    6/16/11
    2011-06-16 00:00:00
    2/21/11
    2011-02-21 00:00:00
    3/18/11
    2011-03-18 00:00:00
    12/10/11
    2011-12-10 00:00:00
    2/11/11
    2011-02-11 00:00:00
    9/9/11
    2011-09-09 00:00:00
    3/12/11
    2011-03-12 00:00:00
    12/20/11
    2011-12-20 00:00:00
    7/15/11
    2011-07-15 00:00:00
    9/30/11
    2011-09-30 00:00:00
    3/26/11
    2011-03-26 00:00:00
    3/28/11
    2011-03-28 00:00:00
    10/14/11
    2011-10-14 00:00:00
    9/16/11
    2011-09-16 00:00:00
    1/12/11
    2011-01-12 00:00:00
    4/29/11
    2011-04-29 00:00:00
    8/26/11
    2011-08-26 00:00:00
    8/11/11
    2011-08-11 00:00:00
    7/9/11
    2011-07-09 00:00:00
    5/5/11
    2011-05-05 00:00:00
    10/5/11
    2011-10-05 00:00:00
    8/23/11
    2011-08-23 00:00:00
    1/28/11
    2011-01-28 00:00:00
    10/10/11
    2011-10-10 00:00:00
    1/28/11
    2011-01-28 00:00:00
    6/24/11
    2011-06-24 00:00:00
    5/17/11
    2011-05-17 00:00:00
    11/18/11
    2011-11-18 00:00:00
    7/1/11
    2011-07-01 00:00:00
    9/20/11
    2011-09-20 00:00:00
    5/16/11
    2011-05-16 00:00:00
    3/13/11
    2011-03-13 00:00:00
    4/26/11
    2011-04-26 00:00:00
    9/8/11
    2011-09-08 00:00:00
    4/29/11
    2011-04-29 00:00:00
    10/11/11
    2011-10-11 00:00:00
    7/15/11
    2011-07-15 00:00:00
    7/22/11
    2011-07-22 00:00:00
    4/18/11
    2011-04-18 00:00:00
    9/11/11
    2011-09-11 00:00:00
    9/12/11
    2011-09-12 00:00:00
    8/25/11
    2011-08-25 00:00:00
    6/10/11
    2011-06-10 00:00:00
    12/2/11
    2011-12-02 00:00:00
    11/28/11
    2011-11-28 00:00:00
    1/20/11
    2011-01-20 00:00:00
    4/22/11
    2011-04-22 00:00:00
    11/27/11
    2011-11-27 00:00:00
    9/22/11
    2011-09-22 00:00:00
    10/21/11
    2011-10-21 00:00:00
    7/8/11
    2011-07-08 00:00:00
    4/5/11
    2011-04-05 00:00:00
    12/14/11
    2011-12-14 00:00:00
    10/10/11
    2011-10-10 00:00:00
    12/14/11
    2011-12-14 00:00:00
    10/14/11
    2011-10-14 00:00:00
    4/9/11
    2011-04-09 00:00:00
    11/23/11
    2011-11-23 00:00:00
    4/14/11
    2011-04-14 00:00:00
    9/26/11
    2011-09-26 00:00:00
    4/10/11
    2011-04-10 00:00:00
    1/30/11
    2011-01-30 00:00:00
    12/27/11
    2011-12-27 00:00:00
    4/22/11
    2011-04-22 00:00:00
    8/19/11
    2011-08-19 00:00:00
    1/22/11
    2011-01-22 00:00:00
    11/11/11
    2011-11-11 00:00:00
    10/30/11
    2011-10-30 00:00:00
    7/8/11
    2011-07-08 00:00:00
    6/23/11
    2011-06-23 00:00:00
    7/31/11
    2011-07-31 00:00:00
    3/27/11
    2011-03-27 00:00:00
    8/5/11
    2011-08-05 00:00:00
    1/28/11
    2011-01-28 00:00:00
    9/20/11
    2011-09-20 00:00:00
    4/26/11
    2011-04-26 00:00:00
    11/16/11
    2011-11-16 00:00:00
    6/6/11
    2011-06-06 00:00:00
    9/9/11
    2011-09-09 00:00:00
    9/3/11
    2011-09-03 00:00:00
    3/21/11
    2011-03-21 00:00:00
    11/16/11
    2011-11-16 00:00:00
    11/19/11
    2011-11-19 00:00:00
    9/9/11
    2011-09-09 00:00:00
    8/11/11
    2011-08-11 00:00:00
    1/21/11
    2011-01-21 00:00:00
    11/2/11
    2011-11-02 00:00:00
    6/1/11
    2011-06-01 00:00:00
    10/8/11
    2011-10-08 00:00:00
    9/16/11
    2011-09-16 00:00:00
    11/16/11
    2011-11-16 00:00:00
    9/20/11
    2011-09-20 00:00:00
    7/25/11
    2011-07-25 00:00:00
    6/9/11
    2011-06-09 00:00:00
    12/16/11
    2011-12-16 00:00:00
    11/2/11
    2011-11-02 00:00:00
    12/18/02
    2002-12-18 00:00:00
    11/13/02
    2002-11-13 00:00:00
    3/10/02
    2002-03-10 00:00:00
    6/14/02
    2002-06-14 00:00:00
    3/15/02
    2002-03-15 00:00:00
    5/1/02
    2002-05-01 00:00:00
    5/15/02
    2002-05-15 00:00:00
    12/25/02
    2002-12-25 00:00:00
    7/3/02
    2002-07-03 00:00:00
    9/24/02
    2002-09-24 00:00:00
    6/20/02
    2002-06-20 00:00:00
    6/21/02
    2002-06-21 00:00:00
    10/2/02
    2002-10-02 00:00:00
    8/9/02
    2002-08-09 00:00:00
    11/26/02
    2002-11-26 00:00:00
    11/17/02
    2002-11-17 00:00:00
    10/31/02
    2002-10-31 00:00:00
    12/12/02
    2002-12-12 00:00:00
    3/22/02
    2002-03-22 00:00:00
    12/14/02
    2002-12-14 00:00:00
    3/29/02
    2002-03-29 00:00:00
    7/19/02
    2002-07-19 00:00:00
    3/29/02
    2002-03-29 00:00:00
    11/8/02
    2002-11-08 00:00:00
    6/28/02
    2002-06-28 00:00:00
    4/11/02
    2002-04-11 00:00:00
    2/23/02
    2002-02-23 00:00:00
    12/19/02
    2002-12-19 00:00:00
    8/2/02
    2002-08-02 00:00:00
    10/18/02
    2002-10-18 00:00:00
    11/2/02
    2002-11-02 00:00:00
    1/25/02
    2002-01-25 00:00:00
    2/22/02
    2002-02-22 00:00:00
    4/26/02
    2002-04-26 00:00:00
    2/5/02
    2002-02-05 00:00:00
    2/6/02
    2002-02-06 00:00:00
    9/9/02
    2002-09-09 00:00:00
    12/12/02
    2002-12-12 00:00:00
    12/6/02
    2002-12-06 00:00:00
    1/12/02
    2002-01-12 00:00:00
    12/27/02
    2002-12-27 00:00:00
    7/12/02
    2002-07-12 00:00:00
    7/26/02
    2002-07-26 00:00:00
    10/31/02
    2002-10-31 00:00:00
    5/24/02
    2002-05-24 00:00:00
    10/25/02
    2002-10-25 00:00:00
    6/14/02
    2002-06-14 00:00:00
    9/27/02
    2002-09-27 00:00:00
    5/24/02
    2002-05-24 00:00:00
    1/4/02
    2002-01-04 00:00:00
    1/11/02
    2002-01-11 00:00:00
    4/16/02
    2002-04-16 00:00:00
    10/31/02
    2002-10-31 00:00:00
    10/9/02
    2002-10-09 00:00:00
    12/16/02
    2002-12-16 00:00:00
    8/31/02
    2002-08-31 00:00:00
    6/24/02
    2002-06-24 00:00:00
    5/31/02
    2002-05-31 00:00:00
    7/10/02
    2002-07-10 00:00:00
    3/1/02
    2002-03-01 00:00:00
    10/21/02
    2002-10-21 00:00:00
    12/13/02
    2002-12-13 00:00:00
    8/29/02
    2002-08-29 00:00:00
    4/19/02
    2002-04-19 00:00:00
    12/6/02
    2002-12-06 00:00:00
    9/26/02
    2002-09-26 00:00:00
    4/1/02
    2002-04-01 00:00:00
    12/14/02
    2002-12-14 00:00:00
    3/4/02
    2002-03-04 00:00:00
    8/23/02
    2002-08-23 00:00:00
    4/12/02
    2002-04-12 00:00:00
    8/21/02
    2002-08-21 00:00:00
    6/14/02
    2002-06-14 00:00:00
    2/15/02
    2002-02-15 00:00:00
    1/1/02
    2002-01-01 00:00:00
    3/1/02
    2002-03-01 00:00:00
    10/1/02
    2002-10-01 00:00:00
    5/10/02
    2002-05-10 00:00:00
    4/7/02
    2002-04-07 00:00:00
    10/25/02
    2002-10-25 00:00:00
    7/19/02
    2002-07-19 00:00:00
    1/4/02
    2002-01-04 00:00:00
    7/17/02
    2002-07-17 00:00:00
    4/15/02
    2002-04-15 00:00:00
    2/10/02
    2002-02-10 00:00:00
    8/7/02
    2002-08-07 00:00:00
    12/17/02
    2002-12-17 00:00:00
    5/24/02
    2002-05-24 00:00:00
    11/22/02
    2002-11-22 00:00:00
    2/14/02
    2002-02-14 00:00:00
    12/27/02
    2002-12-27 00:00:00
    4/26/02
    2002-04-26 00:00:00
    2/1/02
    2002-02-01 00:00:00
    3/22/02
    2002-03-22 00:00:00
    4/13/02
    2002-04-13 00:00:00
    12/26/02
    2002-12-26 00:00:00
    8/21/02
    2002-08-21 00:00:00
    11/6/02
    2002-11-06 00:00:00
    11/27/02
    2002-11-27 00:00:00
    9/6/02
    2002-09-06 00:00:00
    3/21/02
    2002-03-21 00:00:00
    10/2/02
    2002-10-02 00:00:00
    7/12/02
    2002-07-12 00:00:00
    3/14/02
    2002-03-14 00:00:00
    8/23/02
    2002-08-23 00:00:00
    4/3/02
    2002-04-03 00:00:00
    9/9/02
    2002-09-09 00:00:00
    2/15/02
    2002-02-15 00:00:00
    12/13/02
    2002-12-13 00:00:00
    8/8/02
    2002-08-08 00:00:00
    8/4/02
    2002-08-04 00:00:00
    11/27/02
    2002-11-27 00:00:00
    9/20/02
    2002-09-20 00:00:00
    9/8/02
    2002-09-08 00:00:00
    11/22/02
    2002-11-22 00:00:00
    2/19/02
    2002-02-19 00:00:00
    10/24/02
    2002-10-24 00:00:00
    5/10/02
    2002-05-10 00:00:00
    1/25/02
    2002-01-25 00:00:00
    5/12/02
    2002-05-12 00:00:00
    1/18/02
    2002-01-18 00:00:00
    9/1/02
    2002-09-01 00:00:00
    3/25/02
    2002-03-25 00:00:00
    6/7/02
    2002-06-07 00:00:00
    12/28/02
    2002-12-28 00:00:00
    12/30/02
    2002-12-30 00:00:00
    10/11/02
    2002-10-11 00:00:00
    10/4/02
    2002-10-04 00:00:00
    10/15/02
    2002-10-15 00:00:00
    5/10/02
    2002-05-10 00:00:00
    12/2/02
    2002-12-02 00:00:00
    9/20/02
    2002-09-20 00:00:00
    12/6/02
    2002-12-06 00:00:00
    9/20/02
    2002-09-20 00:00:00
    5/19/02
    2002-05-19 00:00:00
    5/3/02
    2002-05-03 00:00:00
    6/1/02
    2002-06-01 00:00:00
    3/17/02
    2002-03-17 00:00:00
    7/1/02
    2002-07-01 00:00:00
    1/1/02
    2002-01-01 00:00:00
    5/10/02
    2002-05-10 00:00:00
    6/28/02
    2002-06-28 00:00:00
    2/8/02
    2002-02-08 00:00:00
    1/23/02
    2002-01-23 00:00:00
    1/14/02
    2002-01-14 00:00:00
    4/17/02
    2002-04-17 00:00:00
    2/8/02
    2002-02-08 00:00:00
    11/1/02
    2002-11-01 00:00:00
    4/9/02
    2002-04-09 00:00:00
    5/17/02
    2002-05-17 00:00:00
    10/21/02
    2002-10-21 00:00:00
    10/5/02
    2002-10-05 00:00:00
    1/25/02
    2002-01-25 00:00:00
    6/7/02
    2002-06-07 00:00:00
    12/19/02
    2002-12-19 00:00:00
    9/10/02
    2002-09-10 00:00:00
    2/13/02
    2002-02-13 00:00:00
    1/19/02
    2002-01-19 00:00:00
    3/7/02
    2002-03-07 00:00:00
    9/13/02
    2002-09-13 00:00:00
    1/18/02
    2002-01-18 00:00:00
    10/11/02
    2002-10-11 00:00:00
    2/4/02
    2002-02-04 00:00:00
    10/1/02
    2002-10-01 00:00:00
    10/11/02
    2002-10-11 00:00:00
    1/13/02
    2002-01-13 00:00:00
    2/15/02
    2002-02-15 00:00:00
    8/15/02
    2002-08-15 00:00:00
    9/2/02
    2002-09-02 00:00:00
    10/14/02
    2002-10-14 00:00:00
    2/6/02
    2002-02-06 00:00:00
    3/10/02
    2002-03-10 00:00:00
    1/18/02
    2002-01-18 00:00:00
    11/24/02
    2002-11-24 00:00:00
    12/13/02
    2002-12-13 00:00:00
    10/6/02
    2002-10-06 00:00:00
    9/29/02
    2002-09-29 00:00:00
    5/18/02
    2002-05-18 00:00:00
    8/20/02
    2002-08-20 00:00:00
    1/16/02
    2002-01-16 00:00:00
    3/12/02
    2002-03-12 00:00:00
    2/22/02
    2002-02-22 00:00:00
    2/27/02
    2002-02-27 00:00:00
    12/13/02
    2002-12-13 00:00:00
    5/17/02
    2002-05-17 00:00:00
    7/3/02
    2002-07-03 00:00:00
    1/12/02
    2002-01-12 00:00:00
    1/26/02
    2002-01-26 00:00:00
    6/14/02
    2002-06-14 00:00:00
    6/28/02
    2002-06-28 00:00:00
    11/4/02
    2002-11-04 00:00:00
    11/22/02
    2002-11-22 00:00:00
    2/22/02
    2002-02-22 00:00:00
    11/15/02
    2002-11-15 00:00:00
    5/24/02
    2002-05-24 00:00:00
    9/5/02
    2002-09-05 00:00:00
    7/30/02
    2002-07-30 00:00:00
    2/13/02
    2002-02-13 00:00:00
    1/1/02
    2002-01-01 00:00:00
    10/25/02
    2002-10-25 00:00:00
    1/10/02
    2002-01-10 00:00:00
    9/16/02
    2002-09-16 00:00:00
    3/8/02
    2002-03-08 00:00:00
    8/30/02
    2002-08-30 00:00:00
    8/9/02
    2002-08-09 00:00:00
    6/18/02
    2002-06-18 00:00:00
    6/21/02
    2002-06-21 00:00:00
    12/20/02
    2002-12-20 00:00:00
    10/18/02
    2002-10-18 00:00:00
    7/3/02
    2002-07-03 00:00:00
    1/14/02
    2002-01-14 00:00:00
    6/9/02
    2002-06-09 00:00:00
    10/2/02
    2002-10-02 00:00:00
    11/26/02
    2002-11-26 00:00:00
    12/13/02
    2002-12-13 00:00:00
    6/21/02
    2002-06-21 00:00:00
    9/2/02
    2002-09-02 00:00:00
    8/2/02
    2002-08-02 00:00:00
    1/18/02
    2002-01-18 00:00:00
    7/26/02
    2002-07-26 00:00:00
    3/28/02
    2002-03-28 00:00:00
    4/5/02
    2002-04-05 00:00:00
    4/22/02
    2002-04-22 00:00:00
    1/25/02
    2002-01-25 00:00:00
    12/20/02
    2002-12-20 00:00:00
    1/1/02
    2002-01-01 00:00:00
    9/6/02
    2002-09-06 00:00:00
    9/6/02
    2002-09-06 00:00:00
    2/1/02
    2002-02-01 00:00:00
    11/13/02
    2002-11-13 00:00:00
    5/31/02
    2002-05-31 00:00:00
    10/29/02
    2002-10-29 00:00:00
    10/11/02
    2002-10-11 00:00:00
    11/22/02
    2002-11-22 00:00:00
    5/8/02
    2002-05-08 00:00:00
    1/1/02
    2002-01-01 00:00:00
    2/11/02
    2002-02-11 00:00:00
    1/1/02
    2002-01-01 00:00:00
    6/11/02
    2002-06-11 00:00:00
    11/21/02
    2002-11-21 00:00:00
    9/10/02
    2002-09-10 00:00:00
    10/8/02
    2002-10-08 00:00:00
    5/23/02
    2002-05-23 00:00:00
    1/1/02
    2002-01-01 00:00:00
    10/11/02
    2002-10-11 00:00:00
    5/9/02
    2002-05-09 00:00:00
    9/27/02
    2002-09-27 00:00:00
    6/7/02
    2002-06-07 00:00:00
    9/23/02
    2002-09-23 00:00:00
    5/31/02
    2002-05-31 00:00:00
    9/13/02
    2002-09-13 00:00:00
    4/24/02
    2002-04-24 00:00:00
    10/4/02
    2002-10-04 00:00:00
    3/17/02
    2002-03-17 00:00:00
    8/2/02
    2002-08-02 00:00:00
    5/21/02
    2002-05-21 00:00:00
    5/3/02
    2002-05-03 00:00:00
    12/22/02
    2002-12-22 00:00:00
    7/20/02
    2002-07-20 00:00:00
    1/1/02
    2002-01-01 00:00:00
    10/4/02
    2002-10-04 00:00:00
    6/10/02
    2002-06-10 00:00:00
    12/20/02
    2002-12-20 00:00:00
    12/17/02
    2002-12-17 00:00:00
    1/11/02
    2002-01-11 00:00:00
    1/18/02
    2002-01-18 00:00:00
    10/14/94
    1994-10-14 00:00:00
    9/10/94
    1994-09-10 00:00:00
    7/6/94
    1994-07-06 00:00:00
    6/23/94
    1994-06-23 00:00:00
    6/9/94
    1994-06-09 00:00:00
    12/16/94
    1994-12-16 00:00:00
    3/18/94
    1994-03-18 00:00:00
    11/11/94
    1994-11-11 00:00:00
    12/15/94
    1994-12-15 00:00:00
    7/14/94
    1994-07-14 00:00:00
    7/29/94
    1994-07-29 00:00:00
    5/20/94
    1994-05-20 00:00:00
    10/28/94
    1994-10-28 00:00:00
    8/3/94
    1994-08-03 00:00:00
    8/25/94
    1994-08-25 00:00:00
    9/13/94
    1994-09-13 00:00:00
    5/24/94
    1994-05-24 00:00:00
    11/10/94
    1994-11-10 00:00:00
    12/16/94
    1994-12-16 00:00:00
    7/1/94
    1994-07-01 00:00:00
    3/9/94
    1994-03-09 00:00:00
    1/1/94
    1994-01-01 00:00:00
    3/25/94
    1994-03-25 00:00:00
    7/20/94
    1994-07-20 00:00:00
    5/11/94
    1994-05-11 00:00:00
    12/19/94
    1994-12-19 00:00:00
    5/26/94
    1994-05-26 00:00:00
    11/18/94
    1994-11-18 00:00:00
    11/17/94
    1994-11-17 00:00:00
    2/4/94
    1994-02-04 00:00:00
    7/29/94
    1994-07-29 00:00:00
    6/9/94
    1994-06-09 00:00:00
    12/9/94
    1994-12-09 00:00:00
    11/22/94
    1994-11-22 00:00:00
    10/14/94
    1994-10-14 00:00:00
    7/1/94
    1994-07-01 00:00:00
    9/2/94
    1994-09-02 00:00:00
    11/18/94
    1994-11-18 00:00:00
    11/23/94
    1994-11-23 00:00:00
    7/18/94
    1994-07-18 00:00:00
    8/5/94
    1994-08-05 00:00:00
    12/12/94
    1994-12-12 00:00:00
    8/19/94
    1994-08-19 00:00:00
    9/27/94
    1994-09-27 00:00:00
    9/15/94
    1994-09-15 00:00:00
    6/24/94
    1994-06-24 00:00:00
    9/17/94
    1994-09-17 00:00:00
    5/4/94
    1994-05-04 00:00:00
    5/31/94
    1994-05-31 00:00:00
    9/16/94
    1994-09-16 00:00:00
    9/23/94
    1994-09-23 00:00:00
    12/22/94
    1994-12-22 00:00:00
    9/29/94
    1994-09-29 00:00:00
    12/22/94
    1994-12-22 00:00:00
    10/7/94
    1994-10-07 00:00:00
    7/28/94
    1994-07-28 00:00:00
    3/30/94
    1994-03-30 00:00:00
    11/29/94
    1994-11-29 00:00:00
    2/18/94
    1994-02-18 00:00:00
    4/22/94
    1994-04-22 00:00:00
    1/1/94
    1994-01-01 00:00:00
    7/29/94
    1994-07-29 00:00:00
    6/17/94
    1994-06-17 00:00:00
    11/4/94
    1994-11-04 00:00:00
    6/3/94
    1994-06-03 00:00:00
    11/23/94
    1994-11-23 00:00:00
    10/26/94
    1994-10-26 00:00:00
    12/23/94
    1994-12-23 00:00:00
    3/4/94
    1994-03-04 00:00:00
    3/11/94
    1994-03-11 00:00:00
    2/18/94
    1994-02-18 00:00:00
    9/2/94
    1994-09-02 00:00:00
    12/23/94
    1994-12-23 00:00:00
    10/14/94
    1994-10-14 00:00:00
    10/7/94
    1994-10-07 00:00:00
    3/11/94
    1994-03-11 00:00:00
    12/21/94
    1994-12-21 00:00:00
    7/15/94
    1994-07-15 00:00:00
    2/11/94
    1994-02-11 00:00:00
    9/7/94
    1994-09-07 00:00:00
    6/17/94
    1994-06-17 00:00:00
    8/5/94
    1994-08-05 00:00:00
    12/28/94
    1994-12-28 00:00:00
    3/23/94
    1994-03-23 00:00:00
    12/8/94
    1994-12-08 00:00:00
    9/24/94
    1994-09-24 00:00:00
    10/14/94
    1994-10-14 00:00:00
    12/21/94
    1994-12-21 00:00:00
    1/26/94
    1994-01-26 00:00:00
    12/24/94
    1994-12-24 00:00:00
    10/21/94
    1994-10-21 00:00:00
    4/29/94
    1994-04-29 00:00:00
    8/31/94
    1994-08-31 00:00:00
    12/23/94
    1994-12-23 00:00:00
    5/6/94
    1994-05-06 00:00:00
    3/4/94
    1994-03-04 00:00:00
    2/11/94
    1994-02-11 00:00:00
    9/12/94
    1994-09-12 00:00:00
    8/12/94
    1994-08-12 00:00:00
    8/12/94
    1994-08-12 00:00:00
    11/4/94
    1994-11-04 00:00:00
    3/18/94
    1994-03-18 00:00:00
    8/24/94
    1994-08-24 00:00:00
    10/22/94
    1994-10-22 00:00:00
    10/28/94
    1994-10-28 00:00:00
    6/3/94
    1994-06-03 00:00:00
    7/22/94
    1994-07-22 00:00:00
    6/9/94
    1994-06-09 00:00:00
    3/9/94
    1994-03-09 00:00:00
    3/30/94
    1994-03-30 00:00:00
    12/23/94
    1994-12-23 00:00:00
    2/4/94
    1994-02-04 00:00:00
    7/22/94
    1994-07-22 00:00:00
    9/10/94
    1994-09-10 00:00:00
    1/7/94
    1994-01-07 00:00:00
    3/9/94
    1994-03-09 00:00:00
    10/14/94
    1994-10-14 00:00:00
    12/23/94
    1994-12-23 00:00:00
    12/2/94
    1994-12-02 00:00:00
    12/22/94
    1994-12-22 00:00:00
    2/18/94
    1994-02-18 00:00:00
    7/22/94
    1994-07-22 00:00:00
    6/10/94
    1994-06-10 00:00:00
    1/14/94
    1994-01-14 00:00:00
    3/11/94
    1994-03-11 00:00:00
    12/16/94
    1994-12-16 00:00:00
    10/1/94
    1994-10-01 00:00:00
    4/22/94
    1994-04-22 00:00:00
    6/29/94
    1994-06-29 00:00:00
    11/23/94
    1994-11-23 00:00:00
    8/5/94
    1994-08-05 00:00:00
    7/27/94
    1994-07-27 00:00:00
    4/7/94
    1994-04-07 00:00:00
    1/15/94
    1994-01-15 00:00:00
    9/15/94
    1994-09-15 00:00:00
    7/1/94
    1994-07-01 00:00:00
    1/1/94
    1994-01-01 00:00:00
    2/11/94
    1994-02-11 00:00:00
    4/15/94
    1994-04-15 00:00:00
    8/26/94
    1994-08-26 00:00:00
    9/30/94
    1994-09-30 00:00:00
    1/21/94
    1994-01-21 00:00:00
    10/27/94
    1994-10-27 00:00:00
    4/15/94
    1994-04-15 00:00:00
    11/26/94
    1994-11-26 00:00:00
    4/13/94
    1994-04-13 00:00:00
    4/29/94
    1994-04-29 00:00:00
    5/6/94
    1994-05-06 00:00:00
    4/8/94
    1994-04-08 00:00:00
    5/26/94
    1994-05-26 00:00:00
    1/14/94
    1994-01-14 00:00:00
    8/17/94
    1994-08-17 00:00:00
    3/18/94
    1994-03-18 00:00:00
    9/2/94
    1994-09-02 00:00:00
    11/4/94
    1994-11-04 00:00:00
    2/25/94
    1994-02-25 00:00:00
    8/19/94
    1994-08-19 00:00:00
    9/30/94
    1994-09-30 00:00:00
    10/21/94
    1994-10-21 00:00:00
    12/10/94
    1994-12-10 00:00:00
    1/20/94
    1994-01-20 00:00:00
    5/13/94
    1994-05-13 00:00:00
    4/1/94
    1994-04-01 00:00:00
    9/10/94
    1994-09-10 00:00:00
    9/9/94
    1994-09-09 00:00:00
    4/24/94
    1994-04-24 00:00:00
    4/20/94
    1994-04-20 00:00:00
    8/24/94
    1994-08-24 00:00:00
    3/16/94
    1994-03-16 00:00:00
    10/1/94
    1994-10-01 00:00:00
    9/4/94
    1994-09-04 00:00:00
    7/13/94
    1994-07-13 00:00:00
    10/13/94
    1994-10-13 00:00:00
    9/22/94
    1994-09-22 00:00:00
    4/22/94
    1994-04-22 00:00:00
    4/29/94
    1994-04-29 00:00:00
    11/7/94
    1994-11-07 00:00:00
    7/12/94
    1994-07-12 00:00:00
    3/24/94
    1994-03-24 00:00:00
    5/2/94
    1994-05-02 00:00:00
    4/29/94
    1994-04-29 00:00:00
    11/5/94
    1994-11-05 00:00:00
    8/26/94
    1994-08-26 00:00:00
    11/19/94
    1994-11-19 00:00:00
    4/25/12
    2012-04-25 00:00:00
    1/19/12
    2012-01-19 00:00:00
    7/16/12
    2012-07-16 00:00:00
    12/25/12
    2012-12-25 00:00:00
    10/25/12
    2012-10-25 00:00:00
    8/8/12
    2012-08-08 00:00:00
    11/26/12
    2012-11-26 00:00:00
    5/30/12
    2012-05-30 00:00:00
    9/8/12
    2012-09-08 00:00:00
    6/27/12
    2012-06-27 00:00:00
    6/21/12
    2012-06-21 00:00:00
    5/30/12
    2012-05-30 00:00:00
    3/12/12
    2012-03-12 00:00:00
    11/20/12
    2012-11-20 00:00:00
    6/26/12
    2012-06-26 00:00:00
    6/29/12
    2012-06-29 00:00:00
    12/20/12
    2012-12-20 00:00:00
    9/9/12
    2012-09-09 00:00:00
    11/1/12
    2012-11-01 00:00:00
    9/28/12
    2012-09-28 00:00:00
    5/23/12
    2012-05-23 00:00:00
    3/12/12
    2012-03-12 00:00:00
    8/2/12
    2012-08-02 00:00:00
    10/26/12
    2012-10-26 00:00:00
    8/8/12
    2012-08-08 00:00:00
    9/27/12
    2012-09-27 00:00:00
    5/8/12
    2012-05-08 00:00:00
    9/20/12
    2012-09-20 00:00:00
    10/11/12
    2012-10-11 00:00:00
    4/4/12
    2012-04-04 00:00:00
    6/28/12
    2012-06-28 00:00:00
    4/12/12
    2012-04-12 00:00:00
    11/21/12
    2012-11-21 00:00:00
    2/5/12
    2012-02-05 00:00:00
    1/19/12
    2012-01-19 00:00:00
    9/14/12
    2012-09-14 00:00:00
    4/16/12
    2012-04-16 00:00:00
    6/20/12
    2012-06-20 00:00:00
    9/27/12
    2012-09-27 00:00:00
    9/6/12
    2012-09-06 00:00:00
    3/1/12
    2012-03-01 00:00:00
    5/16/12
    2012-05-16 00:00:00
    7/21/12
    2012-07-21 00:00:00
    9/7/12
    2012-09-07 00:00:00
    4/11/12
    2012-04-11 00:00:00
    2/10/12
    2012-02-10 00:00:00
    8/17/12
    2012-08-17 00:00:00
    2/14/12
    2012-02-14 00:00:00
    12/18/12
    2012-12-18 00:00:00
    3/15/12
    2012-03-15 00:00:00
    3/7/12
    2012-03-07 00:00:00
    9/26/12
    2012-09-26 00:00:00
    11/13/12
    2012-11-13 00:00:00
    12/19/12
    2012-12-19 00:00:00
    1/18/12
    2012-01-18 00:00:00
    3/27/12
    2012-03-27 00:00:00
    7/26/12
    2012-07-26 00:00:00
    2/1/12
    2012-02-01 00:00:00
    2/24/12
    2012-02-24 00:00:00
    9/20/12
    2012-09-20 00:00:00
    3/29/12
    2012-03-29 00:00:00
    10/11/12
    2012-10-11 00:00:00
    9/20/12
    2012-09-20 00:00:00
    9/6/12
    2012-09-06 00:00:00
    11/9/12
    2012-11-09 00:00:00
    5/17/12
    2012-05-17 00:00:00
    5/15/12
    2012-05-15 00:00:00
    3/10/12
    2012-03-10 00:00:00
    4/19/12
    2012-04-19 00:00:00
    3/1/12
    2012-03-01 00:00:00
    1/12/12
    2012-01-12 00:00:00
    9/9/12
    2012-09-09 00:00:00
    6/6/12
    2012-06-06 00:00:00
    11/2/12
    2012-11-02 00:00:00
    8/29/12
    2012-08-29 00:00:00
    2/23/12
    2012-02-23 00:00:00
    3/9/12
    2012-03-09 00:00:00
    6/13/12
    2012-06-13 00:00:00
    5/17/12
    2012-05-17 00:00:00
    11/7/12
    2012-11-07 00:00:00
    10/25/12
    2012-10-25 00:00:00
    7/26/12
    2012-07-26 00:00:00
    3/12/12
    2012-03-12 00:00:00
    9/7/12
    2012-09-07 00:00:00
    10/19/12
    2012-10-19 00:00:00
    4/27/12
    2012-04-27 00:00:00
    12/20/12
    2012-12-20 00:00:00
    4/12/12
    2012-04-12 00:00:00
    2/24/12
    2012-02-24 00:00:00
    2/1/12
    2012-02-01 00:00:00
    10/4/12
    2012-10-04 00:00:00
    6/14/12
    2012-06-14 00:00:00
    2/9/12
    2012-02-09 00:00:00
    7/6/12
    2012-07-06 00:00:00
    3/15/12
    2012-03-15 00:00:00
    9/11/12
    2012-09-11 00:00:00
    10/19/12
    2012-10-19 00:00:00
    11/2/12
    2012-11-02 00:00:00
    1/13/12
    2012-01-13 00:00:00
    9/7/12
    2012-09-07 00:00:00
    2/2/12
    2012-02-02 00:00:00
    3/9/12
    2012-03-09 00:00:00
    10/23/12
    2012-10-23 00:00:00
    12/26/12
    2012-12-26 00:00:00
    9/11/12
    2012-09-11 00:00:00
    6/29/12
    2012-06-29 00:00:00
    5/11/12
    2012-05-11 00:00:00
    11/22/12
    2012-11-22 00:00:00
    8/30/12
    2012-08-30 00:00:00
    10/10/12
    2012-10-10 00:00:00
    9/20/12
    2012-09-20 00:00:00
    4/4/12
    2012-04-04 00:00:00
    9/25/12
    2012-09-25 00:00:00
    9/14/12
    2012-09-14 00:00:00
    9/1/12
    2012-09-01 00:00:00
    8/9/12
    2012-08-09 00:00:00
    5/25/12
    2012-05-25 00:00:00
    11/21/12
    2012-11-21 00:00:00
    3/2/12
    2012-03-02 00:00:00
    4/13/12
    2012-04-13 00:00:00
    8/7/12
    2012-08-07 00:00:00
    3/8/12
    2012-03-08 00:00:00
    9/2/12
    2012-09-02 00:00:00
    11/2/12
    2012-11-02 00:00:00
    1/6/12
    2012-01-06 00:00:00
    11/25/12
    2012-11-25 00:00:00
    9/9/12
    2012-09-09 00:00:00
    3/7/12
    2012-03-07 00:00:00
    6/16/12
    2012-06-16 00:00:00
    9/7/12
    2012-09-07 00:00:00
    8/27/12
    2012-08-27 00:00:00
    9/21/12
    2012-09-21 00:00:00
    4/20/12
    2012-04-20 00:00:00
    5/24/12
    2012-05-24 00:00:00
    2/14/12
    2012-02-14 00:00:00
    9/7/12
    2012-09-07 00:00:00
    9/21/12
    2012-09-21 00:00:00
    10/18/12
    2012-10-18 00:00:00
    1/26/12
    2012-01-26 00:00:00
    9/8/12
    2012-09-08 00:00:00
    6/22/12
    2012-06-22 00:00:00
    8/15/12
    2012-08-15 00:00:00
    7/25/12
    2012-07-25 00:00:00
    1/21/12
    2012-01-21 00:00:00
    8/3/12
    2012-08-03 00:00:00
    9/23/12
    2012-09-23 00:00:00
    8/2/12
    2012-08-02 00:00:00
    8/27/12
    2012-08-27 00:00:00
    11/30/12
    2012-11-30 00:00:00
    1/17/12
    2012-01-17 00:00:00
    6/1/12
    2012-06-01 00:00:00
    1/19/12
    2012-01-19 00:00:00
    9/11/12
    2012-09-11 00:00:00
    10/14/12
    2012-10-14 00:00:00
    10/25/12
    2012-10-25 00:00:00
    11/29/12
    2012-11-29 00:00:00
    9/10/12
    2012-09-10 00:00:00
    8/9/12
    2012-08-09 00:00:00
    8/24/12
    2012-08-24 00:00:00
    8/6/12
    2012-08-06 00:00:00
    12/25/12
    2012-12-25 00:00:00
    12/19/12
    2012-12-19 00:00:00
    5/3/12
    2012-05-03 00:00:00
    5/2/12
    2012-05-02 00:00:00
    8/30/12
    2012-08-30 00:00:00
    6/22/12
    2012-06-22 00:00:00
    1/13/12
    2012-01-13 00:00:00
    9/28/12
    2012-09-28 00:00:00
    10/17/12
    2012-10-17 00:00:00
    9/7/12
    2012-09-07 00:00:00
    9/27/12
    2012-09-27 00:00:00
    3/30/12
    2012-03-30 00:00:00
    8/24/12
    2012-08-24 00:00:00
    10/17/12
    2012-10-17 00:00:00
    9/13/12
    2012-09-13 00:00:00
    5/22/12
    2012-05-22 00:00:00
    7/25/12
    2012-07-25 00:00:00
    10/4/12
    2012-10-04 00:00:00
    4/1/12
    2012-04-01 00:00:00
    1/29/12
    2012-01-29 00:00:00
    9/19/12
    2012-09-19 00:00:00
    9/13/12
    2012-09-13 00:00:00
    4/27/12
    2012-04-27 00:00:00
    10/25/12
    2012-10-25 00:00:00
    1/1/12
    2012-01-01 00:00:00
    2/17/12
    2012-02-17 00:00:00
    7/27/12
    2012-07-27 00:00:00
    9/8/12
    2012-09-08 00:00:00
    9/18/12
    2012-09-18 00:00:00
    8/31/12
    2012-08-31 00:00:00
    5/18/12
    2012-05-18 00:00:00
    8/5/12
    2012-08-05 00:00:00
    8/15/12
    2012-08-15 00:00:00
    3/6/12
    2012-03-06 00:00:00
    5/11/12
    2012-05-11 00:00:00
    12/25/12
    2012-12-25 00:00:00
    10/12/12
    2012-10-12 00:00:00
    7/28/12
    2012-07-28 00:00:00
    12/7/12
    2012-12-07 00:00:00
    1/6/12
    2012-01-06 00:00:00
    9/28/12
    2012-09-28 00:00:00
    8/5/12
    2012-08-05 00:00:00
    9/21/12
    2012-09-21 00:00:00
    3/28/12
    2012-03-28 00:00:00
    3/19/12
    2012-03-19 00:00:00
    8/22/12
    2012-08-22 00:00:00
    9/12/12
    2012-09-12 00:00:00
    11/9/12
    2012-11-09 00:00:00
    9/14/12
    2012-09-14 00:00:00
    7/27/12
    2012-07-27 00:00:00
    4/16/12
    2012-04-16 00:00:00
    8/17/12
    2012-08-17 00:00:00
    8/22/12
    2012-08-22 00:00:00
    9/6/12
    2012-09-06 00:00:00
    11/4/12
    2012-11-04 00:00:00
    3/8/12
    2012-03-08 00:00:00
    9/8/12
    2012-09-08 00:00:00
    5/2/12
    2012-05-02 00:00:00
    9/13/12
    2012-09-13 00:00:00
    9/13/12
    2012-09-13 00:00:00
    11/8/12
    2012-11-08 00:00:00
    9/7/12
    2012-09-07 00:00:00
    1/23/12
    2012-01-23 00:00:00
    9/14/12
    2012-09-14 00:00:00
    11/9/12
    2012-11-09 00:00:00
    11/30/12
    2012-11-30 00:00:00
    8/23/12
    2012-08-23 00:00:00
    9/12/12
    2012-09-12 00:00:00
    9/26/12
    2012-09-26 00:00:00
    9/6/12
    2012-09-06 00:00:00
    2/17/12
    2012-02-17 00:00:00
    6/7/12
    2012-06-07 00:00:00
    5/29/12
    2012-05-29 00:00:00
    6/29/12
    2012-06-29 00:00:00
    11/9/12
    2012-11-09 00:00:00
    11/5/12
    2012-11-05 00:00:00
    4/16/12
    2012-04-16 00:00:00
    9/11/12
    2012-09-11 00:00:00
    12/1/12
    2012-12-01 00:00:00
    11/10/12
    2012-11-10 00:00:00
    7/2/12
    2012-07-02 00:00:00
    6/5/12
    2012-06-05 00:00:00
    8/31/12
    2012-08-31 00:00:00
    6/8/12
    2012-06-08 00:00:00
    9/14/12
    2012-09-14 00:00:00
    8/9/12
    2012-08-09 00:00:00
    1/1/12
    2012-01-01 00:00:00
    2/9/12
    2012-02-09 00:00:00
    3/11/12
    2012-03-11 00:00:00
    9/9/12
    2012-09-09 00:00:00
    3/2/12
    2012-03-02 00:00:00
    3/10/12
    2012-03-10 00:00:00
    2/3/12
    2012-02-03 00:00:00
    8/23/12
    2012-08-23 00:00:00
    7/1/12
    2012-07-01 00:00:00
    10/9/12
    2012-10-09 00:00:00
    10/5/12
    2012-10-05 00:00:00
    8/21/12
    2012-08-21 00:00:00
    7/12/12
    2012-07-12 00:00:00
    5/18/12
    2012-05-18 00:00:00
    1/1/12
    2012-01-01 00:00:00
    8/3/12
    2012-08-03 00:00:00
    4/12/12
    2012-04-12 00:00:00
    3/28/12
    2012-03-28 00:00:00
    7/3/12
    2012-07-03 00:00:00
    9/7/12
    2012-09-07 00:00:00
    5/10/12
    2012-05-10 00:00:00
    8/16/12
    2012-08-16 00:00:00
    11/16/12
    2012-11-16 00:00:00
    1/13/12
    2012-01-13 00:00:00
    9/5/12
    2012-09-05 00:00:00
    10/13/12
    2012-10-13 00:00:00
    11/9/12
    2012-11-09 00:00:00
    5/10/12
    2012-05-10 00:00:00
    5/25/12
    2012-05-25 00:00:00
    3/10/12
    2012-03-10 00:00:00
    4/20/12
    2012-04-20 00:00:00
    3/15/12
    2012-03-15 00:00:00
    10/12/12
    2012-10-12 00:00:00
    8/8/12
    2012-08-08 00:00:00
    8/31/12
    2012-08-31 00:00:00
    7/31/12
    2012-07-31 00:00:00
    10/9/12
    2012-10-09 00:00:00
    2/1/12
    2012-02-01 00:00:00
    8/19/12
    2012-08-19 00:00:00
    8/24/12
    2012-08-24 00:00:00
    4/23/12
    2012-04-23 00:00:00
    8/10/12
    2012-08-10 00:00:00
    7/11/12
    2012-07-11 00:00:00
    8/21/12
    2012-08-21 00:00:00
    8/8/12
    2012-08-08 00:00:00
    12/7/12
    2012-12-07 00:00:00
    2/17/12
    2012-02-17 00:00:00
    8/10/12
    2012-08-10 00:00:00
    3/9/12
    2012-03-09 00:00:00
    1/12/12
    2012-01-12 00:00:00
    4/3/12
    2012-04-03 00:00:00
    9/7/12
    2012-09-07 00:00:00
    11/25/12
    2012-11-25 00:00:00
    12/6/12
    2012-12-06 00:00:00
    3/13/12
    2012-03-13 00:00:00
    10/12/12
    2012-10-12 00:00:00
    8/4/12
    2012-08-04 00:00:00
    2/11/12
    2012-02-11 00:00:00
    3/6/12
    2012-03-06 00:00:00
    5/25/12
    2012-05-25 00:00:00
    8/4/12
    2012-08-04 00:00:00
    10/20/12
    2012-10-20 00:00:00
    2/25/12
    2012-02-25 00:00:00
    4/24/12
    2012-04-24 00:00:00
    4/6/12
    2012-04-06 00:00:00
    3/10/12
    2012-03-10 00:00:00
    2/23/12
    2012-02-23 00:00:00
    7/16/12
    2012-07-16 00:00:00
    1/13/12
    2012-01-13 00:00:00
    7/6/12
    2012-07-06 00:00:00
    8/27/12
    2012-08-27 00:00:00
    10/26/12
    2012-10-26 00:00:00
    2/23/12
    2012-02-23 00:00:00
    2/12/12
    2012-02-12 00:00:00
    8/3/12
    2012-08-03 00:00:00
    5/3/12
    2012-05-03 00:00:00
    6/16/12
    2012-06-16 00:00:00
    8/1/12
    2012-08-01 00:00:00
    7/11/12
    2012-07-11 00:00:00
    2/3/12
    2012-02-03 00:00:00
    3/22/12
    2012-03-22 00:00:00
    9/13/12
    2012-09-13 00:00:00
    9/5/12
    2012-09-05 00:00:00
    8/21/12
    2012-08-21 00:00:00
    11/27/12
    2012-11-27 00:00:00
    9/15/12
    2012-09-15 00:00:00
    7/13/12
    2012-07-13 00:00:00
    6/29/12
    2012-06-29 00:00:00
    10/19/12
    2012-10-19 00:00:00
    6/29/12
    2012-06-29 00:00:00
    1/1/12
    2012-01-01 00:00:00
    1/3/12
    2012-01-03 00:00:00
    11/2/12
    2012-11-02 00:00:00
    3/27/12
    2012-03-27 00:00:00
    8/23/12
    2012-08-23 00:00:00
    8/10/12
    2012-08-10 00:00:00
    4/26/12
    2012-04-26 00:00:00
    4/20/12
    2012-04-20 00:00:00
    9/12/12
    2012-09-12 00:00:00
    11/29/12
    2012-11-29 00:00:00
    9/12/12
    2012-09-12 00:00:00
    1/17/12
    2012-01-17 00:00:00
    2/11/12
    2012-02-11 00:00:00
    10/12/12
    2012-10-12 00:00:00
    9/12/12
    2012-09-12 00:00:00
    10/12/12
    2012-10-12 00:00:00
    10/31/12
    2012-10-31 00:00:00
    9/11/12
    2012-09-11 00:00:00
    9/12/12
    2012-09-12 00:00:00
    12/28/12
    2012-12-28 00:00:00
    4/27/12
    2012-04-27 00:00:00
    10/27/12
    2012-10-27 00:00:00
    5/31/12
    2012-05-31 00:00:00
    8/29/12
    2012-08-29 00:00:00
    10/5/12
    2012-10-05 00:00:00
    5/16/12
    2012-05-16 00:00:00
    5/20/12
    2012-05-20 00:00:00
    2/28/12
    2012-02-28 00:00:00
    7/3/12
    2012-07-03 00:00:00
    9/12/12
    2012-09-12 00:00:00
    9/21/12
    2012-09-21 00:00:00
    6/6/12
    2012-06-06 00:00:00
    4/13/12
    2012-04-13 00:00:00
    4/26/12
    2012-04-26 00:00:00
    1/22/12
    2012-01-22 00:00:00
    9/9/12
    2012-09-09 00:00:00
    6/15/12
    2012-06-15 00:00:00
    6/12/12
    2012-06-12 00:00:00
    6/29/12
    2012-06-29 00:00:00
    9/29/12
    2012-09-29 00:00:00
    6/6/12
    2012-06-06 00:00:00
    1/16/12
    2012-01-16 00:00:00
    5/25/12
    2012-05-25 00:00:00
    12/7/12
    2012-12-07 00:00:00
    5/27/12
    2012-05-27 00:00:00
    5/4/12
    2012-05-04 00:00:00
    5/18/12
    2012-05-18 00:00:00
    9/7/12
    2012-09-07 00:00:00
    5/11/12
    2012-05-11 00:00:00
    9/19/12
    2012-09-19 00:00:00
    11/15/12
    2012-11-15 00:00:00
    12/17/12
    2012-12-17 00:00:00
    6/7/12
    2012-06-07 00:00:00
    10/9/12
    2012-10-09 00:00:00
    4/20/12
    2012-04-20 00:00:00
    9/9/12
    2012-09-09 00:00:00
    2/13/12
    2012-02-13 00:00:00
    1/1/12
    2012-01-01 00:00:00
    7/3/12
    2012-07-03 00:00:00
    3/5/12
    2012-03-05 00:00:00
    1/2/12
    2012-01-02 00:00:00
    2/10/12
    2012-02-10 00:00:00
    10/26/12
    2012-10-26 00:00:00
    3/6/12
    2012-03-06 00:00:00
    9/3/12
    2012-09-03 00:00:00
    8/10/12
    2012-08-10 00:00:00
    11/7/12
    2012-11-07 00:00:00
    5/29/12
    2012-05-29 00:00:00
    6/12/12
    2012-06-12 00:00:00
    3/2/12
    2012-03-02 00:00:00
    10/1/12
    2012-10-01 00:00:00
    6/14/12
    2012-06-14 00:00:00
    9/7/12
    2012-09-07 00:00:00
    4/20/12
    2012-04-20 00:00:00
    7/19/12
    2012-07-19 00:00:00
    8/23/12
    2012-08-23 00:00:00
    12/14/12
    2012-12-14 00:00:00
    11/24/12
    2012-11-24 00:00:00
    9/27/12
    2012-09-27 00:00:00
    8/17/12
    2012-08-17 00:00:00
    6/1/12
    2012-06-01 00:00:00
    5/11/12
    2012-05-11 00:00:00
    9/7/12
    2012-09-07 00:00:00
    3/15/12
    2012-03-15 00:00:00
    11/2/12
    2012-11-02 00:00:00
    1/21/12
    2012-01-21 00:00:00
    12/1/12
    2012-12-01 00:00:00
    4/27/12
    2012-04-27 00:00:00
    7/1/12
    2012-07-01 00:00:00
    1/1/12
    2012-01-01 00:00:00
    9/1/12
    2012-09-01 00:00:00
    1/1/12
    2012-01-01 00:00:00
    11/30/12
    2012-11-30 00:00:00
    9/14/12
    2012-09-14 00:00:00
    6/15/12
    2012-06-15 00:00:00
    6/30/12
    2012-06-30 00:00:00
    5/15/12
    2012-05-15 00:00:00
    4/18/12
    2012-04-18 00:00:00
    4/22/12
    2012-04-22 00:00:00
    10/27/12
    2012-10-27 00:00:00
    12/25/12
    2012-12-25 00:00:00
    2/14/12
    2012-02-14 00:00:00
    7/8/12
    2012-07-08 00:00:00
    4/27/12
    2012-04-27 00:00:00
    4/21/12
    2012-04-21 00:00:00
    6/15/12
    2012-06-15 00:00:00
    5/16/12
    2012-05-16 00:00:00
    12/15/12
    2012-12-15 00:00:00
    2/8/12
    2012-02-08 00:00:00
    3/30/12
    2012-03-30 00:00:00
    10/23/12
    2012-10-23 00:00:00
    4/5/12
    2012-04-05 00:00:00
    6/15/12
    2012-06-15 00:00:00
    12/21/12
    2012-12-21 00:00:00
    8/18/12
    2012-08-18 00:00:00
    9/1/12
    2012-09-01 00:00:00
    10/12/12
    2012-10-12 00:00:00
    12/14/12
    2012-12-14 00:00:00
    10/3/12
    2012-10-03 00:00:00
    4/20/12
    2012-04-20 00:00:00
    6/5/12
    2012-06-05 00:00:00
    7/12/12
    2012-07-12 00:00:00
    11/15/12
    2012-11-15 00:00:00
    7/20/12
    2012-07-20 00:00:00
    9/23/12
    2012-09-23 00:00:00
    12/9/12
    2012-12-09 00:00:00
    10/12/12
    2012-10-12 00:00:00
    11/2/12
    2012-11-02 00:00:00
    5/18/12
    2012-05-18 00:00:00
    1/20/12
    2012-01-20 00:00:00
    2/16/12
    2012-02-16 00:00:00
    7/9/12
    2012-07-09 00:00:00
    5/28/12
    2012-05-28 00:00:00
    4/12/12
    2012-04-12 00:00:00
    11/3/12
    2012-11-03 00:00:00
    11/11/12
    2012-11-11 00:00:00
    6/1/12
    2012-06-01 00:00:00
    5/24/12
    2012-05-24 00:00:00
    9/28/12
    2012-09-28 00:00:00
    5/24/12
    2012-05-24 00:00:00
    9/25/12
    2012-09-25 00:00:00
    9/10/12
    2012-09-10 00:00:00
    6/8/12
    2012-06-08 00:00:00
    1/1/12
    2012-01-01 00:00:00
    6/1/12
    2012-06-01 00:00:00
    5/19/12
    2012-05-19 00:00:00
    5/4/12
    2012-05-04 00:00:00
    3/29/12
    2012-03-29 00:00:00
    3/7/12
    2012-03-07 00:00:00
    10/10/12
    2012-10-10 00:00:00
    4/26/12
    2012-04-26 00:00:00
    12/20/12
    2012-12-20 00:00:00
    11/13/12
    2012-11-13 00:00:00
    2/23/12
    2012-02-23 00:00:00
    3/9/12
    2012-03-09 00:00:00
    6/15/12
    2012-06-15 00:00:00
    8/17/12
    2012-08-17 00:00:00
    3/2/12
    2012-03-02 00:00:00
    10/11/12
    2012-10-11 00:00:00
    10/12/12
    2012-10-12 00:00:00
    5/21/12
    2012-05-21 00:00:00
    1/1/12
    2012-01-01 00:00:00
    6/15/12
    2012-06-15 00:00:00
    5/31/12
    2012-05-31 00:00:00
    1/31/12
    2012-01-31 00:00:00
    8/25/12
    2012-08-25 00:00:00
    8/30/12
    2012-08-30 00:00:00
    7/15/12
    2012-07-15 00:00:00
    10/22/12
    2012-10-22 00:00:00
    11/9/12
    2012-11-09 00:00:00
    8/8/12
    2012-08-08 00:00:00
    7/8/12
    2012-07-08 00:00:00
    9/9/12
    2012-09-09 00:00:00
    12/7/12
    2012-12-07 00:00:00
    9/29/12
    2012-09-29 00:00:00
    9/23/12
    2012-09-23 00:00:00
    9/14/12
    2012-09-14 00:00:00
    8/18/12
    2012-08-18 00:00:00
    4/24/12
    2012-04-24 00:00:00
    9/8/12
    2012-09-08 00:00:00
    10/12/12
    2012-10-12 00:00:00
    11/23/12
    2012-11-23 00:00:00
    8/16/12
    2012-08-16 00:00:00
    5/1/12
    2012-05-01 00:00:00
    11/30/12
    2012-11-30 00:00:00
    9/21/12
    2012-09-21 00:00:00
    3/2/12
    2012-03-02 00:00:00
    9/9/12
    2012-09-09 00:00:00
    5/17/12
    2012-05-17 00:00:00
    5/10/12
    2012-05-10 00:00:00
    5/25/12
    2012-05-25 00:00:00
    10/17/12
    2012-10-17 00:00:00
    7/13/12
    2012-07-13 00:00:00
    1/1/12
    2012-01-01 00:00:00
    8/22/12
    2012-08-22 00:00:00
    10/19/12
    2012-10-19 00:00:00
    10/7/12
    2012-10-07 00:00:00
    7/6/12
    2012-07-06 00:00:00
    11/28/12
    2012-11-28 00:00:00
    8/2/12
    2012-08-02 00:00:00
    9/21/12
    2012-09-21 00:00:00
    9/7/12
    2012-09-07 00:00:00
    12/23/12
    2012-12-23 00:00:00
    8/24/12
    2012-08-24 00:00:00
    7/31/12
    2012-07-31 00:00:00
    11/17/12
    2012-11-17 00:00:00
    6/28/12
    2012-06-28 00:00:00
    8/8/12
    2012-08-08 00:00:00
    4/11/12
    2012-04-11 00:00:00
    3/20/12
    2012-03-20 00:00:00
    10/25/12
    2012-10-25 00:00:00
    12/14/12
    2012-12-14 00:00:00
    4/22/12
    2012-04-22 00:00:00
    1/31/12
    2012-01-31 00:00:00
    11/29/12
    2012-11-29 00:00:00
    12/18/12
    2012-12-18 00:00:00
    4/15/12
    2012-04-15 00:00:00
    12/25/12
    2012-12-25 00:00:00
    9/8/12
    2012-09-08 00:00:00
    10/26/12
    2012-10-26 00:00:00
    5/11/12
    2012-05-11 00:00:00
    2/1/12
    2012-02-01 00:00:00
    9/25/12
    2012-09-25 00:00:00
    4/26/12
    2012-04-26 00:00:00
    11/6/12
    2012-11-06 00:00:00
    11/3/12
    2012-11-03 00:00:00
    10/5/12
    2012-10-05 00:00:00
    10/4/12
    2012-10-04 00:00:00
    10/5/12
    2012-10-05 00:00:00
    1/1/12
    2012-01-01 00:00:00
    6/15/12
    2012-06-15 00:00:00
    11/2/12
    2012-11-02 00:00:00
    11/9/12
    2012-11-09 00:00:00
    6/20/12
    2012-06-20 00:00:00
    6/13/12
    2012-06-13 00:00:00
    10/30/12
    2012-10-30 00:00:00
    10/5/12
    2012-10-05 00:00:00
    1/31/12
    2012-01-31 00:00:00
    11/25/12
    2012-11-25 00:00:00
    11/23/12
    2012-11-23 00:00:00
    11/9/12
    2012-11-09 00:00:00
    5/16/12
    2012-05-16 00:00:00
    10/16/12
    2012-10-16 00:00:00
    12/24/12
    2012-12-24 00:00:00
    9/21/12
    2012-09-21 00:00:00
    1/1/12
    2012-01-01 00:00:00
    1/22/12
    2012-01-22 00:00:00
    7/21/12
    2012-07-21 00:00:00
    1/28/12
    2012-01-28 00:00:00
    1/21/12
    2012-01-21 00:00:00
    5/24/12
    2012-05-24 00:00:00
    1/20/12
    2012-01-20 00:00:00
    12/1/03
    2003-12-01 00:00:00
    9/19/03
    2003-09-19 00:00:00
    7/9/03
    2003-07-09 00:00:00
    10/10/03
    2003-10-10 00:00:00
    11/5/03
    2003-11-05 00:00:00
    5/15/03
    2003-05-15 00:00:00
    5/30/03
    2003-05-30 00:00:00
    5/23/03
    2003-05-23 00:00:00
    9/7/03
    2003-09-07 00:00:00
    10/9/03
    2003-10-09 00:00:00
    7/2/03
    2003-07-02 00:00:00
    9/1/03
    2003-09-01 00:00:00
    12/25/03
    2003-12-25 00:00:00
    2/14/03
    2003-02-14 00:00:00
    8/1/03
    2003-08-01 00:00:00
    7/11/03
    2003-07-11 00:00:00
    5/30/03
    2003-05-30 00:00:00
    2/7/03
    2003-02-07 00:00:00
    8/31/03
    2003-08-31 00:00:00
    11/25/03
    2003-11-25 00:00:00
    4/6/03
    2003-04-06 00:00:00
    10/20/03
    2003-10-20 00:00:00
    12/25/03
    2003-12-25 00:00:00
    10/24/03
    2003-10-24 00:00:00
    11/24/03
    2003-11-24 00:00:00
    4/4/03
    2003-04-04 00:00:00
    10/3/03
    2003-10-03 00:00:00
    12/5/03
    2003-12-05 00:00:00
    7/18/03
    2003-07-18 00:00:00
    7/21/03
    2003-07-21 00:00:00
    5/30/03
    2003-05-30 00:00:00
    4/25/03
    2003-04-25 00:00:00
    11/14/03
    2003-11-14 00:00:00
    12/19/03
    2003-12-19 00:00:00
    10/7/03
    2003-10-07 00:00:00
    5/9/03
    2003-05-09 00:00:00
    4/11/03
    2003-04-11 00:00:00
    3/28/03
    2003-03-28 00:00:00
    6/19/03
    2003-06-19 00:00:00
    9/11/03
    2003-09-11 00:00:00
    1/21/03
    2003-01-21 00:00:00
    2/21/03
    2003-02-21 00:00:00
    2/7/03
    2003-02-07 00:00:00
    2/6/03
    2003-02-06 00:00:00
    8/8/03
    2003-08-08 00:00:00
    3/7/03
    2003-03-07 00:00:00
    7/2/03
    2003-07-02 00:00:00
    2/25/03
    2003-02-25 00:00:00
    11/21/03
    2003-11-21 00:00:00
    8/3/03
    2003-08-03 00:00:00
    7/2/03
    2003-07-02 00:00:00
    11/26/03
    2003-11-26 00:00:00
    12/25/03
    2003-12-25 00:00:00
    6/27/03
    2003-06-27 00:00:00
    12/12/03
    2003-12-12 00:00:00
    11/25/03
    2003-11-25 00:00:00
    9/26/03
    2003-09-26 00:00:00
    11/21/03
    2003-11-21 00:00:00
    9/6/03
    2003-09-06 00:00:00
    1/31/03
    2003-01-31 00:00:00
    8/18/03
    2003-08-18 00:00:00
    10/17/03
    2003-10-17 00:00:00
    1/25/03
    2003-01-25 00:00:00
    12/24/03
    2003-12-24 00:00:00
    12/24/03
    2003-12-24 00:00:00
    5/16/03
    2003-05-16 00:00:00
    4/18/03
    2003-04-18 00:00:00
    8/15/03
    2003-08-15 00:00:00
    8/20/03
    2003-08-20 00:00:00
    8/15/03
    2003-08-15 00:00:00
    2/28/03
    2003-02-28 00:00:00
    11/4/03
    2003-11-04 00:00:00
    11/14/03
    2003-11-14 00:00:00
    4/18/03
    2003-04-18 00:00:00
    9/26/03
    2003-09-26 00:00:00
    9/28/03
    2003-09-28 00:00:00
    6/5/03
    2003-06-05 00:00:00
    3/21/03
    2003-03-21 00:00:00
    10/21/03
    2003-10-21 00:00:00
    4/25/03
    2003-04-25 00:00:00
    3/27/03
    2003-03-27 00:00:00
    11/24/03
    2003-11-24 00:00:00
    8/29/03
    2003-08-29 00:00:00
    1/9/03
    2003-01-09 00:00:00
    4/30/03
    2003-04-30 00:00:00
    10/22/03
    2003-10-22 00:00:00
    5/3/03
    2003-05-03 00:00:00
    9/1/03
    2003-09-01 00:00:00
    7/22/03
    2003-07-22 00:00:00
    10/3/03
    2003-10-03 00:00:00
    8/8/03
    2003-08-08 00:00:00
    4/25/03
    2003-04-25 00:00:00
    3/21/03
    2003-03-21 00:00:00
    9/15/03
    2003-09-15 00:00:00
    3/22/03
    2003-03-22 00:00:00
    3/28/03
    2003-03-28 00:00:00
    7/25/03
    2003-07-25 00:00:00
    5/8/03
    2003-05-08 00:00:00
    9/5/03
    2003-09-05 00:00:00
    9/2/03
    2003-09-02 00:00:00
    3/11/03
    2003-03-11 00:00:00
    12/19/03
    2003-12-19 00:00:00
    8/26/03
    2003-08-26 00:00:00
    8/8/03
    2003-08-08 00:00:00
    9/26/03
    2003-09-26 00:00:00
    8/15/03
    2003-08-15 00:00:00
    10/9/03
    2003-10-09 00:00:00
    9/2/03
    2003-09-02 00:00:00
    10/24/03
    2003-10-24 00:00:00
    1/17/03
    2003-01-17 00:00:00
    11/16/03
    2003-11-16 00:00:00
    9/9/03
    2003-09-09 00:00:00
    2/23/03
    2003-02-23 00:00:00
    3/7/03
    2003-03-07 00:00:00
    10/29/03
    2003-10-29 00:00:00
    8/1/03
    2003-08-01 00:00:00
    2/21/03
    2003-02-21 00:00:00
    1/31/03
    2003-01-31 00:00:00
    11/14/03
    2003-11-14 00:00:00
    6/13/03
    2003-06-13 00:00:00
    8/31/03
    2003-08-31 00:00:00
    1/17/03
    2003-01-17 00:00:00
    10/6/03
    2003-10-06 00:00:00
    5/17/03
    2003-05-17 00:00:00
    1/18/03
    2003-01-18 00:00:00
    4/10/03
    2003-04-10 00:00:00
    8/15/03
    2003-08-15 00:00:00
    9/5/03
    2003-09-05 00:00:00
    10/23/03
    2003-10-23 00:00:00
    2/21/03
    2003-02-21 00:00:00
    8/27/03
    2003-08-27 00:00:00
    5/18/03
    2003-05-18 00:00:00
    9/5/03
    2003-09-05 00:00:00
    4/11/03
    2003-04-11 00:00:00
    1/30/03
    2003-01-30 00:00:00
    5/18/03
    2003-05-18 00:00:00
    10/31/03
    2003-10-31 00:00:00
    8/6/03
    2003-08-06 00:00:00
    4/11/03
    2003-04-11 00:00:00
    1/17/03
    2003-01-17 00:00:00
    12/30/03
    2003-12-30 00:00:00
    10/24/03
    2003-10-24 00:00:00
    9/26/03
    2003-09-26 00:00:00
    9/19/03
    2003-09-19 00:00:00
    6/13/03
    2003-06-13 00:00:00
    12/8/03
    2003-12-08 00:00:00
    1/19/03
    2003-01-19 00:00:00
    12/2/03
    2003-12-02 00:00:00
    6/9/03
    2003-06-09 00:00:00
    5/18/03
    2003-05-18 00:00:00
    1/1/03
    2003-01-01 00:00:00
    1/18/03
    2003-01-18 00:00:00
    4/16/03
    2003-04-16 00:00:00
    8/15/03
    2003-08-15 00:00:00
    9/19/03
    2003-09-19 00:00:00
    8/5/03
    2003-08-05 00:00:00
    3/16/03
    2003-03-16 00:00:00
    7/10/03
    2003-07-10 00:00:00
    1/26/03
    2003-01-26 00:00:00
    12/12/03
    2003-12-12 00:00:00
    9/17/03
    2003-09-17 00:00:00
    2/19/03
    2003-02-19 00:00:00
    8/29/03
    2003-08-29 00:00:00
    1/24/03
    2003-01-24 00:00:00
    6/7/03
    2003-06-07 00:00:00
    1/1/03
    2003-01-01 00:00:00
    5/23/03
    2003-05-23 00:00:00
    7/18/03
    2003-07-18 00:00:00
    4/14/03
    2003-04-14 00:00:00
    6/25/03
    2003-06-25 00:00:00
    9/30/03
    2003-09-30 00:00:00
    1/1/03
    2003-01-01 00:00:00
    12/7/03
    2003-12-07 00:00:00
    9/11/03
    2003-09-11 00:00:00
    3/11/03
    2003-03-11 00:00:00
    3/14/03
    2003-03-14 00:00:00
    1/24/03
    2003-01-24 00:00:00
    3/4/03
    2003-03-04 00:00:00
    12/12/03
    2003-12-12 00:00:00
    9/2/03
    2003-09-02 00:00:00
    1/30/03
    2003-01-30 00:00:00
    10/3/03
    2003-10-03 00:00:00
    3/7/03
    2003-03-07 00:00:00
    11/20/03
    2003-11-20 00:00:00
    6/6/03
    2003-06-06 00:00:00
    5/30/03
    2003-05-30 00:00:00
    3/12/03
    2003-03-12 00:00:00
    5/16/03
    2003-05-16 00:00:00
    10/15/03
    2003-10-15 00:00:00
    1/31/03
    2003-01-31 00:00:00
    4/3/03
    2003-04-03 00:00:00
    5/23/03
    2003-05-23 00:00:00
    10/23/03
    2003-10-23 00:00:00
    4/10/03
    2003-04-10 00:00:00
    8/15/03
    2003-08-15 00:00:00
    3/18/03
    2003-03-18 00:00:00
    5/18/03
    2003-05-18 00:00:00
    5/9/03
    2003-05-09 00:00:00
    11/26/03
    2003-11-26 00:00:00
    12/10/03
    2003-12-10 00:00:00
    3/19/03
    2003-03-19 00:00:00
    1/13/03
    2003-01-13 00:00:00
    9/9/03
    2003-09-09 00:00:00
    1/1/03
    2003-01-01 00:00:00
    5/15/03
    2003-05-15 00:00:00
    2/1/03
    2003-02-01 00:00:00
    8/14/03
    2003-08-14 00:00:00
    8/15/03
    2003-08-15 00:00:00
    1/18/03
    2003-01-18 00:00:00
    7/4/03
    2003-07-04 00:00:00
    9/8/03
    2003-09-08 00:00:00
    9/10/03
    2003-09-10 00:00:00
    5/2/03
    2003-05-02 00:00:00
    6/2/03
    2003-06-02 00:00:00
    3/25/03
    2003-03-25 00:00:00
    9/10/03
    2003-09-10 00:00:00
    5/9/03
    2003-05-09 00:00:00
    1/1/03
    2003-01-01 00:00:00
    4/16/03
    2003-04-16 00:00:00
    1/1/03
    2003-01-01 00:00:00
    9/9/03
    2003-09-09 00:00:00
    5/1/03
    2003-05-01 00:00:00
    7/19/03
    2003-07-19 00:00:00
    12/30/03
    2003-12-30 00:00:00
    1/17/03
    2003-01-17 00:00:00
    10/10/03
    2003-10-10 00:00:00
    9/5/03
    2003-09-05 00:00:00
    7/11/03
    2003-07-11 00:00:00
    10/29/03
    2003-10-29 00:00:00
    6/13/03
    2003-06-13 00:00:00
    5/31/03
    2003-05-31 00:00:00
    8/1/03
    2003-08-01 00:00:00
    12/14/03
    2003-12-14 00:00:00
    8/21/03
    2003-08-21 00:00:00
    5/16/03
    2003-05-16 00:00:00
    4/24/03
    2003-04-24 00:00:00
    9/5/03
    2003-09-05 00:00:00
    10/23/03
    2003-10-23 00:00:00
    1/1/03
    2003-01-01 00:00:00
    4/5/03
    2003-04-05 00:00:00
    9/9/03
    2003-09-09 00:00:00
    4/19/03
    2003-04-19 00:00:00
    4/13/03
    2003-04-13 00:00:00
    9/5/03
    2003-09-05 00:00:00
    12/9/03
    2003-12-09 00:00:00
    1/1/03
    2003-01-01 00:00:00
    6/16/03
    2003-06-16 00:00:00
    6/20/03
    2003-06-20 00:00:00
    2/7/03
    2003-02-07 00:00:00
    12/10/03
    2003-12-10 00:00:00
    7/1/03
    2003-07-01 00:00:00
    1/1/03
    2003-01-01 00:00:00
    9/4/03
    2003-09-04 00:00:00
    1/1/03
    2003-01-01 00:00:00
    5/12/03
    2003-05-12 00:00:00
    3/21/03
    2003-03-21 00:00:00
    5/18/03
    2003-05-18 00:00:00
    5/9/03
    2003-05-09 00:00:00
    10/17/03
    2003-10-17 00:00:00
    6/1/03
    2003-06-01 00:00:00
    1/23/03
    2003-01-23 00:00:00
    11/7/03
    2003-11-07 00:00:00
    4/7/03
    2003-04-07 00:00:00
    6/12/03
    2003-06-12 00:00:00
    1/1/03
    2003-01-01 00:00:00
    1/1/03
    2003-01-01 00:00:00
    3/26/03
    2003-03-26 00:00:00
    6/25/03
    2003-06-25 00:00:00
    1/1/03
    2003-01-01 00:00:00
    4/2/03
    2003-04-02 00:00:00
    10/21/03
    2003-10-21 00:00:00
    4/1/03
    2003-04-01 00:00:00
    5/7/03
    2003-05-07 00:00:00
    12/25/03
    2003-12-25 00:00:00
    9/17/03
    2003-09-17 00:00:00
    6/27/03
    2003-06-27 00:00:00
    11/23/03
    2003-11-23 00:00:00
    6/1/03
    2003-06-01 00:00:00
    6/3/03
    2003-06-03 00:00:00
    6/1/03
    2003-06-01 00:00:00
    4/25/03
    2003-04-25 00:00:00
    11/17/97
    1997-11-17 00:00:00
    11/18/97
    1997-11-18 00:00:00
    6/26/97
    1997-06-26 00:00:00
    7/1/97
    1997-07-01 00:00:00
    9/7/97
    1997-09-07 00:00:00
    5/7/97
    1997-05-07 00:00:00
    6/20/97
    1997-06-20 00:00:00
    8/4/97
    1997-08-04 00:00:00
    12/11/97
    1997-12-11 00:00:00
    12/5/97
    1997-12-05 00:00:00
    12/24/97
    1997-12-24 00:00:00
    11/12/97
    1997-11-12 00:00:00
    9/12/97
    1997-09-12 00:00:00
    8/15/97
    1997-08-15 00:00:00
    11/20/97
    1997-11-20 00:00:00
    12/12/97
    1997-12-12 00:00:00
    5/2/97
    1997-05-02 00:00:00
    11/6/97
    1997-11-06 00:00:00
    8/13/97
    1997-08-13 00:00:00
    6/1/97
    1997-06-01 00:00:00
    6/27/97
    1997-06-27 00:00:00
    2/27/97
    1997-02-27 00:00:00
    10/17/97
    1997-10-17 00:00:00
    10/17/97
    1997-10-17 00:00:00
    4/11/97
    1997-04-11 00:00:00
    4/4/97
    1997-04-04 00:00:00
    2/7/97
    1997-02-07 00:00:00
    6/12/97
    1997-06-12 00:00:00
    9/19/97
    1997-09-19 00:00:00
    12/9/97
    1997-12-09 00:00:00
    3/21/97
    1997-03-21 00:00:00
    11/11/97
    1997-11-11 00:00:00
    7/11/97
    1997-07-11 00:00:00
    9/27/97
    1997-09-27 00:00:00
    11/18/97
    1997-11-18 00:00:00
    4/25/97
    1997-04-25 00:00:00
    11/13/97
    1997-11-13 00:00:00
    4/11/97
    1997-04-11 00:00:00
    1/1/97
    1997-01-01 00:00:00
    11/26/97
    1997-11-26 00:00:00
    4/3/97
    1997-04-03 00:00:00
    8/7/97
    1997-08-07 00:00:00
    7/23/97
    1997-07-23 00:00:00
    7/15/97
    1997-07-15 00:00:00
    12/25/97
    1997-12-25 00:00:00
    12/19/97
    1997-12-19 00:00:00
    5/2/97
    1997-05-02 00:00:00
    9/12/97
    1997-09-12 00:00:00
    6/19/97
    1997-06-19 00:00:00
    9/11/97
    1997-09-11 00:00:00
    1/15/97
    1997-01-15 00:00:00
    11/7/97
    1997-11-07 00:00:00
    3/18/97
    1997-03-18 00:00:00
    6/18/97
    1997-06-18 00:00:00
    10/21/97
    1997-10-21 00:00:00
    7/29/97
    1997-07-29 00:00:00
    3/12/97
    1997-03-12 00:00:00
    9/10/97
    1997-09-10 00:00:00
    4/18/97
    1997-04-18 00:00:00
    9/6/97
    1997-09-06 00:00:00
    10/3/97
    1997-10-03 00:00:00
    2/14/97
    1997-02-14 00:00:00
    2/13/97
    1997-02-13 00:00:00
    10/3/97
    1997-10-03 00:00:00
    9/12/97
    1997-09-12 00:00:00
    8/22/97
    1997-08-22 00:00:00
    7/29/97
    1997-07-29 00:00:00
    2/14/97
    1997-02-14 00:00:00
    7/31/97
    1997-07-31 00:00:00
    9/26/97
    1997-09-26 00:00:00
    9/9/97
    1997-09-09 00:00:00
    12/25/97
    1997-12-25 00:00:00
    12/25/97
    1997-12-25 00:00:00
    8/22/97
    1997-08-22 00:00:00
    1/10/97
    1997-01-10 00:00:00
    3/7/97
    1997-03-07 00:00:00
    9/1/97
    1997-09-01 00:00:00
    1/31/97
    1997-01-31 00:00:00
    10/24/97
    1997-10-24 00:00:00
    9/6/97
    1997-09-06 00:00:00
    8/26/97
    1997-08-26 00:00:00
    11/24/97
    1997-11-24 00:00:00
    8/27/97
    1997-08-27 00:00:00
    11/29/97
    1997-11-29 00:00:00
    5/9/97
    1997-05-09 00:00:00
    9/5/97
    1997-09-05 00:00:00
    4/4/97
    1997-04-04 00:00:00
    12/19/97
    1997-12-19 00:00:00
    5/14/97
    1997-05-14 00:00:00
    8/5/97
    1997-08-05 00:00:00
    12/25/97
    1997-12-25 00:00:00
    10/9/97
    1997-10-09 00:00:00
    8/15/97
    1997-08-15 00:00:00
    2/13/97
    1997-02-13 00:00:00
    3/21/97
    1997-03-21 00:00:00
    12/19/97
    1997-12-19 00:00:00
    9/27/97
    1997-09-27 00:00:00
    1/1/97
    1997-01-01 00:00:00
    8/1/97
    1997-08-01 00:00:00
    4/25/97
    1997-04-25 00:00:00
    5/23/97
    1997-05-23 00:00:00
    7/18/97
    1997-07-18 00:00:00
    3/26/97
    1997-03-26 00:00:00
    6/11/97
    1997-06-11 00:00:00
    11/21/97
    1997-11-21 00:00:00
    9/9/97
    1997-09-09 00:00:00
    1/16/97
    1997-01-16 00:00:00
    9/28/97
    1997-09-28 00:00:00
    1/29/97
    1997-01-29 00:00:00
    7/18/97
    1997-07-18 00:00:00
    9/19/97
    1997-09-19 00:00:00
    12/11/97
    1997-12-11 00:00:00
    10/30/97
    1997-10-30 00:00:00
    3/27/97
    1997-03-27 00:00:00
    1/17/97
    1997-01-17 00:00:00
    12/5/97
    1997-12-05 00:00:00
    1/1/97
    1997-01-01 00:00:00
    8/22/97
    1997-08-22 00:00:00
    11/14/97
    1997-11-14 00:00:00
    1/9/97
    1997-01-09 00:00:00
    11/26/97
    1997-11-26 00:00:00
    11/15/97
    1997-11-15 00:00:00
    1/19/97
    1997-01-19 00:00:00
    1/24/97
    1997-01-24 00:00:00
    8/15/97
    1997-08-15 00:00:00
    2/7/97
    1997-02-07 00:00:00
    11/21/97
    1997-11-21 00:00:00
    1/1/97
    1997-01-01 00:00:00
    3/7/97
    1997-03-07 00:00:00
    9/6/97
    1997-09-06 00:00:00
    7/18/97
    1997-07-18 00:00:00
    7/13/97
    1997-07-13 00:00:00
    10/17/97
    1997-10-17 00:00:00
    5/18/97
    1997-05-18 00:00:00
    8/1/97
    1997-08-01 00:00:00
    4/4/97
    1997-04-04 00:00:00
    8/8/97
    1997-08-08 00:00:00
    5/23/97
    1997-05-23 00:00:00
    2/7/97
    1997-02-07 00:00:00
    9/7/97
    1997-09-07 00:00:00
    12/31/97
    1997-12-31 00:00:00
    8/29/97
    1997-08-29 00:00:00
    10/10/97
    1997-10-10 00:00:00
    5/9/97
    1997-05-09 00:00:00
    9/8/97
    1997-09-08 00:00:00
    7/2/97
    1997-07-02 00:00:00
    5/8/97
    1997-05-08 00:00:00
    11/21/97
    1997-11-21 00:00:00
    11/7/97
    1997-11-07 00:00:00
    7/30/97
    1997-07-30 00:00:00
    11/21/97
    1997-11-21 00:00:00
    8/17/97
    1997-08-17 00:00:00
    4/4/97
    1997-04-04 00:00:00
    7/11/97
    1997-07-11 00:00:00
    5/30/97
    1997-05-30 00:00:00
    3/28/97
    1997-03-28 00:00:00
    12/25/97
    1997-12-25 00:00:00
    12/3/97
    1997-12-03 00:00:00
    9/26/97
    1997-09-26 00:00:00
    8/28/97
    1997-08-28 00:00:00
    2/21/97
    1997-02-21 00:00:00
    5/23/97
    1997-05-23 00:00:00
    10/31/97
    1997-10-31 00:00:00
    3/28/97
    1997-03-28 00:00:00
    5/1/97
    1997-05-01 00:00:00
    2/26/97
    1997-02-26 00:00:00
    10/8/97
    1997-10-08 00:00:00
    8/29/97
    1997-08-29 00:00:00
    10/26/97
    1997-10-26 00:00:00
    1/1/97
    1997-01-01 00:00:00
    10/10/97
    1997-10-10 00:00:00
    10/17/97
    1997-10-17 00:00:00
    4/1/97
    1997-04-01 00:00:00
    11/2/97
    1997-11-02 00:00:00
    7/2/97
    1997-07-02 00:00:00
    1/1/97
    1997-01-01 00:00:00
    10/18/97
    1997-10-18 00:00:00
    4/18/97
    1997-04-18 00:00:00
    8/29/97
    1997-08-29 00:00:00
    10/5/97
    1997-10-05 00:00:00
    8/22/97
    1997-08-22 00:00:00
    4/4/97
    1997-04-04 00:00:00
    7/25/97
    1997-07-25 00:00:00
    11/1/97
    1997-11-01 00:00:00
    9/26/97
    1997-09-26 00:00:00
    4/10/97
    1997-04-10 00:00:00
    4/11/97
    1997-04-11 00:00:00
    1/28/97
    1997-01-28 00:00:00
    11/18/97
    1997-11-18 00:00:00
    10/15/97
    1997-10-15 00:00:00
    9/7/97
    1997-09-07 00:00:00
    3/12/97
    1997-03-12 00:00:00
    11/27/13
    2013-11-27 00:00:00
    9/27/13
    2013-09-27 00:00:00
    10/29/13
    2013-10-29 00:00:00
    4/18/13
    2013-04-18 00:00:00
    5/29/13
    2013-05-29 00:00:00
    11/15/13
    2013-11-15 00:00:00
    12/25/13
    2013-12-25 00:00:00
    7/11/13
    2013-07-11 00:00:00
    4/10/13
    2013-04-10 00:00:00
    12/11/13
    2013-12-11 00:00:00
    6/12/13
    2013-06-12 00:00:00
    12/18/13
    2013-12-18 00:00:00
    6/25/13
    2013-06-25 00:00:00
    5/5/13
    2013-05-05 00:00:00
    10/18/13
    2013-10-18 00:00:00
    9/2/13
    2013-09-02 00:00:00
    7/18/13
    2013-07-18 00:00:00
    5/10/13
    2013-05-10 00:00:00
    12/25/13
    2013-12-25 00:00:00
    8/7/13
    2013-08-07 00:00:00
    2/6/13
    2013-02-06 00:00:00
    5/31/13
    2013-05-31 00:00:00
    6/20/13
    2013-06-20 00:00:00
    6/20/13
    2013-06-20 00:00:00
    12/18/13
    2013-12-18 00:00:00
    10/23/13
    2013-10-23 00:00:00
    8/7/13
    2013-08-07 00:00:00
    8/2/13
    2013-08-02 00:00:00
    8/16/13
    2013-08-16 00:00:00
    9/2/13
    2013-09-02 00:00:00
    3/26/13
    2013-03-26 00:00:00
    10/6/13
    2013-10-06 00:00:00
    3/20/13
    2013-03-20 00:00:00
    8/7/13
    2013-08-07 00:00:00
    9/18/13
    2013-09-18 00:00:00
    7/18/13
    2013-07-18 00:00:00
    12/12/13
    2013-12-12 00:00:00
    1/31/13
    2013-01-31 00:00:00
    7/26/13
    2013-07-26 00:00:00
    1/17/13
    2013-01-17 00:00:00
    12/6/13
    2013-12-06 00:00:00
    1/18/13
    2013-01-18 00:00:00
    8/21/13
    2013-08-21 00:00:00
    5/15/13
    2013-05-15 00:00:00
    12/24/13
    2013-12-24 00:00:00
    12/25/13
    2013-12-25 00:00:00
    9/7/13
    2013-09-07 00:00:00
    7/18/13
    2013-07-18 00:00:00
    7/18/13
    2013-07-18 00:00:00
    8/9/13
    2013-08-09 00:00:00
    1/24/13
    2013-01-24 00:00:00
    5/16/13
    2013-05-16 00:00:00
    3/14/13
    2013-03-14 00:00:00
    4/5/13
    2013-04-05 00:00:00
    10/31/13
    2013-10-31 00:00:00
    9/13/13
    2013-09-13 00:00:00
    9/12/13
    2013-09-12 00:00:00
    11/8/13
    2013-11-08 00:00:00
    3/20/13
    2013-03-20 00:00:00
    7/11/13
    2013-07-11 00:00:00
    5/23/13
    2013-05-23 00:00:00
    3/20/13
    2013-03-20 00:00:00
    11/27/13
    2013-11-27 00:00:00
    10/9/13
    2013-10-09 00:00:00
    10/16/13
    2013-10-16 00:00:00
    10/10/13
    2013-10-10 00:00:00
    10/25/13
    2013-10-25 00:00:00
    6/27/13
    2013-06-27 00:00:00
    9/7/13
    2013-09-07 00:00:00
    4/25/13
    2013-04-25 00:00:00
    9/14/13
    2013-09-14 00:00:00
    8/23/13
    2013-08-23 00:00:00
    2/27/13
    2013-02-27 00:00:00
    9/24/13
    2013-09-24 00:00:00
    1/31/13
    2013-01-31 00:00:00
    9/7/13
    2013-09-07 00:00:00
    10/10/13
    2013-10-10 00:00:00
    6/12/13
    2013-06-12 00:00:00
    10/22/13
    2013-10-22 00:00:00
    1/23/13
    2013-01-23 00:00:00
    5/30/13
    2013-05-30 00:00:00
    12/12/13
    2013-12-12 00:00:00
    6/27/13
    2013-06-27 00:00:00
    11/16/13
    2013-11-16 00:00:00
    7/17/13
    2013-07-17 00:00:00
    9/8/13
    2013-09-08 00:00:00
    7/3/13
    2013-07-03 00:00:00
    1/10/13
    2013-01-10 00:00:00
    2/7/13
    2013-02-07 00:00:00
    1/12/13
    2013-01-12 00:00:00
    2/28/13
    2013-02-28 00:00:00
    10/4/13
    2013-10-04 00:00:00
    10/18/13
    2013-10-18 00:00:00
    1/3/13
    2013-01-03 00:00:00
    8/2/13
    2013-08-02 00:00:00
    3/27/13
    2013-03-27 00:00:00
    12/18/13
    2013-12-18 00:00:00
    11/5/13
    2013-11-05 00:00:00
    3/7/13
    2013-03-07 00:00:00
    7/11/13
    2013-07-11 00:00:00
    5/7/13
    2013-05-07 00:00:00
    4/18/13
    2013-04-18 00:00:00
    10/4/13
    2013-10-04 00:00:00
    8/28/13
    2013-08-28 00:00:00
    3/22/13
    2013-03-22 00:00:00
    11/14/13
    2013-11-14 00:00:00
    12/27/13
    2013-12-27 00:00:00
    1/17/13
    2013-01-17 00:00:00
    4/12/13
    2013-04-12 00:00:00
    12/25/13
    2013-12-25 00:00:00
    3/21/13
    2013-03-21 00:00:00
    10/22/13
    2013-10-22 00:00:00
    8/16/13
    2013-08-16 00:00:00
    1/1/13
    2013-01-01 00:00:00
    3/1/13
    2013-03-01 00:00:00
    2/14/13
    2013-02-14 00:00:00
    12/25/13
    2013-12-25 00:00:00
    9/8/13
    2013-09-08 00:00:00
    10/17/13
    2013-10-17 00:00:00
    9/12/13
    2013-09-12 00:00:00
    2/7/13
    2013-02-07 00:00:00
    6/27/13
    2013-06-27 00:00:00
    2/25/13
    2013-02-25 00:00:00
    9/6/13
    2013-09-06 00:00:00
    6/12/13
    2013-06-12 00:00:00
    9/3/13
    2013-09-03 00:00:00
    5/26/13
    2013-05-26 00:00:00
    9/26/13
    2013-09-26 00:00:00
    12/4/13
    2013-12-04 00:00:00
    7/25/13
    2013-07-25 00:00:00
    9/9/13
    2013-09-09 00:00:00
    8/11/13
    2013-08-11 00:00:00
    3/7/13
    2013-03-07 00:00:00
    2/8/13
    2013-02-08 00:00:00
    1/11/13
    2013-01-11 00:00:00
    5/30/13
    2013-05-30 00:00:00
    2/7/13
    2013-02-07 00:00:00
    9/20/13
    2013-09-20 00:00:00
    9/7/13
    2013-09-07 00:00:00
    1/1/13
    2013-01-01 00:00:00
    3/14/13
    2013-03-14 00:00:00
    5/1/13
    2013-05-01 00:00:00
    10/13/13
    2013-10-13 00:00:00
    11/28/13
    2013-11-28 00:00:00
    4/3/13
    2013-04-03 00:00:00
    12/26/13
    2013-12-26 00:00:00
    1/18/13
    2013-01-18 00:00:00
    8/23/13
    2013-08-23 00:00:00
    4/25/13
    2013-04-25 00:00:00
    9/13/13
    2013-09-13 00:00:00
    12/18/13
    2013-12-18 00:00:00
    12/10/13
    2013-12-10 00:00:00
    4/16/13
    2013-04-16 00:00:00
    3/9/13
    2013-03-09 00:00:00
    3/15/13
    2013-03-15 00:00:00
    2/13/13
    2013-02-13 00:00:00
    7/18/13
    2013-07-18 00:00:00
    7/30/13
    2013-07-30 00:00:00
    7/3/13
    2013-07-03 00:00:00
    4/12/13
    2013-04-12 00:00:00
    8/8/13
    2013-08-08 00:00:00
    9/10/13
    2013-09-10 00:00:00
    6/6/13
    2013-06-06 00:00:00
    10/11/13
    2013-10-11 00:00:00
    12/5/13
    2013-12-05 00:00:00
    8/22/13
    2013-08-22 00:00:00
    7/23/13
    2013-07-23 00:00:00
    2/21/13
    2013-02-21 00:00:00
    10/30/13
    2013-10-30 00:00:00
    9/21/13
    2013-09-21 00:00:00
    9/6/13
    2013-09-06 00:00:00
    4/11/13
    2013-04-11 00:00:00
    9/7/13
    2013-09-07 00:00:00
    9/11/13
    2013-09-11 00:00:00
    9/11/13
    2013-09-11 00:00:00
    9/19/13
    2013-09-19 00:00:00
    7/4/13
    2013-07-04 00:00:00
    9/27/13
    2013-09-27 00:00:00
    11/15/13
    2013-11-15 00:00:00
    8/10/13
    2013-08-10 00:00:00
    4/18/13
    2013-04-18 00:00:00
    12/26/13
    2013-12-26 00:00:00
    5/17/13
    2013-05-17 00:00:00
    9/21/13
    2013-09-21 00:00:00
    10/31/13
    2013-10-31 00:00:00
    9/12/13
    2013-09-12 00:00:00
    6/6/13
    2013-06-06 00:00:00
    9/6/13
    2013-09-06 00:00:00
    9/12/13
    2013-09-12 00:00:00
    11/23/13
    2013-11-23 00:00:00
    9/23/13
    2013-09-23 00:00:00
    11/21/13
    2013-11-21 00:00:00
    2/9/13
    2013-02-09 00:00:00
    4/5/13
    2013-04-05 00:00:00
    5/31/13
    2013-05-31 00:00:00
    10/3/13
    2013-10-03 00:00:00
    12/31/13
    2013-12-31 00:00:00
    11/15/13
    2013-11-15 00:00:00
    10/25/13
    2013-10-25 00:00:00
    10/17/13
    2013-10-17 00:00:00
    6/30/13
    2013-06-30 00:00:00
    4/17/13
    2013-04-17 00:00:00
    8/16/13
    2013-08-16 00:00:00
    2/21/13
    2013-02-21 00:00:00
    8/9/13
    2013-08-09 00:00:00
    12/3/13
    2013-12-03 00:00:00
    8/14/13
    2013-08-14 00:00:00
    9/5/13
    2013-09-05 00:00:00
    1/1/13
    2013-01-01 00:00:00
    7/12/13
    2013-07-12 00:00:00
    2/8/13
    2013-02-08 00:00:00
    3/7/13
    2013-03-07 00:00:00
    9/17/13
    2013-09-17 00:00:00
    6/30/13
    2013-06-30 00:00:00
    1/20/13
    2013-01-20 00:00:00
    8/25/13
    2013-08-25 00:00:00
    11/29/13
    2013-11-29 00:00:00
    7/19/13
    2013-07-19 00:00:00
    7/25/13
    2013-07-25 00:00:00
    12/10/13
    2013-12-10 00:00:00
    5/9/13
    2013-05-09 00:00:00
    9/9/13
    2013-09-09 00:00:00
    12/6/13
    2013-12-06 00:00:00
    7/24/13
    2013-07-24 00:00:00
    12/24/13
    2013-12-24 00:00:00
    8/7/13
    2013-08-07 00:00:00
    8/22/13
    2013-08-22 00:00:00
    10/19/13
    2013-10-19 00:00:00
    5/6/13
    2013-05-06 00:00:00
    6/28/13
    2013-06-28 00:00:00
    9/10/13
    2013-09-10 00:00:00
    4/21/13
    2013-04-21 00:00:00
    8/30/13
    2013-08-30 00:00:00
    9/6/13
    2013-09-06 00:00:00
    9/21/13
    2013-09-21 00:00:00
    3/10/13
    2013-03-10 00:00:00
    8/9/13
    2013-08-09 00:00:00
    8/13/13
    2013-08-13 00:00:00
    1/22/13
    2013-01-22 00:00:00
    4/5/13
    2013-04-05 00:00:00
    3/8/13
    2013-03-08 00:00:00
    1/14/13
    2013-01-14 00:00:00
    9/28/13
    2013-09-28 00:00:00
    4/20/13
    2013-04-20 00:00:00
    10/23/13
    2013-10-23 00:00:00
    6/15/13
    2013-06-15 00:00:00
    9/4/13
    2013-09-04 00:00:00
    7/5/13
    2013-07-05 00:00:00
    12/10/13
    2013-12-10 00:00:00
    7/28/13
    2013-07-28 00:00:00
    4/21/13
    2013-04-21 00:00:00
    7/18/13
    2013-07-18 00:00:00
    9/11/13
    2013-09-11 00:00:00
    9/20/13
    2013-09-20 00:00:00
    2/9/13
    2013-02-09 00:00:00
    11/8/13
    2013-11-08 00:00:00
    6/7/13
    2013-06-07 00:00:00
    11/27/13
    2013-11-27 00:00:00
    11/10/13
    2013-11-10 00:00:00
    4/4/13
    2013-04-04 00:00:00
    12/27/13
    2013-12-27 00:00:00
    7/15/13
    2013-07-15 00:00:00
    10/3/13
    2013-10-03 00:00:00
    9/8/13
    2013-09-08 00:00:00
    9/6/13
    2013-09-06 00:00:00
    10/11/13
    2013-10-11 00:00:00
    11/1/13
    2013-11-01 00:00:00
    7/11/13
    2013-07-11 00:00:00
    9/27/13
    2013-09-27 00:00:00
    3/1/13
    2013-03-01 00:00:00
    12/1/13
    2013-12-01 00:00:00
    12/25/13
    2013-12-25 00:00:00
    6/14/13
    2013-06-14 00:00:00
    10/30/13
    2013-10-30 00:00:00
    10/1/13
    2013-10-01 00:00:00
    9/2/13
    2013-09-02 00:00:00
    3/1/13
    2013-03-01 00:00:00
    10/4/13
    2013-10-04 00:00:00
    8/9/13
    2013-08-09 00:00:00
    3/8/13
    2013-03-08 00:00:00
    7/12/13
    2013-07-12 00:00:00
    7/3/13
    2013-07-03 00:00:00
    1/24/13
    2013-01-24 00:00:00
    2/14/13
    2013-02-14 00:00:00
    10/5/13
    2013-10-05 00:00:00
    8/30/13
    2013-08-30 00:00:00
    5/20/13
    2013-05-20 00:00:00
    4/12/13
    2013-04-12 00:00:00
    8/4/13
    2013-08-04 00:00:00
    6/1/13
    2013-06-01 00:00:00
    5/16/13
    2013-05-16 00:00:00
    1/3/13
    2013-01-03 00:00:00
    9/7/13
    2013-09-07 00:00:00
    4/3/13
    2013-04-03 00:00:00
    2/15/13
    2013-02-15 00:00:00
    9/7/13
    2013-09-07 00:00:00
    12/8/13
    2013-12-08 00:00:00
    11/16/13
    2013-11-16 00:00:00
    1/30/13
    2013-01-30 00:00:00
    7/27/13
    2013-07-27 00:00:00
    8/30/13
    2013-08-30 00:00:00
    11/5/13
    2013-11-05 00:00:00
    9/30/13
    2013-09-30 00:00:00
    12/10/13
    2013-12-10 00:00:00
    1/4/13
    2013-01-04 00:00:00
    9/25/13
    2013-09-25 00:00:00
    1/16/13
    2013-01-16 00:00:00
    11/15/13
    2013-11-15 00:00:00
    1/10/13
    2013-01-10 00:00:00
    5/6/13
    2013-05-06 00:00:00
    6/13/13
    2013-06-13 00:00:00
    11/8/13
    2013-11-08 00:00:00
    1/4/13
    2013-01-04 00:00:00
    9/18/13
    2013-09-18 00:00:00
    12/13/13
    2013-12-13 00:00:00
    8/28/13
    2013-08-28 00:00:00
    10/18/13
    2013-10-18 00:00:00
    7/25/13
    2013-07-25 00:00:00
    4/21/13
    2013-04-21 00:00:00
    8/15/13
    2013-08-15 00:00:00
    7/3/13
    2013-07-03 00:00:00
    1/21/13
    2013-01-21 00:00:00
    3/29/13
    2013-03-29 00:00:00
    9/3/13
    2013-09-03 00:00:00
    5/31/13
    2013-05-31 00:00:00
    10/11/13
    2013-10-11 00:00:00
    9/20/13
    2013-09-20 00:00:00
    6/15/13
    2013-06-15 00:00:00
    11/15/13
    2013-11-15 00:00:00
    12/6/13
    2013-12-06 00:00:00
    3/8/13
    2013-03-08 00:00:00
    2/1/13
    2013-02-01 00:00:00
    8/26/13
    2013-08-26 00:00:00
    7/6/13
    2013-07-06 00:00:00
    1/18/13
    2013-01-18 00:00:00
    8/26/13
    2013-08-26 00:00:00
    2/16/13
    2013-02-16 00:00:00
    9/6/13
    2013-09-06 00:00:00
    10/19/13
    2013-10-19 00:00:00
    2/19/13
    2013-02-19 00:00:00
    5/1/13
    2013-05-01 00:00:00
    4/21/13
    2013-04-21 00:00:00
    2/27/13
    2013-02-27 00:00:00
    9/13/13
    2013-09-13 00:00:00
    4/26/13
    2013-04-26 00:00:00
    8/8/13
    2013-08-08 00:00:00
    5/22/13
    2013-05-22 00:00:00
    6/28/13
    2013-06-28 00:00:00
    9/8/13
    2013-09-08 00:00:00
    4/1/13
    2013-04-01 00:00:00
    10/16/13
    2013-10-16 00:00:00
    7/25/13
    2013-07-25 00:00:00
    3/28/13
    2013-03-28 00:00:00
    7/29/13
    2013-07-29 00:00:00
    4/2/13
    2013-04-02 00:00:00
    6/27/13
    2013-06-27 00:00:00
    3/17/13
    2013-03-17 00:00:00
    4/26/13
    2013-04-26 00:00:00
    8/30/13
    2013-08-30 00:00:00
    1/22/13
    2013-01-22 00:00:00
    10/15/13
    2013-10-15 00:00:00
    4/23/13
    2013-04-23 00:00:00
    3/22/13
    2013-03-22 00:00:00
    8/7/13
    2013-08-07 00:00:00
    7/25/13
    2013-07-25 00:00:00
    10/25/13
    2013-10-25 00:00:00
    9/6/13
    2013-09-06 00:00:00
    1/20/13
    2013-01-20 00:00:00
    8/23/13
    2013-08-23 00:00:00
    4/5/13
    2013-04-05 00:00:00
    9/27/13
    2013-09-27 00:00:00
    7/26/13
    2013-07-26 00:00:00
    4/10/13
    2013-04-10 00:00:00
    4/18/13
    2013-04-18 00:00:00
    6/13/13
    2013-06-13 00:00:00
    1/19/13
    2013-01-19 00:00:00
    2/14/13
    2013-02-14 00:00:00
    3/1/13
    2013-03-01 00:00:00
    10/8/13
    2013-10-08 00:00:00
    7/26/13
    2013-07-26 00:00:00
    2/12/13
    2013-02-12 00:00:00
    7/7/13
    2013-07-07 00:00:00
    9/19/13
    2013-09-19 00:00:00
    9/12/13
    2013-09-12 00:00:00
    6/7/13
    2013-06-07 00:00:00
    6/24/13
    2013-06-24 00:00:00
    1/18/13
    2013-01-18 00:00:00
    4/20/13
    2013-04-20 00:00:00
    12/4/13
    2013-12-04 00:00:00
    12/31/13
    2013-12-31 00:00:00
    10/2/13
    2013-10-02 00:00:00
    8/22/13
    2013-08-22 00:00:00
    8/24/13
    2013-08-24 00:00:00
    4/22/13
    2013-04-22 00:00:00
    5/10/13
    2013-05-10 00:00:00
    3/13/13
    2013-03-13 00:00:00
    3/29/13
    2013-03-29 00:00:00
    6/14/13
    2013-06-14 00:00:00
    6/7/13
    2013-06-07 00:00:00
    8/14/13
    2013-08-14 00:00:00
    10/11/13
    2013-10-11 00:00:00
    7/30/13
    2013-07-30 00:00:00
    7/31/13
    2013-07-31 00:00:00
    5/1/13
    2013-05-01 00:00:00
    3/15/13
    2013-03-15 00:00:00
    8/30/13
    2013-08-30 00:00:00
    10/11/13
    2013-10-11 00:00:00
    9/10/13
    2013-09-10 00:00:00
    12/25/13
    2013-12-25 00:00:00
    4/27/13
    2013-04-27 00:00:00
    6/20/13
    2013-06-20 00:00:00
    2/28/13
    2013-02-28 00:00:00
    3/22/13
    2013-03-22 00:00:00
    1/18/13
    2013-01-18 00:00:00
    10/16/13
    2013-10-16 00:00:00
    5/2/13
    2013-05-02 00:00:00
    3/13/13
    2013-03-13 00:00:00
    7/19/13
    2013-07-19 00:00:00
    12/6/13
    2013-12-06 00:00:00
    9/12/13
    2013-09-12 00:00:00
    4/13/13
    2013-04-13 00:00:00
    3/5/13
    2013-03-05 00:00:00
    10/8/13
    2013-10-08 00:00:00
    10/25/13
    2013-10-25 00:00:00
    11/1/13
    2013-11-01 00:00:00
    1/29/13
    2013-01-29 00:00:00
    11/22/13
    2013-11-22 00:00:00
    3/28/13
    2013-03-28 00:00:00
    3/15/13
    2013-03-15 00:00:00
    5/31/13
    2013-05-31 00:00:00
    8/30/13
    2013-08-30 00:00:00
    4/9/13
    2013-04-09 00:00:00
    7/26/13
    2013-07-26 00:00:00
    8/9/13
    2013-08-09 00:00:00
    10/11/13
    2013-10-11 00:00:00
    11/14/13
    2013-11-14 00:00:00
    6/10/13
    2013-06-10 00:00:00
    8/29/13
    2013-08-29 00:00:00
    9/19/13
    2013-09-19 00:00:00
    6/29/13
    2013-06-29 00:00:00
    1/19/13
    2013-01-19 00:00:00
    3/5/13
    2013-03-05 00:00:00
    8/16/13
    2013-08-16 00:00:00
    9/6/13
    2013-09-06 00:00:00
    11/29/13
    2013-11-29 00:00:00
    6/21/13
    2013-06-21 00:00:00
    10/9/13
    2013-10-09 00:00:00
    9/27/13
    2013-09-27 00:00:00
    11/18/13
    2013-11-18 00:00:00
    1/17/13
    2013-01-17 00:00:00
    7/24/13
    2013-07-24 00:00:00
    10/12/13
    2013-10-12 00:00:00
    12/12/13
    2013-12-12 00:00:00
    10/17/13
    2013-10-17 00:00:00
    3/18/13
    2013-03-18 00:00:00
    10/18/13
    2013-10-18 00:00:00
    3/12/13
    2013-03-12 00:00:00
    10/11/13
    2013-10-11 00:00:00
    6/14/13
    2013-06-14 00:00:00
    9/4/13
    2013-09-04 00:00:00
    2/12/13
    2013-02-12 00:00:00
    7/5/13
    2013-07-05 00:00:00
    10/10/13
    2013-10-10 00:00:00
    11/16/13
    2013-11-16 00:00:00
    3/24/13
    2013-03-24 00:00:00
    12/19/13
    2013-12-19 00:00:00
    1/24/13
    2013-01-24 00:00:00
    5/16/13
    2013-05-16 00:00:00
    5/16/13
    2013-05-16 00:00:00
    9/19/13
    2013-09-19 00:00:00
    10/29/13
    2013-10-29 00:00:00
    5/28/13
    2013-05-28 00:00:00
    1/31/13
    2013-01-31 00:00:00
    1/7/13
    2013-01-07 00:00:00
    1/18/13
    2013-01-18 00:00:00
    11/26/13
    2013-11-26 00:00:00
    12/11/13
    2013-12-11 00:00:00
    11/29/13
    2013-11-29 00:00:00
    12/27/13
    2013-12-27 00:00:00
    12/26/13
    2013-12-26 00:00:00
    7/12/13
    2013-07-12 00:00:00
    8/29/13
    2013-08-29 00:00:00
    9/27/13
    2013-09-27 00:00:00
    10/8/13
    2013-10-08 00:00:00
    11/28/13
    2013-11-28 00:00:00
    12/6/13
    2013-12-06 00:00:00
    5/23/13
    2013-05-23 00:00:00
    8/31/13
    2013-08-31 00:00:00
    9/6/13
    2013-09-06 00:00:00
    12/1/13
    2013-12-01 00:00:00
    12/29/13
    2013-12-29 00:00:00
    12/18/13
    2013-12-18 00:00:00
    8/28/13
    2013-08-28 00:00:00
    8/16/13
    2013-08-16 00:00:00
    6/21/13
    2013-06-21 00:00:00
    8/31/13
    2013-08-31 00:00:00
    8/28/13
    2013-08-28 00:00:00
    10/11/13
    2013-10-11 00:00:00
    12/18/13
    2013-12-18 00:00:00
    5/18/13
    2013-05-18 00:00:00
    7/12/13
    2013-07-12 00:00:00
    12/5/13
    2013-12-05 00:00:00
    11/15/13
    2013-11-15 00:00:00
    10/8/13
    2013-10-08 00:00:00
    5/15/13
    2013-05-15 00:00:00
    7/12/13
    2013-07-12 00:00:00
    4/12/13
    2013-04-12 00:00:00
    5/11/13
    2013-05-11 00:00:00
    2/28/13
    2013-02-28 00:00:00
    5/28/13
    2013-05-28 00:00:00
    9/26/13
    2013-09-26 00:00:00
    1/22/13
    2013-01-22 00:00:00
    3/25/13
    2013-03-25 00:00:00
    7/22/13
    2013-07-22 00:00:00
    8/23/13
    2013-08-23 00:00:00
    6/10/13
    2013-06-10 00:00:00
    5/10/13
    2013-05-10 00:00:00
    2/8/13
    2013-02-08 00:00:00
    6/6/13
    2013-06-06 00:00:00
    3/5/13
    2013-03-05 00:00:00
    6/22/13
    2013-06-22 00:00:00
    5/1/13
    2013-05-01 00:00:00
    11/6/13
    2013-11-06 00:00:00
    3/20/13
    2013-03-20 00:00:00
    10/26/13
    2013-10-26 00:00:00
    9/2/13
    2013-09-02 00:00:00
    3/17/13
    2013-03-17 00:00:00
    12/16/13
    2013-12-16 00:00:00
    10/18/13
    2013-10-18 00:00:00
    9/25/13
    2013-09-25 00:00:00
    11/5/13
    2013-11-05 00:00:00
    7/15/13
    2013-07-15 00:00:00
    6/17/13
    2013-06-17 00:00:00
    11/27/13
    2013-11-27 00:00:00
    5/9/13
    2013-05-09 00:00:00
    5/24/13
    2013-05-24 00:00:00
    5/20/13
    2013-05-20 00:00:00
    9/27/13
    2013-09-27 00:00:00
    9/2/13
    2013-09-02 00:00:00
    3/1/13
    2013-03-01 00:00:00
    1/24/13
    2013-01-24 00:00:00
    6/21/13
    2013-06-21 00:00:00
    10/3/13
    2013-10-03 00:00:00
    9/6/13
    2013-09-06 00:00:00
    10/30/13
    2013-10-30 00:00:00
    2/19/13
    2013-02-19 00:00:00
    1/26/13
    2013-01-26 00:00:00
    4/14/13
    2013-04-14 00:00:00
    5/25/13
    2013-05-25 00:00:00
    10/18/13
    2013-10-18 00:00:00
    11/17/13
    2013-11-17 00:00:00
    7/23/13
    2013-07-23 00:00:00
    1/14/13
    2013-01-14 00:00:00
    5/17/13
    2013-05-17 00:00:00
    10/4/13
    2013-10-04 00:00:00
    12/18/13
    2013-12-18 00:00:00
    5/28/13
    2013-05-28 00:00:00
    4/17/13
    2013-04-17 00:00:00
    11/29/13
    2013-11-29 00:00:00
    1/6/13
    2013-01-06 00:00:00
    5/13/13
    2013-05-13 00:00:00
    9/20/13
    2013-09-20 00:00:00
    10/19/13
    2013-10-19 00:00:00
    12/19/13
    2013-12-19 00:00:00
    10/18/13
    2013-10-18 00:00:00
    11/9/13
    2013-11-09 00:00:00
    7/28/13
    2013-07-28 00:00:00
    8/20/13
    2013-08-20 00:00:00
    7/24/13
    2013-07-24 00:00:00
    9/6/13
    2013-09-06 00:00:00
    3/7/13
    2013-03-07 00:00:00
    12/1/13
    2013-12-01 00:00:00
    3/7/13
    2013-03-07 00:00:00
    12/6/13
    2013-12-06 00:00:00
    6/5/13
    2013-06-05 00:00:00
    4/23/13
    2013-04-23 00:00:00
    11/22/13
    2013-11-22 00:00:00
    9/5/13
    2013-09-05 00:00:00
    1/10/13
    2013-01-10 00:00:00
    2/15/13
    2013-02-15 00:00:00
    7/9/13
    2013-07-09 00:00:00
    4/24/13
    2013-04-24 00:00:00
    8/11/13
    2013-08-11 00:00:00
    12/18/13
    2013-12-18 00:00:00
    8/16/13
    2013-08-16 00:00:00
    12/13/13
    2013-12-13 00:00:00
    9/13/13
    2013-09-13 00:00:00
    1/18/13
    2013-01-18 00:00:00
    4/19/13
    2013-04-19 00:00:00
    9/8/13
    2013-09-08 00:00:00
    10/11/13
    2013-10-11 00:00:00
    10/31/13
    2013-10-31 00:00:00
    4/23/13
    2013-04-23 00:00:00
    9/13/13
    2013-09-13 00:00:00
    1/19/13
    2013-01-19 00:00:00
    5/2/13
    2013-05-02 00:00:00
    12/13/13
    2013-12-13 00:00:00
    1/10/13
    2013-01-10 00:00:00
    12/2/13
    2013-12-02 00:00:00
    4/26/13
    2013-04-26 00:00:00
    5/13/13
    2013-05-13 00:00:00
    3/14/13
    2013-03-14 00:00:00
    11/22/13
    2013-11-22 00:00:00
    10/4/13
    2013-10-04 00:00:00
    6/24/13
    2013-06-24 00:00:00
    9/12/13
    2013-09-12 00:00:00
    10/18/13
    2013-10-18 00:00:00
    10/15/13
    2013-10-15 00:00:00
    11/22/13
    2013-11-22 00:00:00
    7/27/13
    2013-07-27 00:00:00
    1/21/13
    2013-01-21 00:00:00
    11/25/13
    2013-11-25 00:00:00
    6/19/13
    2013-06-19 00:00:00
    3/8/13
    2013-03-08 00:00:00
    6/10/13
    2013-06-10 00:00:00
    7/17/13
    2013-07-17 00:00:00
    5/14/13
    2013-05-14 00:00:00
    5/2/13
    2013-05-02 00:00:00
    5/29/13
    2013-05-29 00:00:00
    6/7/13
    2013-06-07 00:00:00
    6/22/13
    2013-06-22 00:00:00
    5/3/13
    2013-05-03 00:00:00
    10/10/13
    2013-10-10 00:00:00
    4/5/13
    2013-04-05 00:00:00
    12/6/13
    2013-12-06 00:00:00
    2/14/13
    2013-02-14 00:00:00
    10/11/13
    2013-10-11 00:00:00
    10/16/13
    2013-10-16 00:00:00
    11/5/13
    2013-11-05 00:00:00
    11/1/13
    2013-11-01 00:00:00
    7/19/13
    2013-07-19 00:00:00
    10/4/13
    2013-10-04 00:00:00
    9/19/13
    2013-09-19 00:00:00
    6/21/13
    2013-06-21 00:00:00
    3/22/13
    2013-03-22 00:00:00
    12/11/13
    2013-12-11 00:00:00
    10/11/13
    2013-10-11 00:00:00
    5/28/13
    2013-05-28 00:00:00
    3/9/13
    2013-03-09 00:00:00
    11/7/13
    2013-11-07 00:00:00
    11/6/13
    2013-11-06 00:00:00
    8/10/13
    2013-08-10 00:00:00
    4/16/13
    2013-04-16 00:00:00
    10/9/13
    2013-10-09 00:00:00
    6/10/13
    2013-06-10 00:00:00
    3/5/13
    2013-03-05 00:00:00
    11/10/13
    2013-11-10 00:00:00
    1/1/13
    2013-01-01 00:00:00
    11/7/13
    2013-11-07 00:00:00
    10/8/13
    2013-10-08 00:00:00
    6/21/13
    2013-06-21 00:00:00
    1/19/13
    2013-01-19 00:00:00
    10/15/13
    2013-10-15 00:00:00
    10/15/13
    2013-10-15 00:00:00
    9/19/13
    2013-09-19 00:00:00
    7/9/13
    2013-07-09 00:00:00
    12/4/13
    2013-12-04 00:00:00
    12/5/13
    2013-12-05 00:00:00
    10/20/13
    2013-10-20 00:00:00
    7/3/85
    1985-07-03 00:00:00
    2/15/85
    1985-02-15 00:00:00
    4/3/85
    1985-04-03 00:00:00
    6/29/85
    1985-06-29 00:00:00
    5/24/85
    1985-05-24 00:00:00
    7/19/85
    1985-07-19 00:00:00
    6/6/85
    1985-06-06 00:00:00
    11/1/85
    1985-11-01 00:00:00
    5/21/85
    1985-05-21 00:00:00
    12/10/85
    1985-12-10 00:00:00
    11/21/85
    1985-11-21 00:00:00
    2/20/85
    1985-02-20 00:00:00
    7/24/85
    1985-07-24 00:00:00
    12/18/85
    1985-12-18 00:00:00
    10/3/85
    1985-10-03 00:00:00
    3/20/85
    1985-03-20 00:00:00
    5/15/85
    1985-05-15 00:00:00
    2/8/85
    1985-02-08 00:00:00
    6/26/85
    1985-06-26 00:00:00
    6/21/85
    1985-06-21 00:00:00
    6/14/85
    1985-06-14 00:00:00
    8/2/85
    1985-08-02 00:00:00
    7/25/85
    1985-07-25 00:00:00
    3/28/85
    1985-03-28 00:00:00
    4/9/85
    1985-04-09 00:00:00
    7/19/85
    1985-07-19 00:00:00
    12/13/85
    1985-12-13 00:00:00
    3/22/85
    1985-03-22 00:00:00
    10/29/85
    1985-10-29 00:00:00
    5/3/85
    1985-05-03 00:00:00
    9/8/85
    1985-09-08 00:00:00
    8/30/85
    1985-08-30 00:00:00
    5/31/85
    1985-05-31 00:00:00
    2/8/85
    1985-02-08 00:00:00
    6/28/85
    1985-06-28 00:00:00
    3/1/85
    1985-03-01 00:00:00
    3/8/85
    1985-03-08 00:00:00
    1/31/85
    1985-01-31 00:00:00
    12/4/85
    1985-12-04 00:00:00
    6/20/85
    1985-06-20 00:00:00
    10/18/85
    1985-10-18 00:00:00
    3/22/85
    1985-03-22 00:00:00
    7/18/85
    1985-07-18 00:00:00
    8/7/85
    1985-08-07 00:00:00
    3/1/85
    1985-03-01 00:00:00
    12/6/85
    1985-12-06 00:00:00
    11/1/85
    1985-11-01 00:00:00
    5/3/85
    1985-05-03 00:00:00
    9/13/85
    1985-09-13 00:00:00
    7/26/85
    1985-07-26 00:00:00
    6/14/85
    1985-06-14 00:00:00
    10/11/85
    1985-10-11 00:00:00
    8/16/85
    1985-08-16 00:00:00
    11/24/85
    1985-11-24 00:00:00
    12/10/85
    1985-12-10 00:00:00
    4/12/85
    1985-04-12 00:00:00
    5/22/85
    1985-05-22 00:00:00
    9/27/85
    1985-09-27 00:00:00
    12/12/85
    1985-12-12 00:00:00
    7/10/85
    1985-07-10 00:00:00
    12/9/85
    1985-12-09 00:00:00
    7/11/85
    1985-07-11 00:00:00
    8/23/85
    1985-08-23 00:00:00
    7/26/85
    1985-07-26 00:00:00
    10/11/85
    1985-10-11 00:00:00
    6/28/85
    1985-06-28 00:00:00
    1/25/85
    1985-01-25 00:00:00
    9/19/85
    1985-09-19 00:00:00
    5/3/85
    1985-05-03 00:00:00
    7/19/85
    1985-07-19 00:00:00
    12/4/85
    1985-12-04 00:00:00
    3/29/85
    1985-03-29 00:00:00
    8/9/85
    1985-08-09 00:00:00
    8/30/85
    1985-08-30 00:00:00
    2/8/85
    1985-02-08 00:00:00
    11/15/85
    1985-11-15 00:00:00
    9/25/85
    1985-09-25 00:00:00
    8/16/85
    1985-08-16 00:00:00
    1/25/85
    1985-01-25 00:00:00
    11/22/85
    1985-11-22 00:00:00
    6/21/85
    1985-06-21 00:00:00
    11/8/85
    1985-11-08 00:00:00
    8/9/85
    1985-08-09 00:00:00
    10/4/85
    1985-10-04 00:00:00
    5/15/85
    1985-05-15 00:00:00
    8/28/85
    1985-08-28 00:00:00
    10/11/85
    1985-10-11 00:00:00
    11/15/85
    1985-11-15 00:00:00
    8/16/85
    1985-08-16 00:00:00
    12/1/85
    1985-12-01 00:00:00
    11/1/85
    1985-11-01 00:00:00
    3/22/85
    1985-03-22 00:00:00
    6/14/85
    1985-06-14 00:00:00
    4/12/85
    1985-04-12 00:00:00
    7/12/85
    1985-07-12 00:00:00
    11/1/85
    1985-11-01 00:00:00
    8/16/85
    1985-08-16 00:00:00
    6/7/85
    1985-06-07 00:00:00
    2/15/85
    1985-02-15 00:00:00
    4/26/85
    1985-04-26 00:00:00
    11/3/85
    1985-11-03 00:00:00
    6/14/85
    1985-06-14 00:00:00
    9/7/85
    1985-09-07 00:00:00
    11/27/85
    1985-11-27 00:00:00
    3/1/85
    1985-03-01 00:00:00
    5/25/85
    1985-05-25 00:00:00
    3/23/85
    1985-03-23 00:00:00
    10/4/85
    1985-10-04 00:00:00
    2/15/85
    1985-02-15 00:00:00
    11/5/05
    2005-11-05 00:00:00
    6/14/05
    2005-06-14 00:00:00
    12/7/05
    2005-12-07 00:00:00
    12/11/05
    2005-12-11 00:00:00
    3/31/05
    2005-03-31 00:00:00
    7/9/05
    2005-07-09 00:00:00
    5/17/05
    2005-05-17 00:00:00
    1/15/05
    2005-01-15 00:00:00
    7/20/05
    2005-07-20 00:00:00
    5/25/05
    2005-05-25 00:00:00
    7/7/05
    2005-07-07 00:00:00
    2/15/05
    2005-02-15 00:00:00
    7/14/05
    2005-07-14 00:00:00
    11/30/05
    2005-11-30 00:00:00
    3/4/05
    2005-03-04 00:00:00
    6/28/05
    2005-06-28 00:00:00
    12/6/05
    2005-12-06 00:00:00
    8/11/05
    2005-08-11 00:00:00
    11/4/05
    2005-11-04 00:00:00
    2/10/05
    2005-02-10 00:00:00
    8/2/05
    2005-08-02 00:00:00
    9/21/05
    2005-09-21 00:00:00
    10/21/05
    2005-10-21 00:00:00
    5/3/05
    2005-05-03 00:00:00
    5/13/05
    2005-05-13 00:00:00
    12/14/05
    2005-12-14 00:00:00
    3/4/05
    2005-03-04 00:00:00
    9/30/05
    2005-09-30 00:00:00
    9/16/05
    2005-09-16 00:00:00
    9/2/05
    2005-09-02 00:00:00
    3/11/05
    2005-03-11 00:00:00
    12/5/05
    2005-12-05 00:00:00
    4/27/05
    2005-04-27 00:00:00
    6/1/05
    2005-06-01 00:00:00
    6/7/05
    2005-06-07 00:00:00
    9/9/05
    2005-09-09 00:00:00
    10/24/05
    2005-10-24 00:00:00
    9/16/05
    2005-09-16 00:00:00
    11/6/05
    2005-11-06 00:00:00
    1/13/05
    2005-01-13 00:00:00
    3/17/05
    2005-03-17 00:00:00
    9/9/05
    2005-09-09 00:00:00
    9/4/05
    2005-09-04 00:00:00
    12/23/05
    2005-12-23 00:00:00
    9/10/05
    2005-09-10 00:00:00
    3/4/05
    2005-03-04 00:00:00
    11/4/05
    2005-11-04 00:00:00
    7/8/05
    2005-07-08 00:00:00
    8/25/05
    2005-08-25 00:00:00
    10/28/05
    2005-10-28 00:00:00
    9/16/05
    2005-09-16 00:00:00
    6/18/05
    2005-06-18 00:00:00
    9/3/05
    2005-09-03 00:00:00
    1/29/05
    2005-01-29 00:00:00
    8/11/05
    2005-08-11 00:00:00
    1/6/05
    2005-01-06 00:00:00
    2/12/05
    2005-02-12 00:00:00
    6/2/05
    2005-06-02 00:00:00
    5/12/05
    2005-05-12 00:00:00
    12/16/05
    2005-12-16 00:00:00
    1/27/05
    2005-01-27 00:00:00
    6/21/05
    2005-06-21 00:00:00
    10/7/05
    2005-10-07 00:00:00
    10/30/05
    2005-10-30 00:00:00
    10/20/05
    2005-10-20 00:00:00
    11/23/05
    2005-11-23 00:00:00
    12/21/05
    2005-12-21 00:00:00
    4/6/05
    2005-04-06 00:00:00
    7/21/05
    2005-07-21 00:00:00
    9/5/05
    2005-09-05 00:00:00
    4/4/05
    2005-04-04 00:00:00
    8/25/05
    2005-08-25 00:00:00
    9/9/05
    2005-09-09 00:00:00
    3/9/05
    2005-03-09 00:00:00
    8/5/05
    2005-08-05 00:00:00
    12/22/05
    2005-12-22 00:00:00
    9/9/05
    2005-09-09 00:00:00
    9/5/05
    2005-09-05 00:00:00
    8/6/05
    2005-08-06 00:00:00
    11/23/05
    2005-11-23 00:00:00
    2/18/05
    2005-02-18 00:00:00
    11/11/05
    2005-11-11 00:00:00
    9/23/05
    2005-09-23 00:00:00
    7/29/05
    2005-07-29 00:00:00
    1/17/05
    2005-01-17 00:00:00
    1/6/05
    2005-01-06 00:00:00
    9/16/05
    2005-09-16 00:00:00
    1/14/05
    2005-01-14 00:00:00
    9/23/05
    2005-09-23 00:00:00
    4/15/05
    2005-04-15 00:00:00
    12/21/05
    2005-12-21 00:00:00
    2/2/05
    2005-02-02 00:00:00
    7/1/05
    2005-07-01 00:00:00
    3/25/05
    2005-03-25 00:00:00
    12/2/05
    2005-12-02 00:00:00
    10/7/05
    2005-10-07 00:00:00
    9/11/05
    2005-09-11 00:00:00
    9/22/05
    2005-09-22 00:00:00
    1/19/05
    2005-01-19 00:00:00
    6/17/05
    2005-06-17 00:00:00
    5/17/05
    2005-05-17 00:00:00
    10/20/05
    2005-10-20 00:00:00
    9/11/05
    2005-09-11 00:00:00
    1/1/05
    2005-01-01 00:00:00
    6/24/05
    2005-06-24 00:00:00
    3/11/05
    2005-03-11 00:00:00
    10/14/05
    2005-10-14 00:00:00
    9/13/05
    2005-09-13 00:00:00
    12/14/05
    2005-12-14 00:00:00
    4/22/05
    2005-04-22 00:00:00
    11/9/05
    2005-11-09 00:00:00
    5/13/05
    2005-05-13 00:00:00
    1/25/05
    2005-01-25 00:00:00
    2/24/05
    2005-02-24 00:00:00
    9/24/05
    2005-09-24 00:00:00
    3/17/05
    2005-03-17 00:00:00
    5/19/05
    2005-05-19 00:00:00
    10/7/05
    2005-10-07 00:00:00
    1/20/05
    2005-01-20 00:00:00
    5/15/05
    2005-05-15 00:00:00
    2/10/05
    2005-02-10 00:00:00
    1/1/05
    2005-01-01 00:00:00
    10/14/05
    2005-10-14 00:00:00
    7/28/05
    2005-07-28 00:00:00
    2/4/05
    2005-02-04 00:00:00
    10/22/05
    2005-10-22 00:00:00
    10/13/05
    2005-10-13 00:00:00
    9/3/05
    2005-09-03 00:00:00
    12/25/05
    2005-12-25 00:00:00
    1/11/05
    2005-01-11 00:00:00
    4/6/05
    2005-04-06 00:00:00
    5/12/05
    2005-05-12 00:00:00
    9/30/05
    2005-09-30 00:00:00
    3/25/05
    2005-03-25 00:00:00
    8/31/05
    2005-08-31 00:00:00
    2/1/05
    2005-02-01 00:00:00
    1/14/05
    2005-01-14 00:00:00
    5/6/05
    2005-05-06 00:00:00
    8/12/05
    2005-08-12 00:00:00
    9/22/05
    2005-09-22 00:00:00
    10/18/05
    2005-10-18 00:00:00
    5/3/05
    2005-05-03 00:00:00
    6/2/05
    2005-06-02 00:00:00
    9/8/05
    2005-09-08 00:00:00
    10/7/05
    2005-10-07 00:00:00
    6/10/05
    2005-06-10 00:00:00
    6/13/05
    2005-06-13 00:00:00
    9/30/05
    2005-09-30 00:00:00
    9/30/05
    2005-09-30 00:00:00
    7/29/05
    2005-07-29 00:00:00
    7/19/05
    2005-07-19 00:00:00
    11/17/05
    2005-11-17 00:00:00
    6/28/05
    2005-06-28 00:00:00
    10/25/05
    2005-10-25 00:00:00
    9/16/05
    2005-09-16 00:00:00
    6/6/05
    2005-06-06 00:00:00
    1/7/05
    2005-01-07 00:00:00
    4/22/05
    2005-04-22 00:00:00
    1/30/05
    2005-01-30 00:00:00
    5/20/05
    2005-05-20 00:00:00
    10/11/05
    2005-10-11 00:00:00
    12/25/05
    2005-12-25 00:00:00
    5/16/05
    2005-05-16 00:00:00
    11/1/05
    2005-11-01 00:00:00
    7/29/05
    2005-07-29 00:00:00
    9/12/05
    2005-09-12 00:00:00
    8/4/05
    2005-08-04 00:00:00
    10/14/05
    2005-10-14 00:00:00
    12/23/05
    2005-12-23 00:00:00
    5/20/05
    2005-05-20 00:00:00
    5/5/05
    2005-05-05 00:00:00
    2/8/05
    2005-02-08 00:00:00
    9/9/05
    2005-09-09 00:00:00
    9/13/05
    2005-09-13 00:00:00
    9/14/05
    2005-09-14 00:00:00
    9/16/05
    2005-09-16 00:00:00
    9/23/05
    2005-09-23 00:00:00
    10/5/05
    2005-10-05 00:00:00
    12/31/05
    2005-12-31 00:00:00
    3/25/05
    2005-03-25 00:00:00
    11/23/05
    2005-11-23 00:00:00
    2/11/05
    2005-02-11 00:00:00
    9/1/05
    2005-09-01 00:00:00
    2/19/05
    2005-02-19 00:00:00
    6/21/05
    2005-06-21 00:00:00
    12/25/05
    2005-12-25 00:00:00
    9/11/05
    2005-09-11 00:00:00
    9/30/05
    2005-09-30 00:00:00
    2/6/05
    2005-02-06 00:00:00
    7/22/05
    2005-07-22 00:00:00
    12/23/05
    2005-12-23 00:00:00
    1/22/05
    2005-01-22 00:00:00
    7/22/05
    2005-07-22 00:00:00
    11/1/05
    2005-11-01 00:00:00
    1/28/05
    2005-01-28 00:00:00
    2/4/05
    2005-02-04 00:00:00
    6/15/05
    2005-06-15 00:00:00
    1/1/05
    2005-01-01 00:00:00
    9/9/05
    2005-09-09 00:00:00
    10/16/05
    2005-10-16 00:00:00
    7/9/05
    2005-07-09 00:00:00
    6/7/05
    2005-06-07 00:00:00
    10/14/05
    2005-10-14 00:00:00
    10/14/05
    2005-10-14 00:00:00
    9/9/05
    2005-09-09 00:00:00
    5/13/05
    2005-05-13 00:00:00
    11/22/05
    2005-11-22 00:00:00
    5/13/05
    2005-05-13 00:00:00
    9/6/05
    2005-09-06 00:00:00
    8/25/05
    2005-08-25 00:00:00
    11/23/05
    2005-11-23 00:00:00
    9/24/05
    2005-09-24 00:00:00
    12/7/05
    2005-12-07 00:00:00
    12/26/05
    2005-12-26 00:00:00
    11/24/05
    2005-11-24 00:00:00
    8/12/05
    2005-08-12 00:00:00
    10/5/05
    2005-10-05 00:00:00
    11/24/05
    2005-11-24 00:00:00
    2/10/05
    2005-02-10 00:00:00
    10/7/05
    2005-10-07 00:00:00
    4/24/05
    2005-04-24 00:00:00
    10/27/05
    2005-10-27 00:00:00
    2/14/05
    2005-02-14 00:00:00
    6/25/05
    2005-06-25 00:00:00
    5/12/05
    2005-05-12 00:00:00
    5/6/05
    2005-05-06 00:00:00
    2/19/05
    2005-02-19 00:00:00
    1/22/05
    2005-01-22 00:00:00
    12/21/05
    2005-12-21 00:00:00
    5/13/05
    2005-05-13 00:00:00
    2/11/05
    2005-02-11 00:00:00
    1/23/05
    2005-01-23 00:00:00
    9/23/05
    2005-09-23 00:00:00
    4/21/05
    2005-04-21 00:00:00
    5/12/05
    2005-05-12 00:00:00
    5/27/05
    2005-05-27 00:00:00
    1/24/05
    2005-01-24 00:00:00
    8/16/05
    2005-08-16 00:00:00
    9/2/05
    2005-09-02 00:00:00
    3/13/05
    2005-03-13 00:00:00
    1/19/05
    2005-01-19 00:00:00
    1/15/05
    2005-01-15 00:00:00
    5/12/05
    2005-05-12 00:00:00
    3/31/05
    2005-03-31 00:00:00
    1/1/05
    2005-01-01 00:00:00
    10/21/05
    2005-10-21 00:00:00
    7/17/05
    2005-07-17 00:00:00
    2/12/05
    2005-02-12 00:00:00
    11/3/05
    2005-11-03 00:00:00
    9/13/05
    2005-09-13 00:00:00
    9/8/05
    2005-09-08 00:00:00
    11/18/05
    2005-11-18 00:00:00
    5/11/05
    2005-05-11 00:00:00
    2/25/05
    2005-02-25 00:00:00
    7/23/05
    2005-07-23 00:00:00
    9/2/05
    2005-09-02 00:00:00
    5/13/05
    2005-05-13 00:00:00
    6/10/05
    2005-06-10 00:00:00
    9/10/05
    2005-09-10 00:00:00
    7/22/05
    2005-07-22 00:00:00
    6/2/05
    2005-06-02 00:00:00
    5/12/05
    2005-05-12 00:00:00
    3/12/05
    2005-03-12 00:00:00
    12/5/05
    2005-12-05 00:00:00
    8/3/05
    2005-08-03 00:00:00
    3/8/05
    2005-03-08 00:00:00
    9/10/05
    2005-09-10 00:00:00
    8/5/05
    2005-08-05 00:00:00
    4/4/05
    2005-04-04 00:00:00
    8/20/05
    2005-08-20 00:00:00
    8/31/05
    2005-08-31 00:00:00
    5/8/05
    2005-05-08 00:00:00
    1/1/05
    2005-01-01 00:00:00
    9/2/05
    2005-09-02 00:00:00
    2/18/05
    2005-02-18 00:00:00
    4/1/05
    2005-04-01 00:00:00
    3/10/05
    2005-03-10 00:00:00
    6/3/05
    2005-06-03 00:00:00
    10/12/05
    2005-10-12 00:00:00
    10/6/05
    2005-10-06 00:00:00
    9/9/05
    2005-09-09 00:00:00
    9/16/05
    2005-09-16 00:00:00
    10/14/05
    2005-10-14 00:00:00
    7/14/05
    2005-07-14 00:00:00
    3/18/05
    2005-03-18 00:00:00
    9/3/05
    2005-09-03 00:00:00
    9/27/05
    2005-09-27 00:00:00
    7/21/05
    2005-07-21 00:00:00
    1/1/05
    2005-01-01 00:00:00
    1/1/05
    2005-01-01 00:00:00
    6/27/05
    2005-06-27 00:00:00
    9/29/05
    2005-09-29 00:00:00
    6/17/05
    2005-06-17 00:00:00
    6/28/05
    2005-06-28 00:00:00
    11/23/05
    2005-11-23 00:00:00
    1/1/05
    2005-01-01 00:00:00
    1/1/05
    2005-01-01 00:00:00
    12/15/05
    2005-12-15 00:00:00
    12/16/05
    2005-12-16 00:00:00
    6/8/05
    2005-06-08 00:00:00
    8/19/05
    2005-08-19 00:00:00
    9/14/05
    2005-09-14 00:00:00
    4/8/05
    2005-04-08 00:00:00
    6/17/05
    2005-06-17 00:00:00
    1/1/05
    2005-01-01 00:00:00
    11/10/05
    2005-11-10 00:00:00
    10/17/05
    2005-10-17 00:00:00
    1/20/05
    2005-01-20 00:00:00
    4/3/05
    2005-04-03 00:00:00
    4/21/05
    2005-04-21 00:00:00
    9/12/05
    2005-09-12 00:00:00
    9/23/05
    2005-09-23 00:00:00
    9/3/05
    2005-09-03 00:00:00
    5/16/05
    2005-05-16 00:00:00
    8/8/05
    2005-08-08 00:00:00
    3/2/05
    2005-03-02 00:00:00
    5/21/05
    2005-05-21 00:00:00
    10/25/05
    2005-10-25 00:00:00
    1/1/05
    2005-01-01 00:00:00
    4/8/05
    2005-04-08 00:00:00
    1/23/05
    2005-01-23 00:00:00
    4/16/05
    2005-04-16 00:00:00
    9/14/05
    2005-09-14 00:00:00
    1/1/05
    2005-01-01 00:00:00
    4/21/05
    2005-04-21 00:00:00
    2/25/05
    2005-02-25 00:00:00
    9/8/05
    2005-09-08 00:00:00
    1/1/05
    2005-01-01 00:00:00
    2/2/05
    2005-02-02 00:00:00
    9/10/05
    2005-09-10 00:00:00
    2/3/05
    2005-02-03 00:00:00
    10/15/05
    2005-10-15 00:00:00
    1/1/05
    2005-01-01 00:00:00
    1/21/05
    2005-01-21 00:00:00
    5/31/05
    2005-05-31 00:00:00
    5/19/05
    2005-05-19 00:00:00
    9/10/05
    2005-09-10 00:00:00
    10/24/05
    2005-10-24 00:00:00
    11/14/05
    2005-11-14 00:00:00
    2/20/05
    2005-02-20 00:00:00
    10/24/05
    2005-10-24 00:00:00
    3/9/05
    2005-03-09 00:00:00
    11/7/05
    2005-11-07 00:00:00
    3/1/05
    2005-03-01 00:00:00
    9/28/05
    2005-09-28 00:00:00
    11/20/05
    2005-11-20 00:00:00
    1/21/05
    2005-01-21 00:00:00
    5/11/05
    2005-05-11 00:00:00
    10/15/05
    2005-10-15 00:00:00
    8/18/05
    2005-08-18 00:00:00
    11/4/05
    2005-11-04 00:00:00
    3/26/05
    2005-03-26 00:00:00
    3/7/05
    2005-03-07 00:00:00
    5/15/05
    2005-05-15 00:00:00
    1/1/05
    2005-01-01 00:00:00
    4/22/05
    2005-04-22 00:00:00
    11/5/05
    2005-11-05 00:00:00
    9/14/05
    2005-09-14 00:00:00
    9/3/05
    2005-09-03 00:00:00
    1/23/05
    2005-01-23 00:00:00
    9/12/05
    2005-09-12 00:00:00
    4/13/05
    2005-04-13 00:00:00
    1/1/05
    2005-01-01 00:00:00
    6/24/05
    2005-06-24 00:00:00
    1/12/06
    2006-01-12 00:00:00
    6/20/06
    2006-06-20 00:00:00
    6/8/06
    2006-06-08 00:00:00
    11/14/06
    2006-11-14 00:00:00
    5/17/06
    2006-05-17 00:00:00
    8/11/06
    2006-08-11 00:00:00
    3/23/06
    2006-03-23 00:00:00
    5/3/06
    2006-05-03 00:00:00
    6/30/06
    2006-06-30 00:00:00
    10/5/06
    2006-10-05 00:00:00
    12/8/06
    2006-12-08 00:00:00
    10/19/06
    2006-10-19 00:00:00
    10/20/06
    2006-10-20 00:00:00
    9/8/06
    2006-09-08 00:00:00
    9/22/06
    2006-09-22 00:00:00
    12/7/06
    2006-12-07 00:00:00
    6/28/06
    2006-06-28 00:00:00
    11/2/06
    2006-11-02 00:00:00
    6/22/06
    2006-06-22 00:00:00
    8/16/06
    2006-08-16 00:00:00
    6/1/06
    2006-06-01 00:00:00
    3/23/06
    2006-03-23 00:00:00
    4/22/06
    2006-04-22 00:00:00
    9/13/06
    2006-09-13 00:00:00
    12/7/06
    2006-12-07 00:00:00
    1/27/06
    2006-01-27 00:00:00
    8/31/06
    2006-08-31 00:00:00
    2/24/06
    2006-02-24 00:00:00
    2/17/06
    2006-02-17 00:00:00
    2/10/06
    2006-02-10 00:00:00
    4/21/06
    2006-04-21 00:00:00
    3/17/06
    2006-03-17 00:00:00
    9/6/06
    2006-09-06 00:00:00
    11/16/06
    2006-11-16 00:00:00
    12/20/06
    2006-12-20 00:00:00
    12/1/06
    2006-12-01 00:00:00
    10/22/06
    2006-10-22 00:00:00
    3/6/06
    2006-03-06 00:00:00
    11/21/06
    2006-11-21 00:00:00
    12/14/06
    2006-12-14 00:00:00
    6/16/06
    2006-06-16 00:00:00
    4/13/06
    2006-04-13 00:00:00
    7/21/06
    2006-07-21 00:00:00
    1/20/06
    2006-01-20 00:00:00
    3/10/06
    2006-03-10 00:00:00
    12/7/06
    2006-12-07 00:00:00
    7/27/06
    2006-07-27 00:00:00
    3/10/06
    2006-03-10 00:00:00
    6/15/06
    2006-06-15 00:00:00
    5/12/06
    2006-05-12 00:00:00
    12/11/06
    2006-12-11 00:00:00
    3/3/06
    2006-03-03 00:00:00
    3/1/06
    2006-03-01 00:00:00
    7/27/06
    2006-07-27 00:00:00
    7/27/06
    2006-07-27 00:00:00
    4/7/06
    2006-04-07 00:00:00
    10/19/06
    2006-10-19 00:00:00
    5/20/06
    2006-05-20 00:00:00
    5/18/06
    2006-05-18 00:00:00
    2/10/06
    2006-02-10 00:00:00
    9/1/06
    2006-09-01 00:00:00
    12/9/06
    2006-12-09 00:00:00
    9/16/06
    2006-09-16 00:00:00
    4/28/06
    2006-04-28 00:00:00
    1/13/06
    2006-01-13 00:00:00
    12/12/06
    2006-12-12 00:00:00
    8/18/06
    2006-08-18 00:00:00
    1/18/06
    2006-01-18 00:00:00
    8/25/06
    2006-08-25 00:00:00
    4/5/06
    2006-04-05 00:00:00
    9/10/06
    2006-09-10 00:00:00
    3/31/06
    2006-03-31 00:00:00
    1/26/06
    2006-01-26 00:00:00
    3/29/06
    2006-03-29 00:00:00
    5/25/06
    2006-05-25 00:00:00
    8/4/06
    2006-08-04 00:00:00
    4/11/06
    2006-04-11 00:00:00
    12/14/06
    2006-12-14 00:00:00
    5/24/06
    2006-05-24 00:00:00
    2/17/06
    2006-02-17 00:00:00
    7/27/06
    2006-07-27 00:00:00
    11/22/06
    2006-11-22 00:00:00
    9/9/06
    2006-09-09 00:00:00
    9/1/06
    2006-09-01 00:00:00
    10/27/06
    2006-10-27 00:00:00
    9/1/06
    2006-09-01 00:00:00
    6/16/06
    2006-06-16 00:00:00
    9/15/06
    2006-09-15 00:00:00
    12/12/06
    2006-12-12 00:00:00
    5/12/06
    2006-05-12 00:00:00
    3/4/06
    2006-03-04 00:00:00
    10/10/06
    2006-10-10 00:00:00
    8/8/06
    2006-08-08 00:00:00
    8/30/06
    2006-08-30 00:00:00
    7/14/06
    2006-07-14 00:00:00
    7/26/06
    2006-07-26 00:00:00
    8/31/06
    2006-08-31 00:00:00
    11/9/06
    2006-11-09 00:00:00
    6/23/06
    2006-06-23 00:00:00
    12/1/06
    2006-12-01 00:00:00
    10/6/06
    2006-10-06 00:00:00
    6/6/06
    2006-06-06 00:00:00
    12/19/06
    2006-12-19 00:00:00
    7/7/06
    2006-07-07 00:00:00
    5/25/06
    2006-05-25 00:00:00
    10/10/06
    2006-10-10 00:00:00
    9/9/06
    2006-09-09 00:00:00
    8/9/06
    2006-08-09 00:00:00
    11/1/06
    2006-11-01 00:00:00
    9/9/06
    2006-09-09 00:00:00
    7/10/06
    2006-07-10 00:00:00
    12/9/06
    2006-12-09 00:00:00
    2/9/06
    2006-02-09 00:00:00
    12/26/06
    2006-12-26 00:00:00
    2/17/06
    2006-02-17 00:00:00
    8/18/06
    2006-08-18 00:00:00
    1/6/06
    2006-01-06 00:00:00
    9/29/06
    2006-09-29 00:00:00
    9/28/06
    2006-09-28 00:00:00
    8/4/06
    2006-08-04 00:00:00
    11/3/06
    2006-11-03 00:00:00
    9/5/06
    2006-09-05 00:00:00
    12/25/06
    2006-12-25 00:00:00
    8/6/06
    2006-08-06 00:00:00
    2/15/06
    2006-02-15 00:00:00
    9/7/06
    2006-09-07 00:00:00
    9/22/06
    2006-09-22 00:00:00
    9/28/06
    2006-09-28 00:00:00
    5/17/06
    2006-05-17 00:00:00
    5/15/06
    2006-05-15 00:00:00
    6/3/06
    2006-06-03 00:00:00
    8/25/06
    2006-08-25 00:00:00
    12/3/06
    2006-12-03 00:00:00
    8/11/06
    2006-08-11 00:00:00
    2/1/06
    2006-02-01 00:00:00
    12/8/06
    2006-12-08 00:00:00
    11/8/06
    2006-11-08 00:00:00
    3/3/06
    2006-03-03 00:00:00
    4/21/06
    2006-04-21 00:00:00
    4/25/06
    2006-04-25 00:00:00
    10/14/06
    2006-10-14 00:00:00
    10/13/06
    2006-10-13 00:00:00
    5/23/06
    2006-05-23 00:00:00
    4/27/06
    2006-04-27 00:00:00
    10/27/06
    2006-10-27 00:00:00
    3/9/06
    2006-03-09 00:00:00
    7/21/06
    2006-07-21 00:00:00
    2/24/06
    2006-02-24 00:00:00
    12/15/06
    2006-12-15 00:00:00
    1/17/06
    2006-01-17 00:00:00
    10/6/06
    2006-10-06 00:00:00
    4/7/06
    2006-04-07 00:00:00
    10/27/06
    2006-10-27 00:00:00
    12/15/06
    2006-12-15 00:00:00
    3/17/06
    2006-03-17 00:00:00
    8/11/06
    2006-08-11 00:00:00
    4/25/06
    2006-04-25 00:00:00
    3/14/06
    2006-03-14 00:00:00
    7/27/06
    2006-07-27 00:00:00
    6/24/06
    2006-06-24 00:00:00
    3/3/06
    2006-03-03 00:00:00
    10/9/06
    2006-10-09 00:00:00
    9/6/06
    2006-09-06 00:00:00
    9/9/06
    2006-09-09 00:00:00
    5/18/06
    2006-05-18 00:00:00
    12/1/06
    2006-12-01 00:00:00
    3/1/06
    2006-03-01 00:00:00
    7/27/06
    2006-07-27 00:00:00
    11/30/06
    2006-11-30 00:00:00
    1/13/06
    2006-01-13 00:00:00
    9/10/06
    2006-09-10 00:00:00
    12/25/06
    2006-12-25 00:00:00
    4/28/06
    2006-04-28 00:00:00
    3/3/06
    2006-03-03 00:00:00
    9/7/06
    2006-09-07 00:00:00
    1/29/06
    2006-01-29 00:00:00
    9/15/06
    2006-09-15 00:00:00
    1/4/06
    2006-01-04 00:00:00
    4/19/06
    2006-04-19 00:00:00
    6/8/06
    2006-06-08 00:00:00
    12/8/06
    2006-12-08 00:00:00
    9/22/06
    2006-09-22 00:00:00
    12/28/06
    2006-12-28 00:00:00
    6/20/06
    2006-06-20 00:00:00
    1/6/06
    2006-01-06 00:00:00
    10/23/06
    2006-10-23 00:00:00
    12/15/06
    2006-12-15 00:00:00
    6/16/06
    2006-06-16 00:00:00
    4/30/06
    2006-04-30 00:00:00
    2/15/06
    2006-02-15 00:00:00
    3/8/06
    2006-03-08 00:00:00
    10/20/06
    2006-10-20 00:00:00
    10/13/06
    2006-10-13 00:00:00
    10/6/06
    2006-10-06 00:00:00
    8/10/06
    2006-08-10 00:00:00
    1/1/06
    2006-01-01 00:00:00
    5/19/06
    2006-05-19 00:00:00
    5/2/06
    2006-05-02 00:00:00
    9/10/06
    2006-09-10 00:00:00
    2/19/06
    2006-02-19 00:00:00
    11/10/06
    2006-11-10 00:00:00
    2/10/06
    2006-02-10 00:00:00
    9/13/06
    2006-09-13 00:00:00
    9/11/06
    2006-09-11 00:00:00
    10/20/06
    2006-10-20 00:00:00
    1/1/06
    2006-01-01 00:00:00
    9/7/06
    2006-09-07 00:00:00
    9/15/06
    2006-09-15 00:00:00
    9/8/06
    2006-09-08 00:00:00
    9/1/06
    2006-09-01 00:00:00
    11/22/06
    2006-11-22 00:00:00
    8/31/06
    2006-08-31 00:00:00
    2/21/06
    2006-02-21 00:00:00
    2/12/06
    2006-02-12 00:00:00
    10/9/06
    2006-10-09 00:00:00
    6/28/06
    2006-06-28 00:00:00
    9/8/06
    2006-09-08 00:00:00
    10/28/06
    2006-10-28 00:00:00
    10/17/06
    2006-10-17 00:00:00
    4/11/06
    2006-04-11 00:00:00
    3/16/06
    2006-03-16 00:00:00
    10/7/06
    2006-10-07 00:00:00
    10/13/06
    2006-10-13 00:00:00
    8/9/06
    2006-08-09 00:00:00
    9/14/06
    2006-09-14 00:00:00
    3/30/06
    2006-03-30 00:00:00
    3/24/06
    2006-03-24 00:00:00
    2/3/06
    2006-02-03 00:00:00
    11/16/06
    2006-11-16 00:00:00
    4/28/06
    2006-04-28 00:00:00
    3/20/06
    2006-03-20 00:00:00
    9/9/06
    2006-09-09 00:00:00
    11/24/06
    2006-11-24 00:00:00
    12/25/06
    2006-12-25 00:00:00
    10/15/06
    2006-10-15 00:00:00
    10/9/06
    2006-10-09 00:00:00
    5/16/06
    2006-05-16 00:00:00
    4/21/06
    2006-04-21 00:00:00
    1/3/06
    2006-01-03 00:00:00
    9/7/06
    2006-09-07 00:00:00
    9/11/06
    2006-09-11 00:00:00
    3/31/06
    2006-03-31 00:00:00
    8/11/06
    2006-08-11 00:00:00
    5/19/06
    2006-05-19 00:00:00
    8/22/06
    2006-08-22 00:00:00
    7/21/06
    2006-07-21 00:00:00
    12/8/06
    2006-12-08 00:00:00
    3/30/06
    2006-03-30 00:00:00
    1/1/06
    2006-01-01 00:00:00
    3/8/06
    2006-03-08 00:00:00
    4/25/06
    2006-04-25 00:00:00
    8/18/06
    2006-08-18 00:00:00
    12/1/06
    2006-12-01 00:00:00
    9/15/06
    2006-09-15 00:00:00
    11/10/06
    2006-11-10 00:00:00
    9/9/06
    2006-09-09 00:00:00
    10/22/06
    2006-10-22 00:00:00
    5/26/06
    2006-05-26 00:00:00
    1/24/06
    2006-01-24 00:00:00
    4/18/06
    2006-04-18 00:00:00
    4/30/06
    2006-04-30 00:00:00
    4/27/06
    2006-04-27 00:00:00
    10/26/06
    2006-10-26 00:00:00
    2/2/06
    2006-02-02 00:00:00
    6/23/06
    2006-06-23 00:00:00
    12/26/06
    2006-12-26 00:00:00
    3/11/06
    2006-03-11 00:00:00
    9/29/06
    2006-09-29 00:00:00
    3/29/06
    2006-03-29 00:00:00
    12/1/06
    2006-12-01 00:00:00
    3/24/06
    2006-03-24 00:00:00
    9/1/06
    2006-09-01 00:00:00
    5/19/06
    2006-05-19 00:00:00
    12/18/06
    2006-12-18 00:00:00
    1/1/06
    2006-01-01 00:00:00
    12/9/06
    2006-12-09 00:00:00
    9/11/06
    2006-09-11 00:00:00
    5/20/06
    2006-05-20 00:00:00
    10/2/06
    2006-10-02 00:00:00
    7/15/06
    2006-07-15 00:00:00
    9/13/06
    2006-09-13 00:00:00
    3/26/06
    2006-03-26 00:00:00
    9/2/06
    2006-09-02 00:00:00
    2/27/06
    2006-02-27 00:00:00
    1/24/06
    2006-01-24 00:00:00
    11/17/06
    2006-11-17 00:00:00
    1/30/06
    2006-01-30 00:00:00
    9/6/06
    2006-09-06 00:00:00
    8/11/06
    2006-08-11 00:00:00
    4/7/06
    2006-04-07 00:00:00
    6/6/06
    2006-06-06 00:00:00
    5/1/06
    2006-05-01 00:00:00
    12/11/06
    2006-12-11 00:00:00
    4/26/06
    2006-04-26 00:00:00
    1/1/06
    2006-01-01 00:00:00
    9/19/06
    2006-09-19 00:00:00
    2/15/06
    2006-02-15 00:00:00
    9/12/06
    2006-09-12 00:00:00
    1/1/06
    2006-01-01 00:00:00
    1/12/06
    2006-01-12 00:00:00
    2/7/06
    2006-02-07 00:00:00
    9/12/06
    2006-09-12 00:00:00
    11/19/06
    2006-11-19 00:00:00
    9/29/06
    2006-09-29 00:00:00
    9/29/06
    2006-09-29 00:00:00
    1/27/06
    2006-01-27 00:00:00
    12/5/06
    2006-12-05 00:00:00
    5/25/06
    2006-05-25 00:00:00
    7/28/06
    2006-07-28 00:00:00
    3/24/06
    2006-03-24 00:00:00
    1/15/06
    2006-01-15 00:00:00
    4/5/06
    2006-04-05 00:00:00
    6/24/06
    2006-06-24 00:00:00
    9/25/06
    2006-09-25 00:00:00
    10/20/06
    2006-10-20 00:00:00
    1/26/06
    2006-01-26 00:00:00
    9/23/06
    2006-09-23 00:00:00
    3/24/06
    2006-03-24 00:00:00
    4/14/06
    2006-04-14 00:00:00
    11/10/06
    2006-11-10 00:00:00
    1/1/06
    2006-01-01 00:00:00
    6/29/06
    2006-06-29 00:00:00
    6/9/06
    2006-06-09 00:00:00
    11/1/06
    2006-11-01 00:00:00
    10/25/06
    2006-10-25 00:00:00
    7/5/06
    2006-07-05 00:00:00
    10/13/06
    2006-10-13 00:00:00
    3/9/06
    2006-03-09 00:00:00
    10/3/06
    2006-10-03 00:00:00
    5/24/06
    2006-05-24 00:00:00
    8/15/06
    2006-08-15 00:00:00
    12/6/06
    2006-12-06 00:00:00
    11/7/06
    2006-11-07 00:00:00
    1/23/06
    2006-01-23 00:00:00
    2/21/06
    2006-02-21 00:00:00
    9/26/06
    2006-09-26 00:00:00
    1/23/06
    2006-01-23 00:00:00
    3/22/06
    2006-03-22 00:00:00
    12/19/06
    2006-12-19 00:00:00
    1/18/06
    2006-01-18 00:00:00
    12/10/06
    2006-12-10 00:00:00
    4/20/06
    2006-04-20 00:00:00
    4/28/06
    2006-04-28 00:00:00
    12/1/06
    2006-12-01 00:00:00
    5/23/06
    2006-05-23 00:00:00
    9/12/06
    2006-09-12 00:00:00
    6/10/06
    2006-06-10 00:00:00
    3/10/06
    2006-03-10 00:00:00
    12/29/06
    2006-12-29 00:00:00
    5/23/06
    2006-05-23 00:00:00
    10/21/06
    2006-10-21 00:00:00
    6/20/06
    2006-06-20 00:00:00
    5/1/06
    2006-05-01 00:00:00
    9/15/06
    2006-09-15 00:00:00
    9/9/06
    2006-09-09 00:00:00
    9/11/06
    2006-09-11 00:00:00
    1/26/06
    2006-01-26 00:00:00
    7/9/06
    2006-07-09 00:00:00
    1/1/06
    2006-01-01 00:00:00
    6/23/06
    2006-06-23 00:00:00
    10/6/06
    2006-10-06 00:00:00
    12/1/06
    2006-12-01 00:00:00
    3/18/06
    2006-03-18 00:00:00
    9/15/06
    2006-09-15 00:00:00
    5/5/06
    2006-05-05 00:00:00
    7/31/06
    2006-07-31 00:00:00
    4/30/06
    2006-04-30 00:00:00
    1/1/06
    2006-01-01 00:00:00
    5/5/06
    2006-05-05 00:00:00
    9/9/06
    2006-09-09 00:00:00
    10/27/06
    2006-10-27 00:00:00
    11/22/06
    2006-11-22 00:00:00
    6/21/06
    2006-06-21 00:00:00
    9/15/06
    2006-09-15 00:00:00
    2/16/06
    2006-02-16 00:00:00
    9/2/06
    2006-09-02 00:00:00
    1/1/06
    2006-01-01 00:00:00
    1/1/06
    2006-01-01 00:00:00
    12/1/06
    2006-12-01 00:00:00
    5/26/06
    2006-05-26 00:00:00
    4/5/06
    2006-04-05 00:00:00
    4/7/06
    2006-04-07 00:00:00
    9/4/06
    2006-09-04 00:00:00
    2/24/06
    2006-02-24 00:00:00
    7/21/06
    2006-07-21 00:00:00
    5/1/06
    2006-05-01 00:00:00
    1/9/06
    2006-01-09 00:00:00
    9/24/06
    2006-09-24 00:00:00
    1/1/06
    2006-01-01 00:00:00
    1/20/06
    2006-01-20 00:00:00
    7/28/06
    2006-07-28 00:00:00
    6/24/06
    2006-06-24 00:00:00
    9/5/06
    2006-09-05 00:00:00
    10/20/06
    2006-10-20 00:00:00
    1/30/06
    2006-01-30 00:00:00
    11/10/06
    2006-11-10 00:00:00
    8/8/06
    2006-08-08 00:00:00
    8/25/06
    2006-08-25 00:00:00
    5/10/06
    2006-05-10 00:00:00
    10/15/06
    2006-10-15 00:00:00
    9/13/06
    2006-09-13 00:00:00
    5/23/06
    2006-05-23 00:00:00
    3/17/06
    2006-03-17 00:00:00
    10/3/06
    2006-10-03 00:00:00
    3/1/06
    2006-03-01 00:00:00
    1/1/06
    2006-01-01 00:00:00
    5/25/06
    2006-05-25 00:00:00
    9/22/06
    2006-09-22 00:00:00
    5/31/04
    2004-05-31 00:00:00
    4/16/04
    2004-04-16 00:00:00
    11/10/04
    2004-11-10 00:00:00
    3/19/04
    2004-03-19 00:00:00
    6/25/04
    2004-06-25 00:00:00
    11/5/04
    2004-11-05 00:00:00
    7/23/04
    2004-07-23 00:00:00
    5/13/04
    2004-05-13 00:00:00
    11/19/04
    2004-11-19 00:00:00
    7/15/04
    2004-07-15 00:00:00
    2/13/04
    2004-02-13 00:00:00
    4/30/04
    2004-04-30 00:00:00
    5/5/04
    2004-05-05 00:00:00
    4/2/04
    2004-04-02 00:00:00
    11/9/04
    2004-11-09 00:00:00
    5/19/04
    2004-05-19 00:00:00
    1/22/04
    2004-01-22 00:00:00
    9/10/04
    2004-09-10 00:00:00
    12/9/04
    2004-12-09 00:00:00
    12/15/04
    2004-12-15 00:00:00
    5/26/04
    2004-05-26 00:00:00
    12/25/04
    2004-12-25 00:00:00
    8/4/04
    2004-08-04 00:00:00
    4/9/04
    2004-04-09 00:00:00
    12/8/04
    2004-12-08 00:00:00
    4/20/04
    2004-04-20 00:00:00
    6/17/04
    2004-06-17 00:00:00
    12/16/04
    2004-12-16 00:00:00
    12/14/04
    2004-12-14 00:00:00
    6/10/04
    2004-06-10 00:00:00
    2/6/04
    2004-02-06 00:00:00
    9/20/04
    2004-09-20 00:00:00
    9/17/04
    2004-09-17 00:00:00
    11/3/04
    2004-11-03 00:00:00
    11/14/04
    2004-11-14 00:00:00
    11/26/04
    2004-11-26 00:00:00
    5/28/04
    2004-05-28 00:00:00
    7/9/04
    2004-07-09 00:00:00
    4/13/04
    2004-04-13 00:00:00
    4/9/04
    2004-04-09 00:00:00
    4/9/04
    2004-04-09 00:00:00
    3/24/04
    2004-03-24 00:00:00
    10/17/04
    2004-10-17 00:00:00
    7/10/04
    2004-07-10 00:00:00
    8/12/04
    2004-08-12 00:00:00
    6/18/04
    2004-06-18 00:00:00
    11/9/04
    2004-11-09 00:00:00
    11/21/04
    2004-11-21 00:00:00
    5/30/04
    2004-05-30 00:00:00
    10/1/04
    2004-10-01 00:00:00
    6/25/04
    2004-06-25 00:00:00
    2/20/04
    2004-02-20 00:00:00
    4/23/04
    2004-04-23 00:00:00
    1/16/04
    2004-01-16 00:00:00
    12/1/04
    2004-12-01 00:00:00
    4/15/04
    2004-04-15 00:00:00
    7/22/04
    2004-07-22 00:00:00
    11/24/04
    2004-11-24 00:00:00
    7/16/04
    2004-07-16 00:00:00
    3/19/04
    2004-03-19 00:00:00
    9/28/04
    2004-09-28 00:00:00
    7/7/04
    2004-07-07 00:00:00
    6/11/04
    2004-06-11 00:00:00
    12/17/04
    2004-12-17 00:00:00
    3/12/04
    2004-03-12 00:00:00
    9/30/04
    2004-09-30 00:00:00
    11/12/04
    2004-11-12 00:00:00
    7/23/04
    2004-07-23 00:00:00
    8/6/04
    2004-08-06 00:00:00
    4/7/04
    2004-04-07 00:00:00
    2/10/04
    2004-02-10 00:00:00
    10/22/04
    2004-10-22 00:00:00
    9/6/04
    2004-09-06 00:00:00
    7/30/04
    2004-07-30 00:00:00
    2/9/04
    2004-02-09 00:00:00
    11/11/04
    2004-11-11 00:00:00
    10/6/04
    2004-10-06 00:00:00
    10/22/04
    2004-10-22 00:00:00
    10/18/04
    2004-10-18 00:00:00
    10/15/04
    2004-10-15 00:00:00
    3/25/04
    2004-03-25 00:00:00
    6/10/04
    2004-06-10 00:00:00
    10/8/04
    2004-10-08 00:00:00
    4/17/04
    2004-04-17 00:00:00
    4/2/04
    2004-04-02 00:00:00
    3/5/04
    2004-03-05 00:00:00
    9/10/04
    2004-09-10 00:00:00
    6/25/04
    2004-06-25 00:00:00
    1/23/04
    2004-01-23 00:00:00
    6/23/04
    2004-06-23 00:00:00
    9/11/04
    2004-09-11 00:00:00
    3/6/04
    2004-03-06 00:00:00
    4/2/04
    2004-04-02 00:00:00
    2/27/04
    2004-02-27 00:00:00
    8/3/04
    2004-08-03 00:00:00
    8/6/04
    2004-08-06 00:00:00
    10/26/04
    2004-10-26 00:00:00
    10/11/04
    2004-10-11 00:00:00
    1/2/04
    2004-01-02 00:00:00
    1/1/04
    2004-01-01 00:00:00
    5/1/04
    2004-05-01 00:00:00
    3/13/04
    2004-03-13 00:00:00
    7/14/04
    2004-07-14 00:00:00
    10/22/04
    2004-10-22 00:00:00
    8/20/04
    2004-08-20 00:00:00
    1/21/04
    2004-01-21 00:00:00
    3/5/04
    2004-03-05 00:00:00
    10/15/04
    2004-10-15 00:00:00
    2/17/04
    2004-02-17 00:00:00
    7/30/04
    2004-07-30 00:00:00
    4/7/04
    2004-04-07 00:00:00
    2/25/04
    2004-02-25 00:00:00
    8/26/04
    2004-08-26 00:00:00
    10/29/04
    2004-10-29 00:00:00
    2/27/04
    2004-02-27 00:00:00
    1/28/04
    2004-01-28 00:00:00
    2/11/04
    2004-02-11 00:00:00
    9/8/04
    2004-09-08 00:00:00
    9/13/04
    2004-09-13 00:00:00
    3/19/04
    2004-03-19 00:00:00
    1/30/04
    2004-01-30 00:00:00
    6/20/04
    2004-06-20 00:00:00
    12/8/04
    2004-12-08 00:00:00
    10/15/04
    2004-10-15 00:00:00
    1/16/04
    2004-01-16 00:00:00
    6/11/04
    2004-06-11 00:00:00
    9/24/04
    2004-09-24 00:00:00
    8/20/04
    2004-08-20 00:00:00
    7/9/04
    2004-07-09 00:00:00
    9/3/04
    2004-09-03 00:00:00
    9/17/04
    2004-09-17 00:00:00
    12/10/04
    2004-12-10 00:00:00
    6/16/04
    2004-06-16 00:00:00
    1/20/04
    2004-01-20 00:00:00
    2/9/04
    2004-02-09 00:00:00
    3/28/04
    2004-03-28 00:00:00
    12/8/04
    2004-12-08 00:00:00
    6/6/04
    2004-06-06 00:00:00
    2/13/04
    2004-02-13 00:00:00
    5/11/04
    2004-05-11 00:00:00
    9/24/04
    2004-09-24 00:00:00
    9/21/04
    2004-09-21 00:00:00
    9/16/04
    2004-09-16 00:00:00
    1/23/04
    2004-01-23 00:00:00
    4/29/04
    2004-04-29 00:00:00
    12/29/04
    2004-12-29 00:00:00
    6/14/04
    2004-06-14 00:00:00
    10/22/04
    2004-10-22 00:00:00
    1/1/04
    2004-01-01 00:00:00
    12/5/04
    2004-12-05 00:00:00
    12/28/04
    2004-12-28 00:00:00
    2/19/04
    2004-02-19 00:00:00
    12/17/04
    2004-12-17 00:00:00
    12/17/04
    2004-12-17 00:00:00
    4/30/04
    2004-04-30 00:00:00
    11/12/04
    2004-11-12 00:00:00
    9/12/04
    2004-09-12 00:00:00
    10/8/04
    2004-10-08 00:00:00
    3/25/04
    2004-03-25 00:00:00
    10/29/04
    2004-10-29 00:00:00
    5/27/04
    2004-05-27 00:00:00
    9/3/04
    2004-09-03 00:00:00
    1/1/04
    2004-01-01 00:00:00
    7/30/04
    2004-07-30 00:00:00
    9/3/04
    2004-09-03 00:00:00
    4/13/04
    2004-04-13 00:00:00
    4/4/04
    2004-04-04 00:00:00
    8/10/04
    2004-08-10 00:00:00
    11/19/04
    2004-11-19 00:00:00
    10/6/04
    2004-10-06 00:00:00
    9/1/04
    2004-09-01 00:00:00
    4/14/04
    2004-04-14 00:00:00
    9/4/04
    2004-09-04 00:00:00
    4/7/04
    2004-04-07 00:00:00
    5/17/04
    2004-05-17 00:00:00
    6/21/04
    2004-06-21 00:00:00
    5/25/04
    2004-05-25 00:00:00
    6/11/04
    2004-06-11 00:00:00
    7/2/04
    2004-07-02 00:00:00
    1/21/04
    2004-01-21 00:00:00
    1/16/04
    2004-01-16 00:00:00
    2/6/04
    2004-02-06 00:00:00
    1/1/04
    2004-01-01 00:00:00
    5/7/04
    2004-05-07 00:00:00
    12/7/04
    2004-12-07 00:00:00
    9/3/04
    2004-09-03 00:00:00
    6/21/04
    2004-06-21 00:00:00
    5/21/04
    2004-05-21 00:00:00
    12/31/04
    2004-12-31 00:00:00
    1/16/04
    2004-01-16 00:00:00
    5/28/04
    2004-05-28 00:00:00
    4/24/04
    2004-04-24 00:00:00
    7/30/04
    2004-07-30 00:00:00
    2/27/04
    2004-02-27 00:00:00
    9/11/04
    2004-09-11 00:00:00
    3/9/04
    2004-03-09 00:00:00
    1/9/04
    2004-01-09 00:00:00
    8/27/04
    2004-08-27 00:00:00
    9/9/04
    2004-09-09 00:00:00
    1/30/04
    2004-01-30 00:00:00
    5/18/04
    2004-05-18 00:00:00
    8/12/04
    2004-08-12 00:00:00
    12/5/04
    2004-12-05 00:00:00
    4/30/04
    2004-04-30 00:00:00
    9/15/04
    2004-09-15 00:00:00
    6/15/04
    2004-06-15 00:00:00
    1/1/04
    2004-01-01 00:00:00
    4/16/04
    2004-04-16 00:00:00
    5/14/04
    2004-05-14 00:00:00
    10/15/04
    2004-10-15 00:00:00
    3/12/04
    2004-03-12 00:00:00
    12/3/04
    2004-12-03 00:00:00
    4/7/04
    2004-04-07 00:00:00
    10/9/04
    2004-10-09 00:00:00
    10/15/04
    2004-10-15 00:00:00
    9/3/04
    2004-09-03 00:00:00
    1/17/04
    2004-01-17 00:00:00
    7/23/04
    2004-07-23 00:00:00
    8/6/04
    2004-08-06 00:00:00
    9/3/04
    2004-09-03 00:00:00
    12/12/04
    2004-12-12 00:00:00
    2/20/04
    2004-02-20 00:00:00
    3/12/04
    2004-03-12 00:00:00
    2/14/04
    2004-02-14 00:00:00
    9/2/04
    2004-09-02 00:00:00
    9/11/04
    2004-09-11 00:00:00
    1/17/04
    2004-01-17 00:00:00
    9/12/04
    2004-09-12 00:00:00
    12/13/04
    2004-12-13 00:00:00
    12/28/04
    2004-12-28 00:00:00
    9/12/04
    2004-09-12 00:00:00
    4/18/04
    2004-04-18 00:00:00
    1/1/04
    2004-01-01 00:00:00
    5/18/04
    2004-05-18 00:00:00
    5/13/04
    2004-05-13 00:00:00
    10/1/04
    2004-10-01 00:00:00
    5/6/04
    2004-05-06 00:00:00
    10/8/04
    2004-10-08 00:00:00
    7/16/04
    2004-07-16 00:00:00
    4/23/04
    2004-04-23 00:00:00
    3/18/04
    2004-03-18 00:00:00
    8/13/04
    2004-08-13 00:00:00
    2/5/04
    2004-02-05 00:00:00
    2/6/04
    2004-02-06 00:00:00
    1/2/04
    2004-01-02 00:00:00
    11/27/04
    2004-11-27 00:00:00
    6/15/04
    2004-06-15 00:00:00
    12/24/04
    2004-12-24 00:00:00
    4/14/04
    2004-04-14 00:00:00
    1/1/04
    2004-01-01 00:00:00
    8/20/04
    2004-08-20 00:00:00
    5/14/04
    2004-05-14 00:00:00
    10/20/04
    2004-10-20 00:00:00
    1/1/04
    2004-01-01 00:00:00
    7/18/04
    2004-07-18 00:00:00
    10/7/04
    2004-10-07 00:00:00
    11/12/04
    2004-11-12 00:00:00
    10/15/04
    2004-10-15 00:00:00
    8/27/04
    2004-08-27 00:00:00
    7/9/04
    2004-07-09 00:00:00
    11/26/04
    2004-11-26 00:00:00
    1/20/04
    2004-01-20 00:00:00
    7/30/04
    2004-07-30 00:00:00
    1/15/04
    2004-01-15 00:00:00
    8/12/04
    2004-08-12 00:00:00
    2/13/04
    2004-02-13 00:00:00
    1/20/04
    2004-01-20 00:00:00
    9/29/04
    2004-09-29 00:00:00
    10/5/04
    2004-10-05 00:00:00
    7/2/04
    2004-07-02 00:00:00
    9/4/04
    2004-09-04 00:00:00
    9/17/04
    2004-09-17 00:00:00
    2/24/04
    2004-02-24 00:00:00
    8/9/04
    2004-08-09 00:00:00
    4/17/04
    2004-04-17 00:00:00
    4/30/04
    2004-04-30 00:00:00
    8/10/04
    2004-08-10 00:00:00
    10/21/04
    2004-10-21 00:00:00
    8/27/04
    2004-08-27 00:00:00
    1/12/04
    2004-01-12 00:00:00
    1/1/04
    2004-01-01 00:00:00
    10/15/04
    2004-10-15 00:00:00
    3/26/04
    2004-03-26 00:00:00
    5/30/04
    2004-05-30 00:00:00
    2/27/04
    2004-02-27 00:00:00
    10/19/04
    2004-10-19 00:00:00
    1/1/04
    2004-01-01 00:00:00
    9/10/04
    2004-09-10 00:00:00
    12/1/04
    2004-12-01 00:00:00
    6/11/04
    2004-06-11 00:00:00
    1/21/04
    2004-01-21 00:00:00
    1/30/04
    2004-01-30 00:00:00
    10/8/04
    2004-10-08 00:00:00
    9/3/04
    2004-09-03 00:00:00
    6/1/04
    2004-06-01 00:00:00
    9/8/04
    2004-09-08 00:00:00
    1/15/04
    2004-01-15 00:00:00
    9/11/04
    2004-09-11 00:00:00
    1/13/04
    2004-01-13 00:00:00
    7/16/04
    2004-07-16 00:00:00
    12/3/04
    2004-12-03 00:00:00
    1/23/04
    2004-01-23 00:00:00
    5/14/04
    2004-05-14 00:00:00
    6/11/04
    2004-06-11 00:00:00
    10/29/04
    2004-10-29 00:00:00
    10/3/04
    2004-10-03 00:00:00
    12/8/04
    2004-12-08 00:00:00
    3/15/72
    1972-03-15 00:00:00
    7/30/72
    1972-07-30 00:00:00
    6/29/72
    1972-06-29 00:00:00
    2/13/72
    1972-02-13 00:00:00
    5/25/72
    1972-05-25 00:00:00
    7/14/72
    1972-07-14 00:00:00
    3/9/72
    1972-03-09 00:00:00
    3/15/72
    1972-03-15 00:00:00
    3/12/72
    1972-03-12 00:00:00
    8/6/72
    1972-08-06 00:00:00
    9/30/72
    1972-09-30 00:00:00
    1/1/72
    1972-01-01 00:00:00
    3/9/72
    1972-03-09 00:00:00
    9/10/72
    1972-09-10 00:00:00
    1/1/72
    1972-01-01 00:00:00
    12/1/72
    1972-12-01 00:00:00
    10/6/72
    1972-10-06 00:00:00
    12/10/72
    1972-12-10 00:00:00
    1/13/72
    1972-01-13 00:00:00
    12/13/72
    1972-12-13 00:00:00
    7/12/72
    1972-07-12 00:00:00
    3/9/72
    1972-03-09 00:00:00
    5/4/72
    1972-05-04 00:00:00
    6/14/72
    1972-06-14 00:00:00
    8/31/72
    1972-08-31 00:00:00
    12/1/72
    1972-12-01 00:00:00
    12/17/72
    1972-12-17 00:00:00
    4/12/72
    1972-04-12 00:00:00
    8/23/72
    1972-08-23 00:00:00
    12/19/72
    1972-12-19 00:00:00
    9/8/72
    1972-09-08 00:00:00
    12/18/72
    1972-12-18 00:00:00
    5/23/72
    1972-05-23 00:00:00
    11/17/72
    1972-11-17 00:00:00
    8/25/72
    1972-08-25 00:00:00
    2/15/72
    1972-02-15 00:00:00
    5/5/72
    1972-05-05 00:00:00
    10/4/72
    1972-10-04 00:00:00
    7/1/72
    1972-07-01 00:00:00
    7/26/72
    1972-07-26 00:00:00
    1/1/80
    1980-01-01 00:00:00
    5/22/80
    1980-05-22 00:00:00
    7/2/80
    1980-07-02 00:00:00
    6/17/80
    1980-06-17 00:00:00
    7/5/80
    1980-07-05 00:00:00
    11/14/80
    1980-11-14 00:00:00
    12/4/80
    1980-12-04 00:00:00
    5/9/80
    1980-05-09 00:00:00
    5/16/80
    1980-05-16 00:00:00
    9/1/80
    1980-09-01 00:00:00
    10/3/80
    1980-10-03 00:00:00
    7/18/80
    1980-07-18 00:00:00
    8/15/80
    1980-08-15 00:00:00
    10/6/80
    1980-10-06 00:00:00
    9/10/80
    1980-09-10 00:00:00
    2/8/80
    1980-02-08 00:00:00
    11/25/80
    1980-11-25 00:00:00
    7/25/80
    1980-07-25 00:00:00
    6/25/80
    1980-06-25 00:00:00
    7/11/80
    1980-07-11 00:00:00
    2/7/80
    1980-02-07 00:00:00
    12/12/80
    1980-12-12 00:00:00
    6/11/80
    1980-06-11 00:00:00
    11/19/80
    1980-11-19 00:00:00
    10/3/80
    1980-10-03 00:00:00
    9/10/80
    1980-09-10 00:00:00
    1/31/80
    1980-01-31 00:00:00
    1/1/80
    1980-01-01 00:00:00
    10/2/80
    1980-10-02 00:00:00
    12/12/80
    1980-12-12 00:00:00
    6/25/80
    1980-06-25 00:00:00
    12/17/80
    1980-12-17 00:00:00
    11/1/80
    1980-11-01 00:00:00
    12/17/80
    1980-12-17 00:00:00
    12/25/80
    1980-12-25 00:00:00
    6/6/80
    1980-06-06 00:00:00
    9/10/80
    1980-09-10 00:00:00
    9/19/80
    1980-09-19 00:00:00
    5/28/80
    1980-05-28 00:00:00
    2/8/80
    1980-02-08 00:00:00
    9/23/80
    1980-09-23 00:00:00
    7/2/80
    1980-07-02 00:00:00
    12/18/80
    1980-12-18 00:00:00
    9/2/80
    1980-09-02 00:00:00
    2/9/80
    1980-02-09 00:00:00
    9/19/80
    1980-09-19 00:00:00
    9/26/80
    1980-09-26 00:00:00
    3/28/80
    1980-03-28 00:00:00
    7/11/80
    1980-07-11 00:00:00
    3/2/80
    1980-03-02 00:00:00
    2/15/80
    1980-02-15 00:00:00
    11/11/80
    1980-11-11 00:00:00
    7/18/80
    1980-07-18 00:00:00
    9/15/80
    1980-09-15 00:00:00
    5/15/80
    1980-05-15 00:00:00
    5/9/80
    1980-05-09 00:00:00
    6/20/80
    1980-06-20 00:00:00
    2/29/80
    1980-02-29 00:00:00
    9/8/80
    1980-09-08 00:00:00
    12/19/80
    1980-12-19 00:00:00
    3/7/80
    1980-03-07 00:00:00
    8/8/80
    1980-08-08 00:00:00
    5/12/80
    1980-05-12 00:00:00
    12/26/80
    1980-12-26 00:00:00
    5/1/80
    1980-05-01 00:00:00
    4/17/80
    1980-04-17 00:00:00
    8/1/80
    1980-08-01 00:00:00
    4/25/80
    1980-04-25 00:00:00
    3/7/80
    1980-03-07 00:00:00
    8/14/80
    1980-08-14 00:00:00
    2/29/80
    1980-02-29 00:00:00
    3/21/80
    1980-03-21 00:00:00
    3/27/80
    1980-03-27 00:00:00
    7/31/80
    1980-07-31 00:00:00
    10/1/80
    1980-10-01 00:00:00
    9/26/80
    1980-09-26 00:00:00
    6/27/80
    1980-06-27 00:00:00
    8/14/80
    1980-08-14 00:00:00
    5/19/07
    2007-05-19 00:00:00
    6/28/07
    2007-06-28 00:00:00
    6/22/07
    2007-06-22 00:00:00
    8/3/07
    2007-08-03 00:00:00
    11/8/07
    2007-11-08 00:00:00
    12/14/07
    2007-12-14 00:00:00
    12/13/07
    2007-12-13 00:00:00
    5/1/07
    2007-05-01 00:00:00
    11/21/07
    2007-11-21 00:00:00
    8/9/07
    2007-08-09 00:00:00
    5/21/07
    2007-05-21 00:00:00
    6/20/07
    2007-06-20 00:00:00
    8/8/07
    2007-08-08 00:00:00
    9/11/07
    2007-09-11 00:00:00
    3/7/07
    2007-03-07 00:00:00
    2/16/07
    2007-02-16 00:00:00
    12/5/07
    2007-12-05 00:00:00
    2/16/07
    2007-02-16 00:00:00
    3/22/07
    2007-03-22 00:00:00
    8/29/07
    2007-08-29 00:00:00
    12/28/07
    2007-12-28 00:00:00
    10/2/07
    2007-10-02 00:00:00
    11/2/07
    2007-11-02 00:00:00
    2/14/07
    2007-02-14 00:00:00
    5/17/07
    2007-05-17 00:00:00
    12/20/07
    2007-12-20 00:00:00
    8/17/07
    2007-08-17 00:00:00
    12/13/07
    2007-12-13 00:00:00
    10/28/07
    2007-10-28 00:00:00
    3/23/07
    2007-03-23 00:00:00
    11/15/07
    2007-11-15 00:00:00
    6/1/07
    2007-06-01 00:00:00
    2/23/07
    2007-02-23 00:00:00
    4/6/07
    2007-04-06 00:00:00
    6/13/07
    2007-06-13 00:00:00
    11/20/07
    2007-11-20 00:00:00
    3/2/07
    2007-03-02 00:00:00
    5/29/07
    2007-05-29 00:00:00
    6/7/07
    2007-06-07 00:00:00
    3/22/07
    2007-03-22 00:00:00
    7/25/07
    2007-07-25 00:00:00
    12/19/07
    2007-12-19 00:00:00
    4/26/07
    2007-04-26 00:00:00
    8/22/07
    2007-08-22 00:00:00
    12/25/07
    2007-12-25 00:00:00
    4/5/07
    2007-04-05 00:00:00
    3/22/07
    2007-03-22 00:00:00
    6/1/07
    2007-06-01 00:00:00
    10/17/07
    2007-10-17 00:00:00
    12/4/07
    2007-12-04 00:00:00
    3/23/07
    2007-03-23 00:00:00
    4/20/07
    2007-04-20 00:00:00
    2/14/07
    2007-02-14 00:00:00
    9/14/07
    2007-09-14 00:00:00
    4/12/07
    2007-04-12 00:00:00
    10/22/07
    2007-10-22 00:00:00
    9/6/07
    2007-09-06 00:00:00
    11/21/07
    2007-11-21 00:00:00
    11/28/07
    2007-11-28 00:00:00
    8/24/07
    2007-08-24 00:00:00
    9/14/07
    2007-09-14 00:00:00
    6/9/07
    2007-06-09 00:00:00
    7/12/07
    2007-07-12 00:00:00
    4/24/07
    2007-04-24 00:00:00
    2/6/07
    2007-02-06 00:00:00
    2/8/07
    2007-02-08 00:00:00
    2/12/07
    2007-02-12 00:00:00
    10/25/07
    2007-10-25 00:00:00
    11/3/07
    2007-11-03 00:00:00
    4/4/07
    2007-04-04 00:00:00
    7/13/07
    2007-07-13 00:00:00
    3/22/07
    2007-03-22 00:00:00
    8/17/07
    2007-08-17 00:00:00
    7/26/07
    2007-07-26 00:00:00
    9/14/07
    2007-09-14 00:00:00
    11/5/07
    2007-11-05 00:00:00
    7/25/07
    2007-07-25 00:00:00
    12/10/07
    2007-12-10 00:00:00
    10/5/07
    2007-10-05 00:00:00
    6/8/07
    2007-06-08 00:00:00
    7/12/07
    2007-07-12 00:00:00
    11/21/07
    2007-11-21 00:00:00
    9/28/07
    2007-09-28 00:00:00
    3/2/07
    2007-03-02 00:00:00
    4/6/07
    2007-04-06 00:00:00
    9/6/07
    2007-09-06 00:00:00
    2/2/07
    2007-02-02 00:00:00
    6/27/07
    2007-06-27 00:00:00
    12/25/07
    2007-12-25 00:00:00
    6/8/07
    2007-06-08 00:00:00
    2/9/07
    2007-02-09 00:00:00
    3/30/07
    2007-03-30 00:00:00
    1/19/07
    2007-01-19 00:00:00
    7/19/07
    2007-07-19 00:00:00
    8/25/07
    2007-08-25 00:00:00
    7/27/07
    2007-07-27 00:00:00
    8/31/07
    2007-08-31 00:00:00
    3/2/07
    2007-03-02 00:00:00
    9/23/07
    2007-09-23 00:00:00
    10/20/07
    2007-10-20 00:00:00
    11/29/07
    2007-11-29 00:00:00
    9/9/07
    2007-09-09 00:00:00
    3/25/07
    2007-03-25 00:00:00
    9/28/07
    2007-09-28 00:00:00
    8/17/07
    2007-08-17 00:00:00
    11/14/07
    2007-11-14 00:00:00
    9/7/07
    2007-09-07 00:00:00
    9/26/07
    2007-09-26 00:00:00
    1/4/07
    2007-01-04 00:00:00
    8/8/07
    2007-08-08 00:00:00
    3/9/07
    2007-03-09 00:00:00
    5/25/07
    2007-05-25 00:00:00
    6/10/07
    2007-06-10 00:00:00
    8/3/07
    2007-08-03 00:00:00
    11/9/07
    2007-11-09 00:00:00
    9/20/07
    2007-09-20 00:00:00
    9/8/07
    2007-09-08 00:00:00
    9/2/07
    2007-09-02 00:00:00
    1/1/07
    2007-01-01 00:00:00
    2/9/07
    2007-02-09 00:00:00
    4/5/07
    2007-04-05 00:00:00
    9/9/07
    2007-09-09 00:00:00
    4/12/07
    2007-04-12 00:00:00
    9/21/07
    2007-09-21 00:00:00
    5/11/07
    2007-05-11 00:00:00
    2/8/07
    2007-02-08 00:00:00
    2/9/07
    2007-02-09 00:00:00
    9/7/07
    2007-09-07 00:00:00
    9/1/07
    2007-09-01 00:00:00
    9/8/07
    2007-09-08 00:00:00
    6/6/07
    2007-06-06 00:00:00
    11/28/07
    2007-11-28 00:00:00
    5/1/07
    2007-05-01 00:00:00
    11/2/07
    2007-11-02 00:00:00
    8/31/07
    2007-08-31 00:00:00
    9/26/07
    2007-09-26 00:00:00
    9/21/07
    2007-09-21 00:00:00
    11/24/07
    2007-11-24 00:00:00
    1/21/07
    2007-01-21 00:00:00
    1/25/07
    2007-01-25 00:00:00
    1/1/07
    2007-01-01 00:00:00
    4/16/07
    2007-04-16 00:00:00
    2/20/07
    2007-02-20 00:00:00
    10/5/07
    2007-10-05 00:00:00
    9/6/07
    2007-09-06 00:00:00
    8/24/07
    2007-08-24 00:00:00
    2/2/07
    2007-02-02 00:00:00
    12/27/07
    2007-12-27 00:00:00
    10/5/07
    2007-10-05 00:00:00
    12/18/07
    2007-12-18 00:00:00
    8/14/07
    2007-08-14 00:00:00
    1/5/07
    2007-01-05 00:00:00
    9/3/07
    2007-09-03 00:00:00
    4/27/07
    2007-04-27 00:00:00
    9/14/07
    2007-09-14 00:00:00
    9/14/07
    2007-09-14 00:00:00
    8/3/07
    2007-08-03 00:00:00
    9/3/07
    2007-09-03 00:00:00
    9/6/07
    2007-09-06 00:00:00
    5/25/07
    2007-05-25 00:00:00
    10/14/07
    2007-10-14 00:00:00
    4/20/07
    2007-04-20 00:00:00
    10/26/07
    2007-10-26 00:00:00
    1/31/07
    2007-01-31 00:00:00
    9/21/07
    2007-09-21 00:00:00
    4/27/07
    2007-04-27 00:00:00
    6/18/07
    2007-06-18 00:00:00
    9/9/07
    2007-09-09 00:00:00
    6/22/07
    2007-06-22 00:00:00
    2/9/07
    2007-02-09 00:00:00
    7/4/07
    2007-07-04 00:00:00
    5/11/07
    2007-05-11 00:00:00
    7/27/07
    2007-07-27 00:00:00
    9/13/07
    2007-09-13 00:00:00
    9/28/07
    2007-09-28 00:00:00
    10/12/07
    2007-10-12 00:00:00
    9/23/07
    2007-09-23 00:00:00
    9/30/07
    2007-09-30 00:00:00
    12/25/07
    2007-12-25 00:00:00
    8/31/07
    2007-08-31 00:00:00
    7/23/07
    2007-07-23 00:00:00
    1/26/07
    2007-01-26 00:00:00
    8/17/07
    2007-08-17 00:00:00
    3/23/07
    2007-03-23 00:00:00
    10/3/07
    2007-10-03 00:00:00
    11/8/07
    2007-11-08 00:00:00
    6/1/07
    2007-06-01 00:00:00
    7/27/07
    2007-07-27 00:00:00
    1/25/07
    2007-01-25 00:00:00
    3/9/07
    2007-03-09 00:00:00
    3/7/07
    2007-03-07 00:00:00
    4/27/07
    2007-04-27 00:00:00
    9/14/07
    2007-09-14 00:00:00
    10/20/07
    2007-10-20 00:00:00
    1/19/07
    2007-01-19 00:00:00
    7/17/07
    2007-07-17 00:00:00
    2/7/07
    2007-02-07 00:00:00
    12/7/07
    2007-12-07 00:00:00
    5/18/07
    2007-05-18 00:00:00
    1/11/07
    2007-01-11 00:00:00
    10/19/07
    2007-10-19 00:00:00
    1/19/07
    2007-01-19 00:00:00
    3/10/07
    2007-03-10 00:00:00
    4/27/07
    2007-04-27 00:00:00
    9/14/07
    2007-09-14 00:00:00
    9/5/07
    2007-09-05 00:00:00
    6/16/07
    2007-06-16 00:00:00
    4/25/07
    2007-04-25 00:00:00
    12/21/07
    2007-12-21 00:00:00
    10/24/07
    2007-10-24 00:00:00
    10/28/07
    2007-10-28 00:00:00
    9/7/07
    2007-09-07 00:00:00
    1/1/07
    2007-01-01 00:00:00
    1/1/07
    2007-01-01 00:00:00
    9/26/07
    2007-09-26 00:00:00
    9/13/07
    2007-09-13 00:00:00
    5/19/07
    2007-05-19 00:00:00
    9/11/07
    2007-09-11 00:00:00
    9/12/07
    2007-09-12 00:00:00
    4/4/07
    2007-04-04 00:00:00
    10/12/07
    2007-10-12 00:00:00
    4/17/07
    2007-04-17 00:00:00
    8/3/07
    2007-08-03 00:00:00
    2/9/07
    2007-02-09 00:00:00
    2/6/07
    2007-02-06 00:00:00
    3/23/07
    2007-03-23 00:00:00
    10/4/07
    2007-10-04 00:00:00
    9/21/07
    2007-09-21 00:00:00
    4/13/07
    2007-04-13 00:00:00
    9/11/07
    2007-09-11 00:00:00
    10/1/07
    2007-10-01 00:00:00
    9/6/07
    2007-09-06 00:00:00
    4/13/07
    2007-04-13 00:00:00
    6/1/07
    2007-06-01 00:00:00
    6/14/07
    2007-06-14 00:00:00
    5/21/07
    2007-05-21 00:00:00
    11/6/07
    2007-11-06 00:00:00
    5/27/07
    2007-05-27 00:00:00
    5/17/07
    2007-05-17 00:00:00
    9/7/07
    2007-09-07 00:00:00
    3/18/07
    2007-03-18 00:00:00
    9/11/07
    2007-09-11 00:00:00
    1/24/07
    2007-01-24 00:00:00
    4/22/07
    2007-04-22 00:00:00
    3/4/07
    2007-03-04 00:00:00
    9/17/07
    2007-09-17 00:00:00
    9/14/07
    2007-09-14 00:00:00
    1/20/07
    2007-01-20 00:00:00
    4/9/07
    2007-04-09 00:00:00
    1/25/07
    2007-01-25 00:00:00
    1/25/07
    2007-01-25 00:00:00
    4/19/07
    2007-04-19 00:00:00
    1/30/07
    2007-01-30 00:00:00
    7/1/07
    2007-07-01 00:00:00
    1/5/07
    2007-01-05 00:00:00
    1/19/07
    2007-01-19 00:00:00
    11/27/07
    2007-11-27 00:00:00
    1/23/07
    2007-01-23 00:00:00
    7/31/07
    2007-07-31 00:00:00
    11/27/07
    2007-11-27 00:00:00
    10/7/07
    2007-10-07 00:00:00
    12/5/07
    2007-12-05 00:00:00
    4/29/07
    2007-04-29 00:00:00
    9/1/07
    2007-09-01 00:00:00
    12/21/07
    2007-12-21 00:00:00
    8/28/07
    2007-08-28 00:00:00
    5/16/07
    2007-05-16 00:00:00
    2/9/07
    2007-02-09 00:00:00
    1/1/07
    2007-01-01 00:00:00
    1/1/07
    2007-01-01 00:00:00
    11/14/07
    2007-11-14 00:00:00
    9/18/07
    2007-09-18 00:00:00
    5/18/07
    2007-05-18 00:00:00
    4/1/07
    2007-04-01 00:00:00
    6/9/07
    2007-06-09 00:00:00
    11/15/07
    2007-11-15 00:00:00
    4/27/07
    2007-04-27 00:00:00
    9/7/07
    2007-09-07 00:00:00
    9/29/07
    2007-09-29 00:00:00
    1/1/07
    2007-01-01 00:00:00
    5/16/07
    2007-05-16 00:00:00
    12/17/07
    2007-12-17 00:00:00
    2/9/07
    2007-02-09 00:00:00
    2/16/07
    2007-02-16 00:00:00
    5/18/07
    2007-05-18 00:00:00
    6/15/07
    2007-06-15 00:00:00
    6/6/07
    2007-06-06 00:00:00
    11/5/07
    2007-11-05 00:00:00
    3/16/07
    2007-03-16 00:00:00
    8/11/07
    2007-08-11 00:00:00
    12/21/07
    2007-12-21 00:00:00
    10/4/07
    2007-10-04 00:00:00
    5/30/07
    2007-05-30 00:00:00
    9/18/07
    2007-09-18 00:00:00
    1/21/07
    2007-01-21 00:00:00
    9/9/07
    2007-09-09 00:00:00
    5/18/07
    2007-05-18 00:00:00
    8/16/07
    2007-08-16 00:00:00
    2/22/07
    2007-02-22 00:00:00
    9/18/07
    2007-09-18 00:00:00
    3/11/07
    2007-03-11 00:00:00
    7/10/07
    2007-07-10 00:00:00
    1/1/07
    2007-01-01 00:00:00
    6/15/07
    2007-06-15 00:00:00
    4/28/07
    2007-04-28 00:00:00
    12/21/07
    2007-12-21 00:00:00
    9/26/07
    2007-09-26 00:00:00
    5/11/07
    2007-05-11 00:00:00
    8/3/07
    2007-08-03 00:00:00
    4/28/07
    2007-04-28 00:00:00
    9/7/07
    2007-09-07 00:00:00
    1/19/07
    2007-01-19 00:00:00
    10/19/07
    2007-10-19 00:00:00
    1/24/07
    2007-01-24 00:00:00
    1/21/07
    2007-01-21 00:00:00
    5/21/07
    2007-05-21 00:00:00
    4/27/07
    2007-04-27 00:00:00
    1/23/07
    2007-01-23 00:00:00
    9/12/07
    2007-09-12 00:00:00
    1/31/07
    2007-01-31 00:00:00
    1/22/07
    2007-01-22 00:00:00
    3/16/07
    2007-03-16 00:00:00
    9/15/07
    2007-09-15 00:00:00
    12/17/07
    2007-12-17 00:00:00
    8/17/07
    2007-08-17 00:00:00
    1/12/07
    2007-01-12 00:00:00
    1/31/07
    2007-01-31 00:00:00
    9/20/07
    2007-09-20 00:00:00
    4/27/07
    2007-04-27 00:00:00
    5/15/07
    2007-05-15 00:00:00
    2/6/07
    2007-02-06 00:00:00
    4/30/07
    2007-04-30 00:00:00
    11/23/07
    2007-11-23 00:00:00
    11/9/07
    2007-11-09 00:00:00
    1/1/07
    2007-01-01 00:00:00
    9/4/07
    2007-09-04 00:00:00
    6/1/07
    2007-06-01 00:00:00
    4/27/07
    2007-04-27 00:00:00
    9/7/07
    2007-09-07 00:00:00
    5/22/07
    2007-05-22 00:00:00
    2/11/07
    2007-02-11 00:00:00
    8/29/07
    2007-08-29 00:00:00
    11/7/07
    2007-11-07 00:00:00
    10/9/07
    2007-10-09 00:00:00
    8/4/07
    2007-08-04 00:00:00
    1/1/07
    2007-01-01 00:00:00
    10/12/07
    2007-10-12 00:00:00
    8/16/07
    2007-08-16 00:00:00
    6/1/07
    2007-06-01 00:00:00
    1/21/07
    2007-01-21 00:00:00
    3/28/07
    2007-03-28 00:00:00
    8/4/07
    2007-08-04 00:00:00
    10/9/07
    2007-10-09 00:00:00
    10/9/07
    2007-10-09 00:00:00
    9/22/07
    2007-09-22 00:00:00
    1/1/07
    2007-01-01 00:00:00
    9/4/07
    2007-09-04 00:00:00
    12/18/07
    2007-12-18 00:00:00
    12/5/07
    2007-12-05 00:00:00
    7/15/07
    2007-07-15 00:00:00
    1/12/07
    2007-01-12 00:00:00
    12/26/07
    2007-12-26 00:00:00
    6/5/07
    2007-06-05 00:00:00
    2/17/07
    2007-02-17 00:00:00
    5/19/07
    2007-05-19 00:00:00
    2/13/07
    2007-02-13 00:00:00
    5/21/07
    2007-05-21 00:00:00
    10/14/07
    2007-10-14 00:00:00
    9/14/07
    2007-09-14 00:00:00
    4/17/07
    2007-04-17 00:00:00
    3/3/07
    2007-03-03 00:00:00
    9/12/07
    2007-09-12 00:00:00
    9/27/07
    2007-09-27 00:00:00
    5/16/07
    2007-05-16 00:00:00
    2/14/07
    2007-02-14 00:00:00
    4/27/07
    2007-04-27 00:00:00
    6/10/07
    2007-06-10 00:00:00
    7/13/07
    2007-07-13 00:00:00
    1/22/07
    2007-01-22 00:00:00
    6/17/07
    2007-06-17 00:00:00
    7/6/07
    2007-07-06 00:00:00
    11/1/07
    2007-11-01 00:00:00
    11/27/07
    2007-11-27 00:00:00
    6/15/07
    2007-06-15 00:00:00
    11/21/07
    2007-11-21 00:00:00
    6/26/07
    2007-06-26 00:00:00
    1/1/07
    2007-01-01 00:00:00
    10/13/07
    2007-10-13 00:00:00
    5/22/07
    2007-05-22 00:00:00
    10/1/07
    2007-10-01 00:00:00
    5/17/07
    2007-05-17 00:00:00
    3/1/07
    2007-03-01 00:00:00
    4/6/07
    2007-04-06 00:00:00
    1/17/07
    2007-01-17 00:00:00
    4/27/07
    2007-04-27 00:00:00
    1/22/07
    2007-01-22 00:00:00
    11/27/07
    2007-11-27 00:00:00
    1/1/07
    2007-01-01 00:00:00
    2/23/07
    2007-02-23 00:00:00
    12/26/07
    2007-12-26 00:00:00
    6/8/07
    2007-06-08 00:00:00
    10/1/07
    2007-10-01 00:00:00
    9/7/07
    2007-09-07 00:00:00
    6/20/07
    2007-06-20 00:00:00
    1/1/07
    2007-01-01 00:00:00
    6/1/07
    2007-06-01 00:00:00
    1/1/07
    2007-01-01 00:00:00
    9/1/07
    2007-09-01 00:00:00
    12/9/07
    2007-12-09 00:00:00
    12/20/07
    2007-12-20 00:00:00
    1/5/07
    2007-01-05 00:00:00
    3/6/07
    2007-03-06 00:00:00
    9/6/07
    2007-09-06 00:00:00
    2/28/07
    2007-02-28 00:00:00
    8/24/07
    2007-08-24 00:00:00
    11/23/07
    2007-11-23 00:00:00
    10/12/07
    2007-10-12 00:00:00
    5/9/07
    2007-05-09 00:00:00
    5/18/07
    2007-05-18 00:00:00
    4/4/07
    2007-04-04 00:00:00
    1/1/07
    2007-01-01 00:00:00
    1/1/07
    2007-01-01 00:00:00
    1/1/07
    2007-01-01 00:00:00
    8/31/07
    2007-08-31 00:00:00
    3/27/07
    2007-03-27 00:00:00
    2/7/07
    2007-02-07 00:00:00
    11/11/07
    2007-11-11 00:00:00
    10/16/07
    2007-10-16 00:00:00
    3/9/07
    2007-03-09 00:00:00
    1/19/07
    2007-01-19 00:00:00
    11/10/07
    2007-11-10 00:00:00
    10/12/07
    2007-10-12 00:00:00
    7/27/07
    2007-07-27 00:00:00
    7/8/07
    2007-07-08 00:00:00
    3/2/07
    2007-03-02 00:00:00
    5/8/07
    2007-05-08 00:00:00
    11/21/07
    2007-11-21 00:00:00
    10/5/07
    2007-10-05 00:00:00
    5/25/79
    1979-05-25 00:00:00
    8/15/79
    1979-08-15 00:00:00
    4/12/79
    1979-04-12 00:00:00
    12/6/79
    1979-12-06 00:00:00
    6/26/79
    1979-06-26 00:00:00
    8/16/79
    1979-08-16 00:00:00
    5/11/79
    1979-05-11 00:00:00
    2/9/79
    1979-02-09 00:00:00
    6/15/79
    1979-06-15 00:00:00
    1/21/79
    1979-01-21 00:00:00
    4/25/79
    1979-04-25 00:00:00
    3/28/79
    1979-03-28 00:00:00
    3/14/79
    1979-03-14 00:00:00
    7/27/79
    1979-07-27 00:00:00
    6/22/79
    1979-06-22 00:00:00
    12/20/79
    1979-12-20 00:00:00
    4/11/79
    1979-04-11 00:00:00
    12/18/79
    1979-12-18 00:00:00
    12/19/79
    1979-12-19 00:00:00
    6/28/79
    1979-06-28 00:00:00
    7/26/79
    1979-07-26 00:00:00
    11/17/79
    1979-11-17 00:00:00
    8/17/79
    1979-08-17 00:00:00
    12/19/79
    1979-12-19 00:00:00
    3/16/79
    1979-03-16 00:00:00
    12/14/79
    1979-12-14 00:00:00
    12/14/79
    1979-12-14 00:00:00
    10/19/79
    1979-10-19 00:00:00
    8/31/79
    1979-08-31 00:00:00
    5/24/79
    1979-05-24 00:00:00
    9/14/79
    1979-09-14 00:00:00
    10/6/79
    1979-10-06 00:00:00
    7/6/79
    1979-07-06 00:00:00
    1/25/79
    1979-01-25 00:00:00
    10/4/79
    1979-10-04 00:00:00
    10/17/79
    1979-10-17 00:00:00
    12/21/79
    1979-12-21 00:00:00
    9/30/79
    1979-09-30 00:00:00
    11/14/79
    1979-11-14 00:00:00
    6/27/79
    1979-06-27 00:00:00
    10/19/79
    1979-10-19 00:00:00
    9/12/79
    1979-09-12 00:00:00
    2/9/79
    1979-02-09 00:00:00
    5/18/79
    1979-05-18 00:00:00
    4/27/79
    1979-04-27 00:00:00
    5/25/79
    1979-05-25 00:00:00
    3/22/79
    1979-03-22 00:00:00
    1/1/79
    1979-01-01 00:00:00
    10/26/79
    1979-10-26 00:00:00
    9/9/79
    1979-09-09 00:00:00
    7/13/79
    1979-07-13 00:00:00
    11/2/79
    1979-11-02 00:00:00
    1/25/79
    1979-01-25 00:00:00
    8/24/79
    1979-08-24 00:00:00
    3/2/79
    1979-03-02 00:00:00
    7/20/79
    1979-07-20 00:00:00
    7/10/79
    1979-07-10 00:00:00
    10/26/84
    1984-10-26 00:00:00
    5/23/84
    1984-05-23 00:00:00
    6/7/84
    1984-06-07 00:00:00
    6/7/84
    1984-06-07 00:00:00
    4/5/84
    1984-04-05 00:00:00
    11/30/84
    1984-11-30 00:00:00
    2/17/84
    1984-02-17 00:00:00
    10/26/84
    1984-10-26 00:00:00
    6/22/84
    1984-06-22 00:00:00
    11/15/84
    1984-11-15 00:00:00
    6/29/84
    1984-06-29 00:00:00
    12/6/84
    1984-12-06 00:00:00
    12/14/84
    1984-12-14 00:00:00
    3/30/84
    1984-03-30 00:00:00
    3/22/84
    1984-03-22 00:00:00
    8/10/84
    1984-08-10 00:00:00
    5/31/84
    1984-05-31 00:00:00
    6/4/84
    1984-06-04 00:00:00
    2/17/84
    1984-02-17 00:00:00
    4/13/84
    1984-04-13 00:00:00
    2/20/84
    1984-02-20 00:00:00
    10/10/84
    1984-10-10 00:00:00
    11/20/84
    1984-11-20 00:00:00
    12/17/84
    1984-12-17 00:00:00
    12/14/84
    1984-12-14 00:00:00
    6/29/84
    1984-06-29 00:00:00
    5/4/84
    1984-05-04 00:00:00
    8/17/84
    1984-08-17 00:00:00
    7/27/84
    1984-07-27 00:00:00
    5/4/84
    1984-05-04 00:00:00
    11/2/84
    1984-11-02 00:00:00
    11/16/84
    1984-11-16 00:00:00
    8/15/84
    1984-08-15 00:00:00
    11/25/84
    1984-11-25 00:00:00
    7/13/84
    1984-07-13 00:00:00
    12/17/84
    1984-12-17 00:00:00
    8/17/84
    1984-08-17 00:00:00
    9/7/84
    1984-09-07 00:00:00
    3/9/84
    1984-03-09 00:00:00
    5/19/84
    1984-05-19 00:00:00
    10/26/84
    1984-10-26 00:00:00
    8/15/84
    1984-08-15 00:00:00
    12/5/84
    1984-12-05 00:00:00
    8/15/84
    1984-08-15 00:00:00
    5/4/84
    1984-05-04 00:00:00
    12/14/84
    1984-12-14 00:00:00
    11/7/84
    1984-11-07 00:00:00
    7/13/84
    1984-07-13 00:00:00
    6/8/84
    1984-06-08 00:00:00
    3/2/84
    1984-03-02 00:00:00
    2/17/84
    1984-02-17 00:00:00
    10/1/84
    1984-10-01 00:00:00
    3/30/84
    1984-03-30 00:00:00
    12/14/84
    1984-12-14 00:00:00
    7/27/84
    1984-07-27 00:00:00
    11/21/84
    1984-11-21 00:00:00
    11/16/84
    1984-11-16 00:00:00
    9/7/84
    1984-09-07 00:00:00
    5/11/84
    1984-05-11 00:00:00
    7/20/84
    1984-07-20 00:00:00
    3/9/84
    1984-03-09 00:00:00
    12/14/84
    1984-12-14 00:00:00
    5/1/84
    1984-05-01 00:00:00
    6/1/84
    1984-06-01 00:00:00
    12/14/84
    1984-12-14 00:00:00
    6/1/84
    1984-06-01 00:00:00
    11/16/84
    1984-11-16 00:00:00
    8/30/84
    1984-08-30 00:00:00
    8/10/84
    1984-08-10 00:00:00
    12/21/84
    1984-12-21 00:00:00
    6/29/84
    1984-06-29 00:00:00
    1/10/84
    1984-01-10 00:00:00
    6/22/84
    1984-06-22 00:00:00
    10/19/84
    1984-10-19 00:00:00
    12/21/84
    1984-12-21 00:00:00
    11/9/84
    1984-11-09 00:00:00
    4/13/84
    1984-04-13 00:00:00
    1/13/84
    1984-01-13 00:00:00
    8/3/84
    1984-08-03 00:00:00
    3/16/84
    1984-03-16 00:00:00
    9/14/84
    1984-09-14 00:00:00
    5/11/84
    1984-05-11 00:00:00
    9/23/84
    1984-09-23 00:00:00
    8/31/84
    1984-08-31 00:00:00
    8/3/84
    1984-08-03 00:00:00
    12/14/84
    1984-12-14 00:00:00
    8/31/84
    1984-08-31 00:00:00
    11/8/84
    1984-11-08 00:00:00
    6/1/84
    1984-06-01 00:00:00
    7/20/84
    1984-07-20 00:00:00
    6/22/84
    1984-06-22 00:00:00
    1/22/84
    1984-01-22 00:00:00
    3/16/84
    1984-03-16 00:00:00
    4/19/84
    1984-04-19 00:00:00
    1/27/84
    1984-01-27 00:00:00
    5/18/84
    1984-05-18 00:00:00
    1/27/84
    1984-01-27 00:00:00
    2/7/84
    1984-02-07 00:00:00
    9/14/84
    1984-09-14 00:00:00
    10/7/84
    1984-10-07 00:00:00
    6/8/84
    1984-06-08 00:00:00
    7/20/84
    1984-07-20 00:00:00
    9/12/84
    1984-09-12 00:00:00
    6/12/84
    1984-06-12 00:00:00
    9/15/84
    1984-09-15 00:00:00
    5/23/83
    1983-05-23 00:00:00
    12/8/83
    1983-12-08 00:00:00
    10/7/83
    1983-10-07 00:00:00
    6/5/83
    1983-06-05 00:00:00
    11/18/83
    1983-11-18 00:00:00
    6/17/83
    1983-06-17 00:00:00
    6/7/83
    1983-06-07 00:00:00
    6/3/83
    1983-06-03 00:00:00
    7/28/83
    1983-07-28 00:00:00
    3/16/83
    1983-03-16 00:00:00
    3/31/83
    1983-03-31 00:00:00
    11/20/83
    1983-11-20 00:00:00
    7/29/83
    1983-07-29 00:00:00
    12/8/83
    1983-12-08 00:00:00
    8/5/83
    1983-08-05 00:00:00
    10/21/83
    1983-10-21 00:00:00
    5/13/83
    1983-05-13 00:00:00
    6/24/83
    1983-06-24 00:00:00
    10/19/83
    1983-10-19 00:00:00
    5/6/83
    1983-05-06 00:00:00
    4/29/83
    1983-04-29 00:00:00
    12/9/83
    1983-12-09 00:00:00
    3/25/83
    1983-03-25 00:00:00
    6/24/83
    1983-06-24 00:00:00
    10/21/83
    1983-10-21 00:00:00
    6/10/83
    1983-06-10 00:00:00
    11/20/83
    1983-11-20 00:00:00
    8/10/83
    1983-08-10 00:00:00
    11/18/83
    1983-11-18 00:00:00
    2/5/83
    1983-02-05 00:00:00
    10/21/83
    1983-10-21 00:00:00
    4/8/83
    1983-04-08 00:00:00
    10/21/83
    1983-10-21 00:00:00
    7/15/83
    1983-07-15 00:00:00
    12/15/83
    1983-12-15 00:00:00
    4/1/83
    1983-04-01 00:00:00
    4/14/83
    1983-04-14 00:00:00
    12/14/83
    1983-12-14 00:00:00
    8/12/83
    1983-08-12 00:00:00
    8/12/83
    1983-08-12 00:00:00
    11/18/83
    1983-11-18 00:00:00
    6/24/83
    1983-06-24 00:00:00
    12/16/83
    1983-12-16 00:00:00
    9/21/83
    1983-09-21 00:00:00
    7/29/83
    1983-07-29 00:00:00
    10/21/83
    1983-10-21 00:00:00
    7/11/83
    1983-07-11 00:00:00
    4/14/83
    1983-04-14 00:00:00
    9/9/83
    1983-09-09 00:00:00
    6/3/83
    1983-06-03 00:00:00
    4/8/83
    1983-04-08 00:00:00
    5/1/83
    1983-05-01 00:00:00
    2/4/83
    1983-02-04 00:00:00
    1/1/83
    1983-01-01 00:00:00
    10/14/83
    1983-10-14 00:00:00
    9/2/83
    1983-09-02 00:00:00
    12/2/83
    1983-12-02 00:00:00
    2/17/83
    1983-02-17 00:00:00
    1/1/83
    1983-01-01 00:00:00
    3/25/83
    1983-03-25 00:00:00
    4/15/83
    1983-04-15 00:00:00
    8/5/83
    1983-08-05 00:00:00
    3/25/83
    1983-03-25 00:00:00
    9/30/83
    1983-09-30 00:00:00
    7/22/83
    1983-07-22 00:00:00
    8/26/83
    1983-08-26 00:00:00
    10/27/83
    1983-10-27 00:00:00
    12/16/83
    1983-12-16 00:00:00
    9/23/83
    1983-09-23 00:00:00
    7/22/83
    1983-07-22 00:00:00
    12/29/83
    1983-12-29 00:00:00
    4/29/83
    1983-04-29 00:00:00
    11/18/83
    1983-11-18 00:00:00
    10/18/83
    1983-10-18 00:00:00
    1/21/83
    1983-01-21 00:00:00
    8/19/83
    1983-08-19 00:00:00
    8/12/83
    1983-08-12 00:00:00
    12/16/83
    1983-12-16 00:00:00
    8/17/83
    1983-08-17 00:00:00
    5/20/83
    1983-05-20 00:00:00
    9/22/95
    1995-09-22 00:00:00
    10/30/95
    1995-10-30 00:00:00
    7/19/95
    1995-07-19 00:00:00
    5/24/95
    1995-05-24 00:00:00
    11/16/95
    1995-11-16 00:00:00
    12/15/95
    1995-12-15 00:00:00
    5/19/95
    1995-05-19 00:00:00
    7/7/95
    1995-07-07 00:00:00
    12/29/95
    1995-12-29 00:00:00
    6/30/95
    1995-06-30 00:00:00
    4/7/95
    1995-04-07 00:00:00
    6/14/95
    1995-06-14 00:00:00
    12/15/95
    1995-12-15 00:00:00
    1/27/95
    1995-01-27 00:00:00
    6/30/95
    1995-06-30 00:00:00
    5/31/95
    1995-05-31 00:00:00
    11/22/95
    1995-11-22 00:00:00
    7/19/95
    1995-07-19 00:00:00
    7/18/95
    1995-07-18 00:00:00
    5/26/95
    1995-05-26 00:00:00
    11/17/95
    1995-11-17 00:00:00
    12/25/95
    1995-12-25 00:00:00
    7/28/95
    1995-07-28 00:00:00
    4/7/95
    1995-04-07 00:00:00
    11/10/95
    1995-11-10 00:00:00
    5/28/95
    1995-05-28 00:00:00
    10/27/95
    1995-10-27 00:00:00
    7/28/95
    1995-07-28 00:00:00
    12/24/95
    1995-12-24 00:00:00
    5/26/95
    1995-05-26 00:00:00
    8/25/95
    1995-08-25 00:00:00
    4/21/95
    1995-04-21 00:00:00
    3/10/95
    1995-03-10 00:00:00
    5/20/95
    1995-05-20 00:00:00
    7/13/95
    1995-07-13 00:00:00
    12/22/95
    1995-12-22 00:00:00
    8/18/95
    1995-08-18 00:00:00
    12/13/95
    1995-12-13 00:00:00
    2/10/95
    1995-02-10 00:00:00
    6/30/95
    1995-06-30 00:00:00
    10/20/95
    1995-10-20 00:00:00
    10/6/95
    1995-10-06 00:00:00
    2/9/95
    1995-02-09 00:00:00
    5/5/95
    1995-05-05 00:00:00
    8/1/95
    1995-08-01 00:00:00
    12/22/95
    1995-12-22 00:00:00
    10/27/95
    1995-10-27 00:00:00
    3/31/95
    1995-03-31 00:00:00
    4/26/95
    1995-04-26 00:00:00
    12/29/95
    1995-12-29 00:00:00
    5/12/95
    1995-05-12 00:00:00
    9/22/95
    1995-09-22 00:00:00
    9/13/95
    1995-09-13 00:00:00
    5/26/95
    1995-05-26 00:00:00
    8/11/95
    1995-08-11 00:00:00
    12/8/95
    1995-12-08 00:00:00
    12/22/95
    1995-12-22 00:00:00
    8/4/95
    1995-08-04 00:00:00
    9/1/95
    1995-09-01 00:00:00
    4/21/95
    1995-04-21 00:00:00
    3/24/95
    1995-03-24 00:00:00
    11/21/95
    1995-11-21 00:00:00
    10/27/95
    1995-10-27 00:00:00
    4/13/95
    1995-04-13 00:00:00
    9/7/95
    1995-09-07 00:00:00
    9/14/95
    1995-09-14 00:00:00
    12/22/95
    1995-12-22 00:00:00
    12/22/95
    1995-12-22 00:00:00
    3/24/95
    1995-03-24 00:00:00
    7/7/95
    1995-07-07 00:00:00
    2/3/95
    1995-02-03 00:00:00
    9/29/95
    1995-09-29 00:00:00
    11/3/95
    1995-11-03 00:00:00
    5/14/95
    1995-05-14 00:00:00
    10/6/95
    1995-10-06 00:00:00
    10/20/95
    1995-10-20 00:00:00
    5/5/95
    1995-05-05 00:00:00
    11/17/95
    1995-11-17 00:00:00
    5/10/95
    1995-05-10 00:00:00
    6/8/95
    1995-06-08 00:00:00
    12/15/95
    1995-12-15 00:00:00
    10/13/95
    1995-10-13 00:00:00
    1/20/95
    1995-01-20 00:00:00
    2/17/95
    1995-02-17 00:00:00
    6/2/95
    1995-06-02 00:00:00
    11/22/95
    1995-11-22 00:00:00
    7/12/95
    1995-07-12 00:00:00
    6/9/95
    1995-06-09 00:00:00
    3/31/95
    1995-03-31 00:00:00
    12/15/95
    1995-12-15 00:00:00
    11/2/95
    1995-11-02 00:00:00
    8/4/95
    1995-08-04 00:00:00
    2/17/95
    1995-02-17 00:00:00
    9/8/95
    1995-09-08 00:00:00
    9/8/95
    1995-09-08 00:00:00
    8/4/95
    1995-08-04 00:00:00
    2/16/95
    1995-02-16 00:00:00
    7/28/95
    1995-07-28 00:00:00
    3/17/95
    1995-03-17 00:00:00
    7/19/95
    1995-07-19 00:00:00
    8/11/95
    1995-08-11 00:00:00
    9/22/95
    1995-09-22 00:00:00
    5/12/95
    1995-05-12 00:00:00
    10/13/95
    1995-10-13 00:00:00
    1/20/95
    1995-01-20 00:00:00
    4/23/95
    1995-04-23 00:00:00
    9/24/95
    1995-09-24 00:00:00
    4/21/95
    1995-04-21 00:00:00
    1/13/95
    1995-01-13 00:00:00
    12/29/95
    1995-12-29 00:00:00
    4/28/95
    1995-04-28 00:00:00
    10/27/95
    1995-10-27 00:00:00
    12/22/95
    1995-12-22 00:00:00
    8/4/95
    1995-08-04 00:00:00
    12/1/95
    1995-12-01 00:00:00
    8/25/95
    1995-08-25 00:00:00
    3/17/95
    1995-03-17 00:00:00
    4/21/95
    1995-04-21 00:00:00
    3/16/95
    1995-03-16 00:00:00
    5/26/95
    1995-05-26 00:00:00
    9/12/95
    1995-09-12 00:00:00
    6/9/95
    1995-06-09 00:00:00
    3/3/95
    1995-03-03 00:00:00
    6/23/95
    1995-06-23 00:00:00
    12/13/95
    1995-12-13 00:00:00
    5/24/95
    1995-05-24 00:00:00
    11/2/95
    1995-11-02 00:00:00
    8/25/95
    1995-08-25 00:00:00
    10/17/95
    1995-10-17 00:00:00
    10/6/95
    1995-10-06 00:00:00
    9/29/95
    1995-09-29 00:00:00
    10/13/95
    1995-10-13 00:00:00
    9/15/95
    1995-09-15 00:00:00
    5/30/95
    1995-05-30 00:00:00
    7/28/95
    1995-07-28 00:00:00
    11/15/95
    1995-11-15 00:00:00
    12/22/95
    1995-12-22 00:00:00
    9/8/95
    1995-09-08 00:00:00
    4/7/95
    1995-04-07 00:00:00
    1/10/95
    1995-01-10 00:00:00
    10/20/95
    1995-10-20 00:00:00
    2/25/95
    1995-02-25 00:00:00
    4/12/95
    1995-04-12 00:00:00
    5/13/95
    1995-05-13 00:00:00
    12/29/95
    1995-12-29 00:00:00
    12/29/95
    1995-12-29 00:00:00
    1/13/95
    1995-01-13 00:00:00
    8/25/95
    1995-08-25 00:00:00
    7/14/95
    1995-07-14 00:00:00
    8/11/95
    1995-08-11 00:00:00
    9/27/95
    1995-09-27 00:00:00
    1/1/95
    1995-01-01 00:00:00
    10/27/95
    1995-10-27 00:00:00
    12/22/95
    1995-12-22 00:00:00
    5/21/95
    1995-05-21 00:00:00
    9/20/95
    1995-09-20 00:00:00
    3/16/95
    1995-03-16 00:00:00
    10/6/95
    1995-10-06 00:00:00
    4/11/95
    1995-04-11 00:00:00
    10/20/95
    1995-10-20 00:00:00
    1/11/95
    1995-01-11 00:00:00
    7/4/95
    1995-07-04 00:00:00
    9/15/95
    1995-09-15 00:00:00
    7/11/95
    1995-07-11 00:00:00
    3/2/95
    1995-03-02 00:00:00
    9/8/95
    1995-09-08 00:00:00
    12/1/95
    1995-12-01 00:00:00
    12/15/95
    1995-12-15 00:00:00
    5/24/95
    1995-05-24 00:00:00
    5/19/95
    1995-05-19 00:00:00
    10/27/95
    1995-10-27 00:00:00
    3/3/95
    1995-03-03 00:00:00
    1/31/95
    1995-01-31 00:00:00
    10/28/95
    1995-10-28 00:00:00
    4/28/95
    1995-04-28 00:00:00
    6/25/92
    1992-06-25 00:00:00
    11/25/92
    1992-11-25 00:00:00
    11/19/92
    1992-11-19 00:00:00
    6/19/92
    1992-06-19 00:00:00
    5/22/92
    1992-05-22 00:00:00
    5/15/92
    1992-05-15 00:00:00
    11/13/92
    1992-11-13 00:00:00
    10/9/92
    1992-10-09 00:00:00
    10/9/92
    1992-10-09 00:00:00
    7/30/92
    1992-07-30 00:00:00
    12/11/92
    1992-12-11 00:00:00
    2/14/92
    1992-02-14 00:00:00
    4/3/92
    1992-04-03 00:00:00
    3/20/92
    1992-03-20 00:00:00
    7/10/92
    1992-07-10 00:00:00
    9/15/92
    1992-09-15 00:00:00
    7/16/92
    1992-07-16 00:00:00
    9/9/92
    1992-09-09 00:00:00
    9/16/92
    1992-09-16 00:00:00
    8/7/92
    1992-08-07 00:00:00
    5/28/92
    1992-05-28 00:00:00
    3/26/92
    1992-03-26 00:00:00
    10/9/92
    1992-10-09 00:00:00
    8/28/92
    1992-08-28 00:00:00
    12/23/92
    1992-12-23 00:00:00
    11/5/92
    1992-11-05 00:00:00
    8/13/92
    1992-08-13 00:00:00
    9/30/92
    1992-09-30 00:00:00
    10/9/92
    1992-10-09 00:00:00
    11/25/92
    1992-11-25 00:00:00
    3/5/92
    1992-03-05 00:00:00
    5/8/92
    1992-05-08 00:00:00
    9/18/92
    1992-09-18 00:00:00
    12/10/92
    1992-12-10 00:00:00
    2/21/92
    1992-02-21 00:00:00
    6/4/92
    1992-06-04 00:00:00
    10/1/92
    1992-10-01 00:00:00
    5/1/92
    1992-05-01 00:00:00
    4/3/92
    1992-04-03 00:00:00
    9/11/92
    1992-09-11 00:00:00
    11/18/92
    1992-11-18 00:00:00
    9/1/92
    1992-09-01 00:00:00
    9/18/92
    1992-09-18 00:00:00
    12/18/92
    1992-12-18 00:00:00
    3/13/92
    1992-03-13 00:00:00
    9/25/92
    1992-09-25 00:00:00
    9/4/92
    1992-09-04 00:00:00
    1/10/92
    1992-01-10 00:00:00
    9/2/92
    1992-09-02 00:00:00
    1/17/92
    1992-01-17 00:00:00
    9/15/92
    1992-09-15 00:00:00
    10/2/92
    1992-10-02 00:00:00
    4/15/92
    1992-04-15 00:00:00
    12/2/92
    1992-12-02 00:00:00
    2/7/92
    1992-02-07 00:00:00
    12/16/92
    1992-12-16 00:00:00
    1/31/92
    1992-01-31 00:00:00
    7/1/92
    1992-07-01 00:00:00
    9/16/92
    1992-09-16 00:00:00
    11/13/92
    1992-11-13 00:00:00
    3/13/92
    1992-03-13 00:00:00
    7/10/92
    1992-07-10 00:00:00
    9/2/92
    1992-09-02 00:00:00
    4/10/92
    1992-04-10 00:00:00
    6/30/92
    1992-06-30 00:00:00
    3/5/92
    1992-03-05 00:00:00
    10/16/92
    1992-10-16 00:00:00
    12/18/92
    1992-12-18 00:00:00
    2/7/92
    1992-02-07 00:00:00
    4/22/92
    1992-04-22 00:00:00
    6/12/92
    1992-06-12 00:00:00
    6/26/92
    1992-06-26 00:00:00
    9/18/92
    1992-09-18 00:00:00
    2/28/92
    1992-02-28 00:00:00
    9/17/92
    1992-09-17 00:00:00
    9/4/92
    1992-09-04 00:00:00
    7/24/92
    1992-07-24 00:00:00
    5/22/92
    1992-05-22 00:00:00
    7/24/92
    1992-07-24 00:00:00
    4/3/92
    1992-04-03 00:00:00
    3/27/92
    1992-03-27 00:00:00
    5/22/92
    1992-05-22 00:00:00
    2/21/92
    1992-02-21 00:00:00
    1/17/92
    1992-01-17 00:00:00
    9/18/92
    1992-09-18 00:00:00
    1/15/92
    1992-01-15 00:00:00
    3/6/92
    1992-03-06 00:00:00
    12/4/92
    1992-12-04 00:00:00
    8/14/92
    1992-08-14 00:00:00
    8/8/92
    1992-08-08 00:00:00
    10/16/92
    1992-10-16 00:00:00
    3/13/92
    1992-03-13 00:00:00
    10/1/92
    1992-10-01 00:00:00
    8/20/92
    1992-08-20 00:00:00
    7/17/92
    1992-07-17 00:00:00
    10/23/92
    1992-10-23 00:00:00
    4/10/92
    1992-04-10 00:00:00
    7/10/92
    1992-07-10 00:00:00
    8/7/92
    1992-08-07 00:00:00
    5/1/92
    1992-05-01 00:00:00
    11/6/92
    1992-11-06 00:00:00
    12/18/92
    1992-12-18 00:00:00
    8/28/92
    1992-08-28 00:00:00
    1/14/92
    1992-01-14 00:00:00
    4/24/92
    1992-04-24 00:00:00
    10/1/92
    1992-10-01 00:00:00
    4/15/92
    1992-04-15 00:00:00
    4/10/92
    1992-04-10 00:00:00
    8/14/92
    1992-08-14 00:00:00
    2/28/92
    1992-02-28 00:00:00
    8/7/92
    1992-08-07 00:00:00
    12/25/92
    1992-12-25 00:00:00
    9/18/92
    1992-09-18 00:00:00
    12/26/92
    1992-12-26 00:00:00
    1/10/92
    1992-01-10 00:00:00
    9/25/92
    1992-09-25 00:00:00
    4/17/92
    1992-04-17 00:00:00
    12/17/92
    1992-12-17 00:00:00
    9/2/92
    1992-09-02 00:00:00
    7/24/92
    1992-07-24 00:00:00
    12/25/92
    1992-12-25 00:00:00
    12/4/92
    1992-12-04 00:00:00
    12/30/92
    1992-12-30 00:00:00
    6/26/92
    1992-06-26 00:00:00
    10/1/92
    1992-10-01 00:00:00
    8/20/92
    1992-08-20 00:00:00
    3/27/92
    1992-03-27 00:00:00
    8/14/92
    1992-08-14 00:00:00
    1/1/92
    1992-01-01 00:00:00
    7/31/92
    1992-07-31 00:00:00
    10/23/92
    1992-10-23 00:00:00
    7/24/92
    1992-07-24 00:00:00
    1/1/92
    1992-01-01 00:00:00
    6/12/81
    1981-06-12 00:00:00
    8/21/81
    1981-08-21 00:00:00
    6/23/81
    1981-06-23 00:00:00
    7/10/81
    1981-07-10 00:00:00
    12/23/81
    1981-12-23 00:00:00
    6/12/81
    1981-06-12 00:00:00
    8/21/81
    1981-08-21 00:00:00
    8/7/81
    1981-08-07 00:00:00
    10/15/81
    1981-10-15 00:00:00
    5/22/81
    1981-05-22 00:00:00
    4/10/81
    1981-04-10 00:00:00
    12/4/81
    1981-12-04 00:00:00
    5/15/81
    1981-05-15 00:00:00
    5/1/81
    1981-05-01 00:00:00
    9/25/81
    1981-09-25 00:00:00
    8/13/81
    1981-08-13 00:00:00