# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:34:22 2019
@author: Aline

Helper functions to download and parse data from ArXiv, and to update 
datasets based on new records
"""
import feedparser
import urllib.request as libreq
import csv
import time
import pandas as pd
import numpy as np


# -------------------- set global vars ------------------------------------
# Base ARXIV api query url
base_url = 'http://export.arxiv.org/api/query?'
wait_time = 4 # s don't overload arxiv servers
# -------------------------------------------------------------------------

# Search parameters
search_query = 'all:machine+learning' # search for electron in all fields
start = 0                       
total_results = 52000              
results_per_iteration = 500

for i in range(start, total_results, results_per_iteration):    
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, i, results_per_iteration)
    # perform a GET request using the base_url and query
    response = libreq.urlopen(base_url+query).read()
    # parse the response using feedparser
    feed = feedparser.parse(response)
    # Run through each entry and apeend the information to csv
    record = []
    for entry in feed.entries:
        record.append([entry.id.split('/abs/')[-1],
                       entry.published,
                       entry.title,
                       entry.author,
                       entry.summary])
    # Remember to play nice and sleep a bit before you call
    # the api again!
    time.sleep(wait_time)
    # add lines to file in disk
    with open("output.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        try:
            wr.writerows(record)
        except:
            pass
    print("iteration:", i)
    



search_query = 'all:machine+learning' # search for electron in all fields
start = 0                       
total_results = 52000              
results_per_iteration = 500      
wait_time = 4 

def get_ids_from_query():
    for i in range(start, total_results, results_per_iteration):    
        query = 'search_query=%s&start=%i&max_results=%i' % (search_query, i, results_per_iteration)
        response = libreq.urlopen(base_url+query).read()
        feed = feedparser.parse(response)
        record = []
        for entry in feed.entries:
            record.append([entry.id.split('/abs/')[-1]])
        time.sleep(wait_time)
        with open("missing.csv", "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            try:
                wr.writerows(record)
            except:
                print('error at:', i)
        print("iteration:", i)


def get_papers_from_id(id_list, start=0, results_per_iteration=10, wait_time =3):
    total_results = len(id_list)
    sep = ','
    for i in range(start, total_results, results_per_iteration):
        # concatenates series of ids to be retrieved from arxiv
        qids = sep.join(id_list[start:start + results_per_iteration])
        query = 'id_list=%s' % (qids)
        response = libreq.urlopen(base_url+query).read()
        feed = feedparser.parse(response)
        record = []
        for entry in feed.entries:
            record.append([entry.id.split('/abs/')[-1],
                           entry.published,
                           entry.title,
                           entry.author,
                           entry.summary])
        time.sleep(wait_time)
        record = pd.DataFrame(record)
        print(record.shape[0])
        try:
            record.to_csv('new.csv', mode='a', index=False, header=False)
        except:
            print('error at:', i)
        print("iteration:", i)


#### 
    
   
#    try:
#        data.decode('UTF-8')
#    except UnicodeDecodeError:
#        return False
#    else:
#        return True

# TODO update 
def append_new_data(path_olddata, path_newdata):
    new = pd.read_csv(path_newdata)
    new_count = new.shape[0]
    try:
        new.to_csv(path_olddata, mode='a', index=False, header=False)
        print('appended {} new rows'.format(new_count))
    except:
        print('failed')



def updated_missing_list(path_full_list, path_data):
    downloaded = pd.read_csv(path_data, encoding='latin_1')
    full_list = pd.read_csv(path_full_list)
    missing = full_list[~full_list.id.isin(downloaded.id)] 
    
    return missing
      
get_papers_from_id(missing)
append_new_data('output.csv', 'new.csv')
updated_missing_list
                