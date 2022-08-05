import argparse
import datetime
import json
import os
import pickle
import time
import urllib.request

import feedparser
import requests


def main(args):
    base_url = 'http://export.arxiv.org/api/query?'

    search_query = args.query
    start_idx = args.start_idx
    max_results = args.max_results
    day_minus = -1*args.day_minus
    n_requests = args.n_requests
    append = args.append

    target_day = (datetime.date.today() + datetime.timedelta(days=day_minus))

    if not os.path.exists('data'):
        os.mkdir('data')
        
    if append:
        with open('./data/raw_title.pkl', 'rb') as f:
            title_list = pickle.load(f)
        with open('./data/raw_abst.pkl', 'rb') as f:
            abst_list = pickle.load(f)
        with open('./data/raw_link.pkl', 'rb') as f:
            link_list = pickle.load(f)
    else:
        title_list = []
        abst_list = []
        link_list = []

    print('Initial number of papers: {}'.format(len(title_list)))

    n_papers = 0
    break_flag = False
    for i in range(n_requests):
        query = 'search_query={0}&start={1}&max_results={2}&sortBy=lastUpdatedDate&sortOrder=descending'.format(search_query, n_papers+start_idx, max_results)
        # feedparser v4.1
        feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
        feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'
        
        # perform a GET request using the base_url and query
        with urllib.request.urlopen(base_url + query) as url:
            response = url.read()
            
        # parse the response using feedparser
        feed = feedparser.parse(response)
        total_results = int(feed.feed.opensearch_totalresults)

        if len(feed.entries) != max_results and len(feed.entries)+n_papers+start_idx != total_results:
            print('Number of responses < max_results')
            print('Current number of papers: {}'.format(n_papers+start_idx))
            time.sleep(10)
            continue
        
        for entry in feed.entries:
            published_date = (entry.updated.split('T'))[0]
            published_date = datetime.datetime.strptime(published_date, '%Y-%m-%d').date()
            if published_date < target_day:
                break_flag = True
                break
            title_list.append(entry.title)
            abst_list.append(entry.summary.replace('\n', ' '))

            link_exists = False
            for link in entry.links:
                if link.rel == 'alternate':
                    link_list.append(link['href'])
                    link_exists = True
                    break
            if not link_exists:
                link_list.append(entry.links[0]['href'])
            
            n_papers += 1
            if total_results <= n_papers+start_idx:
                break_flag = True
                break
        
        print('Current number of papers: {}'.format(n_papers+start_idx))
        if break_flag:
            break
        time.sleep(10)
    
    print('Total number of papers: {}'.format(len(title_list)))

    with open('./data/raw_title.pkl', 'wb') as f:
        pickle.dump(title_list, f)
    with open('./data/raw_abst.pkl', 'wb') as f:
        pickle.dump(abst_list, f)
    with open('./data/raw_link.pkl', 'wb') as f:
        pickle.dump(link_list, f)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='cat:cs.LG')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--max-results', type=int, default=10000)
    parser.add_argument('--day-minus', type=int, default=10000)
    parser.add_argument('--n-requests', type=int, default=10000)
    parser.add_argument('--append', action='store_true')

    args = parser.parse_args()
    main(args)