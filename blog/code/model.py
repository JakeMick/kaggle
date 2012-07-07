#! /usr/bin/python
"""
Blog Favoriting Analysis
"""
from os import path
import json
import sqlite3
import nltk

class processing():
    def __init__(self):
        """General code for data projects"""
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submissions')
        self.load_file_handles()

    def load_file_handles(self):
        """Application specific code for data projects"""
        json_filenames = {
                'kaggle_blog': 'kaggle-stats-blogs-20111123-20120423.json',
                'kaggle_user' : 'kaggle-stats-users-20111123-20120423.json',
                'test_post' : 'testPosts.json',
                'test_post_thin' : 'testPostsThin.json',
                'test_user' : 'testUsers.json',
                'train_post' : 'trainPosts.json',
                'train_post_this' : 'trainPostsThin.json'}
        self.data = {}
        for name, filepath in json_filenames.items():
            long_filepath = path.join(self.data_dir, filepath)
            self.data[name] = open(long_filepath, 'r')

def bullshit():
    testUsers = open(folderin + "testUsers.json", 'r')
    testPostsThin = open(folderin + "testPosts.json", 'r')
    blogsAg = open(folderin + "aggregate/kaggle-stats-blogs-20111123-20120423.json", "r")
    popsub = open(folderout + "popsub.csv", "w")


    max = 0
    for line in blogsAg:
        blog = json.loads(line)
        num_posts = float(blog['num_posts'])
        num_likes = float(blog['num_likes'])
        if num_posts > 1000 and num_likes/num_posts > max:
            best = blog
            max = num_likes/num_posts


    print best

    ids = list()
    for line in testPostsThin:
        post = json.loads(line)
        blog = post['blog']
        if int(blog) == best['blog_id']:
            ids.append(post['post_id'])
            print ids

    popsub.write('''"posts"\n''')


    for i in range(16262):
        popsub.write(" ".join(ids) + "\n")


    testUsers.close()
    testPostsThin.close()
    blogsAg.close()
    popsub.close()
