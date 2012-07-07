#! /usr/bin/python

import json
import csv
import json
import sys
from collections import defaultdict



folderout = "./submissions/"
folderin = "./"

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
