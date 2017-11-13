#!/usr/bin/env python
#coding=utf-8

import os
import sys
import time
import Queue
import urllib2
import threading
import json
from logzero import logger
import sys
sys.path.append('..')
import load
import time

path_data = ""
res_file = path_data + 'page_lab_cv_2w.csv'
bad_page_file = path_data + 'bad_page_cv_2w.csv'

lock = threading.Lock()

class WikiReader(threading.Thread):

    def __init__(self, pageQueue, file, bad_file):
        threading.Thread.__init__(self)
        self.pageQueue = pageQueue
        self.file = file
        self.bad_file = bad_file

    def stop(self):
        pass

    def fetchContent(self, page, pageUrl):
        for i in range(3):
            try:
                res = urllib2.urlopen(pageUrl, timeout=10)
                return res.read()
            except urllib2.URLError,e:
                logger.error(e.reason)
                logger.error('can not find %s', pageUrl)
                time.sleep(0.05)
        return None

    def writeContent(self,page,content):
        if lock.acquire():
            try:
                if content == None:
                    self.bad_file.write(page + '\n')
                else:
                    res = json.loads(content)
                    for item in res['items']:
                        time = item['timestamp']
                        views = item['views']
                        page_new = page
                        if ',' in page or '"' in page:
                            page_new = page.replace('"','""')
                            page_new = '"' + page_new + '"'
                        self.file.write(page_new + ',' + str(time) + ',' + str(views) + '\n')
            except  Exception, e:
                logger.error(e.message)
                logger.error('can not save %s', page, e.message)
            lock.release()

    def run(self):
        while not self.pageQueue.empty():
            count = 145063 - self.pageQueue.qsize()
            if count % 1000 == 0:
                logger.info('finished %d pages', count)
            [page, pageUrl] = self.pageQueue.get()
            content = self.fetchContent(page, pageUrl)
            self.writeContent(page, content)

class PageUrl(object):
    queue = Queue.Queue()

    def __init__(self):
        page_info = load.load_page_info()
        for Page,agent_type,access_type,project,page_name in page_info[['Page','agent_type','access_type','project','page_name']].values:
            url = self.get_url(project, access_type, agent_type, '2016121800', '2016123100',page_name)
            self.queue.put([Page, url])

    def getQueue(self):
        return self.queue

    def get_url(self,project, access, agent, start, end, page_name):
        url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'
        page_name = urllib2.quote(page_name)
        '''
        1. +  URL 中+号表示空格 %2B   
        2. 空格 URL中的空格可以用+号或者编码 %20   
        3. /  分隔目录和子目录 %2F    
        4. ?  分隔实际的 URL 和参数 %3F    
        5. % 指定特殊字符 %25    
        6. # 表示书签 %23    
        7. & URL 中指定的参数间的分隔符 %26    
        8. = URL 中指定参数的值 %3D  
        '''
        page_name = page_name.replace('+','%2B')
        page_name = page_name.replace(' ','%20')
        page_name = page_name.replace('?','%3F')
        page_name = page_name.replace('/','%2F')
        page_name = page_name.replace('#','%23')
        page_name = page_name.replace('&','%26')
        page_name = page_name.replace('=','%3D')
        return url+project + '/' + access + '/' + agent + '/' + page_name + '/daily/' + start + '/' + end

if __name__ == '__main__':
    pageURLQueue = PageUrl().getQueue()
    file = open(res_file,'w')
    bad_file = open(bad_page_file,'w')
    for i in range(100):
        WikiReader(pageURLQueue, file, bad_file).start()
