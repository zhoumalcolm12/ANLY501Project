#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:12:26 2018

@author: zhoumengzhi
"""

import requests

import sys
from suds import null, WebFault
from suds.client import Client
import logging


username = 'yangxi1994'
apikey = '7038ce657e42de52eaf5af4126a5fbbf0f4f0a43'
url_base = ''.join(['http://',username,':',apikey,'@flightxml.flightaware.com/json/FlightXML2/Enroute'])

print(url_base)

url_post = {'airport': 'KSMO',
                'filter': '',
                'howmany':'10',
                'offset': '0'}

# Use the get function of requests library to get data.
response=requests.get(url_base, url_post)

# The result is convereted to json format
jsontxt = response.json()
print(jsontxt)