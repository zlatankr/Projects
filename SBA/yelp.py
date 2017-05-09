# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:33:48 2017

@author: User
"""

import requests
from bs4 import BeautifulSoup

page = requests.get('https://maps.googleapis.com/maps/api/place/textsearch/json?query=El+Farolito+2779+Mission+Street+San+Francisco&key=AIzaSyD_xdgTbmgL-pVFykN-rMZIQF3DdeMszA0')
soup = BeautifulSoup(page.content, 'html.parser')

data = {'grant_type': 'client_credentials',
        'client_id': 'kiD4d8ra_jebdWlH9XZyLw',
        'client_secret': 'yAm96WC7WVyKhCUobTu0YX3nl5EqmlnVQnQrdRMomrMCDg4auhg7aK9QJyEIyUIm'}
token = requests.post('https://api.yelp.com/oauth2/token', data=data)
access_token = token.json()['access_token']
url = 'https://api.yelp.com/v3/businesses/search'
headers = {'Authorization': 'bearer %s' % access_token}
params = {'location': 'San Bruno',
          'term': 'Japanese Restaurant',
          'pricing_filter': '1, 2',
          'sort_by': 'rating'
         }

resp = requests.get(url=url, params=params, headers=headers)

import pprint
pprint.pprint(resp.json()['businesses'])





page = requests.get('https://api.yelp.com/v3/businesses/search?term=delis&latitude=37.786882&longitude=-122.399972')
soup = BeautifulSoup(page.content, 'html.parser')

