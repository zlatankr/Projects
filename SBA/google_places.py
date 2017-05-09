# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:13:37 2017

@author: User
"""

restaurants = [{'name':'La Taqueria', 'location': 'San Francisco'},{'name':'La Corneta',
               'location': 'San Francisco'},{'name': 'Taqueria El Farolito', 
                                          'location': 'San Francisco'}]

def GetGooglePlacesRatings(restaurants):
    # 'restaurants': array of dicts = [{u"name":u"El Celler de Can Roca",u"location":u"Girona"},{u"name":u"Koy Shunka",u"location":u"Barcelona"}]
 
    # set google places api
    from googleplaces import GooglePlaces, types, lang
    google_places_api_key = 'AIzaSyD_xdgTbmgL-pVFykN-rMZIQF3DdeMszA0'
    google_places = GooglePlaces(google_places_api_key)
     
    # search restaurants
    for restaurant in restaurants:
        restaurantFound = 0
        restaurantName = restaurant["name"].lower()
        restaurantLocation = restaurant["location"].lower()
 
        query_result = google_places.nearby_search(keyword=restaurantName,
                                                   location=restaurantLocation, radius="50000",
                                                   types=[types.TYPE_FOOD], sensor="false")
         
        # loop query results
        for place in query_result.places:
            if (restaurantName == place.name.lower() or
                'restaurant ' + restaurantName  == place.name.lower() or
                'restaurante ' + restaurantName == place.name.lower() or
                restaurantName + ' restaurant'  == place.name.lower() or
                restaurantName + ' restaurante'  == place.name.lower() or
                'restaurant ' + place.name.lower()  == restaurantName or
                'restaurante ' + place.name.lower() == restaurantName or
                place.name.lower() + ' restaurant'  == restaurantName or
                place.name.lower() + ' restaurante'  == restaurantName
                ):
                place.get_details()
                if 'user_ratings_total' in place.details:
                    restarurantNumRatings=place.details['user_ratings_total']
                else:
                    restarurantNumRatings='-'
                print place.name, '\t', place.rating, '\t', restarurantNumRatings
                restaurantFound = 1
                break
         
        # restaurant not found, print query results for further manual inspection
        if(restaurantFound==0):
            print restaurant["name"], '\t-\t-\t(',
            for place in query_result.places:
                print place.name + ' - ',
            print ')'
    return

import requests
from bs4 import BeautifulSoup

page = requests.get('https://maps.googleapis.com/maps/api/place/textsearch/json?query=El+Farolito+2779+Mission+Street+San+Francisco&key=AIzaSyD_xdgTbmgL-pVFykN-rMZIQF3DdeMszA0')
soup = BeautifulSoup(page.content, 'html.parser')