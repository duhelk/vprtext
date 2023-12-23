"""
Using google mapping platform
"""

import urllib.parse

key = ""
size = '640x640'

# with open('key.txt') as reader:
#     key = reader.readline()

#Create request for Google Static Streetview API
def create_streetview_request(loc,head):
    request = "https://maps.googleapis.com/maps/api/streetview?%s" % urllib.parse.urlencode((
    ('size', size),
    ('location', loc),
    ('heading', head),
    ('key', key)
    ))
    return request

#Create request for Google Directions API
def create_direction_request(origin, destination, mode='walking'):
    request = "https://maps.googleapis.com/maps/api/directions/json?%s" % urllib.parse.urlencode((
    ('origin', origin),
    ('destination', destination),
    ('mode', mode),
    ('key', key)
    ))
    return request

#Place Detail request
def create_place_detail_request(place_id):
    request = "https://maps.googleapis.com/maps/api/place/details/json?%s" % urllib.parse.urlencode((
    ('place_id', place_id),
    #('fields', 'name,rating,formatted_phone_number'),
    ('key', key)
    ))
    return request

#Place Photo request
def create_place_photo_request(photo_ref):
    request = "https://maps.googleapis.com/maps/api/place/photo?%s" % urllib.parse.urlencode((
    ('photoreference', photo_ref),
    ('maxwidth', 400),
    ('key', key)
    ))
    return request

def create_nearby_search_request(location, radius):
    request = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?%s" % urllib.parse.urlencode((
    ('location', location),
    ('radius', radius),
    #('type', 'restaurant'),
    #('keyword', 'cruise')
    ('key', key)
    ))
    return request

def create_next_page_request(page_token):
    request = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?%s" % urllib.parse.urlencode((
    ('pagetoken', page_token),
    ('key', key)
    ))
    return request