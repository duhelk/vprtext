
import os
import pandas as pd
import time
import urllib
import json
from .maprequest_util import create_nearby_search_request, create_next_page_request 
from .imagemeta_util import get_image_location

def get_nearby_places(location, radius):
    url = create_nearby_search_request(location, radius)
    ur = urllib.request.urlopen(url)
    result = json.load(ur)
    return result

def get_next_page(page_token):
    url = create_next_page_request(page_token)
    ur = urllib.request.urlopen(url)
    result = json.load(ur)
    return result

def get_location(image_path):
    lat,lon = get_image_location(image_path)
    return f"{lat},{lon}"

def get_place_list(location, radius=100.0):
    place_list = []
    while True:
        if len(place_list) == 0:
            places_result = get_nearby_places(location, radius)
        else:
            places_result = get_next_page(page_token)
        place_list.extend(places_result['results'])
        if 'next_page_token' not in places_result:
            break
        page_token = places_result['next_page_token']
        #-- Wait for the request to be ready
        time.sleep(2)
    return place_list

def get_place_names(place_list):
    place_names = []
    for place in place_list[:]:
        place_names.append(place['name'])
    return place_names

def get_nearby_place_list(image_path, map_csv="MapData.csv", radius=50.0):
    nearby_places = []
    query_id = image_path.split('/')[-1].split('.')[0]
    inc_header = True
    if os.path.exists(map_csv):
        map_df = pd.read_csv(map_csv)
        if not map_df.empty:
            inc_header = False
            nearby_places = map_df.loc[map_df.QID == query_id,'Places'].tolist()
            
    if len(nearby_places) == 0:
        location = get_location(image_path)
        nearby_places = get_place_list(location, radius=radius)
        df = pd.DataFrame(data=[{'QID' :query_id, 'Places': nearby_places}])
        df.to_csv(map_csv, mode='a', header=inc_header)
    else:
        nearby_places = eval(nearby_places[0])
    return nearby_places


def get_nearby_place_names(image_path, map_csv="MapData.csv", radius=50.0):
    nearby_places = get_nearby_place_list(image_path, map_csv, radius)
    place_names = get_place_names(nearby_places)
    return place_names
