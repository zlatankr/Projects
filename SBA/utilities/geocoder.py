"""
Geocoding Utility:

We'll be using this: https://github.com/slimkrazy/python-google-places#python-google-places
to geocode addresses.
"""
import os

import pandas as pd

from googleplaces import GooglePlaces, types, lang

YOUR_API_KEY = os.getenv('GOOGLE_PLACES_API')


def geocode(df, api_key=YOUR_API_KEY):
    """Add Geocoded columns to df

    Keyword Args:
    df: Dataframe which must have an "address" column with a clean address
    api_key: Google Places API Key
    """
    google_places = GooglePlaces(api_key)
    matches = []

    # This counter is just for debugging purposes since I don't want to hit the API threshold
    i = 0
    for place in df.address:
        print(place)
        print(i)
        query_result = google_places.nearby_search(
            location=place,
            radius=100
        )
        matches.append(query_result.places)
        i = i + 1
        if i == 10:
            break

    for i in range(len(matches), len(df)):
        matches.append(None)
    df['matches'] = matches

    return df

