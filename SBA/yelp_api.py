import requests
from sqlalchemy import create_engine
import pandas as pd

def main():
    data = {'grant_type': 'client_credentials',
            'client_id': 'kiD4d8ra_jebdWlH9XZyLw',
            'client_secret': 'yAm96WC7WVyKhCUobTu0YX3nl5EqmlnVQnQrdRMomrMCDg4auhg7aK9QJyEIyUIm'}
    token = requests.post('https://api.yelp.com/oauth2/token', data=data)
    access_token = token.json()['access_token']
    url = 'https://api.yelp.com/v3/businesses/search'
    headers = {'Authorization': 'bearer %s' % access_token}

    engine = create_engine('postgresql://datascisba@c4sf-sba:M14yeGIejisf9kz@c4sf-sba.postgres.database.azure.com:5432/postgres')
    sfdo = pd.read_sql('select * from stg_analytics.sba_sfdo;', con=engine)
    sfdo['full_address'] = sfdo['borr_street']+', '+sfdo['borr_city']+', '+sfdo['borr_state']+', '+sfdo['borr_zip']
    
    sfdo['yelp_rating'] = None
    sfdo['yelp_total_reviews'] = None
    sfdo['yelp_url'] = None
    
    for i in range(len(sfdo)):
        address = sfdo.iloc[i]['full_address']
        name = sfdo.iloc[i]['borr_name']
        params = {'location': address,
                  'term': name,
                  'radius': 100,
                  'limit':1
                  }
    
        resp = requests.get(url=url, params=params, headers=headers)
    
        try:
            sfdo['yelp_rating'].loc[i] = resp.json()['businesses'][0]['rating']
            sfdo['yelp_total_reviews'].loc[i] = resp.json()['businesses'][0]['review_count']
            sfdo['yelp_url'].loc[i] = resp.json()['businesses'][0]['url']
        except:
            pass

    sfdo.to_sql(name='sba_sfdo_yelp', con=engine, schema='stg_analytics', if_exists='replace',chunksize=1000, index=False)

if __name__ == '__main__':
    """See https://stackoverflow.com/questions/419163/what-does-if-name-main-do"""
    main()