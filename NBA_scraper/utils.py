import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import datetime
from sqlalchemy import create_engine
import os
import timeit

def box_scores(url):
    start = timeit.default_timer()
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    boxes = pd.DataFrame()
    for i in re.findall('div_box_.*?_basic', str(soup)):
        box = str(soup.find_all(id=i))

        stats = re.findall('data-stat=.+?>(.+?)<',box)
        del stats[0]
        for s, m in enumerate(stats):
            if len(re.findall('html">(.+)',str(m))) > 0:
                stats[s] = re.findall('html">(.+)',str(m))[0]
        
        len_column_headers = len([th.getText() for th in 
                          soup.findAll('tr', limit=2)[1].findAll('th')])
        
        new = []
        for j in range(0,len(stats)-1):
            if (stats[j+1][:7] <> 'Did Not' and stats[j][:7] <> 'Did Not') and (stats[j+1][:8] <> 'Not With' and stats[j][:8] <> 'Not With'):
                new.append(stats[j])
        new.append(stats[-1])
        stats_clean = [new[u:u+len_column_headers] for u in range(0, len(new), len_column_headers)]
        teamid = re.findall('div_box_(.*?)_basic', i)[0]
        team = re.findall('id=\"box_'+teamid+'_basic\"><caption>(.*?)\(', str(soup))[0].strip()
        stats_clean[0].insert(0,'Team')
        for z in range(1, len(stats_clean)):
            stats_clean[z].insert(0, team)
            
        stats_clean[0].insert(0,'Date')
        for k in range(1, len(stats_clean)):
            stats_clean[k].insert(0, re.findall('[0-9]{8}', url)[0])
        boxes = boxes.append(stats_clean)
    
    boxes = boxes.rename(columns = dict(zip(boxes.columns, boxes.iloc[0,:])))
    boxes = boxes.rename(columns = {'Starters': 'Player'})
    boxes = boxes.drop(boxes[boxes['MP'] == 'MP'].index)
    boxes['Season'] = url[-17:-13] if url[-13] == '0' else str(int(url[-17:-13])+1)
    boxes.replace('</td>', '', inplace=True)
    print timeit.default_timer() - start
    return boxes
    

def get_schedule(years):
    months = [datetime.date(2000, m, 1).strftime('%m - %B').split(' ')[-1].lower() for m in [10,11,12,1,2,3,4]]
    all_box_urls = []
    for year in years:
        print year
        for month in months:
            url = 'http://www.basketball-reference.com/leagues/NBA_'+str(year)+'_games-'+month+'.html'
            page = requests.get(url)
            if page.status_code == 404:
                pass
            else:
                soup = str(BeautifulSoup(page.content, 'html.parser'))
                try:
                    soup = soup.replace(soup[soup.index('Playoffs</th></tr>'):],'')
                except:
                    pass
                addresses = re.findall('box_score_text"><a href="(/boxscores/.+?html)', soup)
                addresses = ['http://www.basketball-reference.com'+i for i in addresses]
                all_box_urls.extend(addresses)
    return all_box_urls

all_box_urls = get_schedule(range(1980,2018))

engine = create_engine('mysql+pymysql://'+os.environ['dbuser']+':'+os.environ['pw']+'@z1.cdhwgvcyc4xh.us-west-1.rds.amazonaws.com/Z1')

all_data = pd.DataFrame()
for i in all_box_urls:
    try:
        box = box_scores(i)
    except:
        print 'error with', i
        break
    try:
        pd.concat((all_data, box))
    except:
        print 'error with', i
    

















page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

# We can find individual box scores using a different `id` tag:

re.findall('div_box_.*?_basic', str(soup))

re.findall('id=\"box_gsw_basic\"><caption>(.*?)\(', str(soup))

box = soup.find_all(id="div_box_lal_basic")

# note: need to change 'box' object into a string, since it's currently a BeautifulSoup' ResultSet object, which is useless to us
box = str(box)

# If we open up the box object as a text file, we can start to glean information about how the data is encoded. For example, we can get all the headers by doing a regex search for text starting with `col">`. 



# Looking closer at the box object, we can see that the column headers _and_ the stats we seek are located inside a <data stat> tag:

# Therefore, we can collect all of our data using regular expressions.


stats = re.findall('data-stat=.+?>(.+?)<',box)

# We need to clean up the data. First, let's delete the first item in the list, since that's just the table name:

del stats[0]

# Currently, our player names contain HTML encoding, so we will want to clean that up with with regular expressions.

for i, m in enumerate(stats):
    if len(re.findall('html">(.+)',str(m))) > 0:
        stats[i] = re.findall('html">(.+)',str(m))[0]


len_column_headers = len([th.getText() for th in 
                  soup.findAll('tr', limit=2)[1].findAll('th')])

a = []
a.append(stats.index('Did Not Dress'))

b = []
for i in a:
    b.append(i)
    b.append(i-1)

for index in sorted(b, reverse=True):
    del stats[index]


# Now we want to iterate through the list and group (i.e. subset into separate lists) the stats by their proper rows. 

stats_clean = [stats[i:i+len_column_headers] for i in range(0, len(stats), len_column_headers)]

teamid = re.findall('div_box_(.*?)_basic', 'div_box_lal_basic')[0]
team = re.findall('id=\"box_'+teamid+'_basic\"><caption>(.*?)\(', str(soup))[0].strip()
stats_clean[0].insert(0,'Team')
# Now our stats are clean and ready to go. Our last step for this exercise will be to put the stats into Pandas DataFrame in order to resemble a true, clean box score. Although tehre is still cleanup that needs to be done (removing the reserves header column, as well as cleaning up percentages for non-shooters), we have done the majority of the work with a relatively few steps.


pd.DataFrame(stats_clean[1:], columns=stats_clean[0])
