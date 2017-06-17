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
            if (stats[j+1][:7] <> 'Did Not' and stats[j][:7] <> 'Did Not') and (stats[j+1][:8] <> 'Not With' and stats[j][:8] <> 'Not With') and (stats[j+1][:6] <> 'Player' and stats[j][:6] <> 'Player'):
                new.append(stats[j])
        new.append(stats[-1])
        stats_clean = [new[u:u+len_column_headers] for u in range(0, len(new), len_column_headers)]
        teamid = re.findall('div_box_(.*?)_basic', i)[0]
        team = re.findall('id=\"box_'+teamid+'_basic\"><caption>(.*?)\(', str(soup))[0].strip()
        print stats_clean
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
    boxes['Feature'] = (re.findall('<meta content="Box Score -(.+?)" name="Description">', str(soup)))[0]
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
count = 23074
for i in all_box_urls[23074:]:
    print count,'of 42100'
    try:
        box = box_scores(i)
    except:
        print 'error with', i
        break
    try:
        all_data = pd.concat((all_data, box))
    except:
        print 'error concating with', i
    count += 1
    

all_data.to_sql(name = 'nba_box_scores', con=engine, if_exists='replace',chunksize=1000)


start = timeit.default_timer()
page = requests.get(all_box_urls[1])
soup = BeautifulSoup(page.content, 'html.parser')


games = []
for i in range(len(all_box_urls)):
    page = requests.get(all_box_urls[i])
    soup = BeautifulSoup(page.content, 'html.parser')
    games.extend(re.findall('<meta content="Box Score -(.+?)" name="Description">', str(soup)))
    print i

import calendar


away = [re.findall('(.+?)\(', a)[0].strip() for a in games]
home = [re.findall('vs.(.+?)\(', a)[0].strip() for a in games]
away_points = [re.findall('\((.+?)\)', a)[0] for a in games]
home_points = [re.findall('\((.+?)\)', a)[1] for a in games]
year = [re.findall('- (.+)', a)[0].split()[2] for a in games]
months = {k:v for v, k in enumerate(calendar.month_name)}
month = [months[re.findall('- (.+)', a)[0].split()[0]] for a in games]
day = [re.findall('- (.+)', a)[0].split()[1][:-1] for a in games]
month_clean = [str(i) if len(str(i)) == 2 else '0' + str(i) for i in month]
day_clean = [i if len(i) == 2 else '0' + i for i in day]
date = [a+b+c for a, b, c in zip(year, month_clean, day_clean)]

all_scores =  pd.DataFrame({'Home': home, 
              'Away': away, 
              'Away_Points': away_points, 
              'Home_Points': home_points, 
              'Game_Date': date, 
              }
                )

def winner(row):
    if int(row['Away_Points']) > int(row['Home_Points']):
        return row['Away']
    else:
        return row['Home']

def loser(row):
    if int(row['Away_Points']) < int(row['Home_Points']):
        return row['Away']
    else:
        return row['Home']

all_scores['Winner'] = all_scores.apply(winner, axis=1)
all_scores['Loser'] = all_scores.apply(loser, axis=1)

all_scores.to_sql(name = 'nba_games_scores', con=engine, if_exists='replace', chunksize=1000)

nba_box_scores = pd.read_sql_table('nba_box_scores', con=engine)

homes = pd.merge(nba_box_scores, all_scores, how='inner', left_on = ['Date', 'Team'], 
         right_on = ['Game_Date', 'Home'])

aways = pd.merge(nba_box_scores, all_scores, how='inner', left_on = ['Date', 'Team'], 
         right_on = ['Game_Date', 'Away'])

both = pd.concat((homes, aways))

both['Win'] = both['Team'] == both['Winner']
both['Win'] = both['Win'].apply(lambda x: int(x))

both.to_sql(name = 'nba_box_scores_detail', con=engine, if_exists='replace',chunksize=1000)
