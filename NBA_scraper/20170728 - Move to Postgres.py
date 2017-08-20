# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:27:57 2017

@author: User
"""
import pandas as pd
from sqlalchemy import create_engine
import os

engine = create_engine('mysql+pymysql://'+os.environ['dbuser']+':'+os.environ['pw']+'@z1.cdhwgvcyc4xh.us-west-1.rds.amazonaws.com/Z1')

pd.read_sql('select count(*) from Z1.nba_box_scores_detail', con=engine)

nba_box_scores_detail = pd.read_sql('select * from Z1.nba_box_scores_detail', con=engine)

nba_box_scores_detail = nba_box_scores_detail.rename(columns={'3P%':'3PP', 'FG%':'FGP', 'FT%':'FTP'})


engine2 = create_engine('postgresql://postgres:Happy123!@localhost')

pd.read_sql('select * from pg_catalog.pg_tables;', con=engine2)

nba_box_scores_detail.to_sql(name = 'nba_box_scores_detail', con=engine2, if_exists='replace',chunksize=1000, index=False)

tables = ['player_career_totals', 'player_season_totals', 'team_game_totals', 'team_season_totals']

for i in tables:
    query = 'select * from Z1.'+i
    table = pd.read_sql(query, con=engine)
    table = table.rename(columns={'3P%':'3PP', 'FG%':'FGP', 'FT%':'FTP'})
    table.to_sql(name = i, con=engine2, if_exists='replace',chunksize=1000, index=False)
    
nba_games_scores = pd.read_sql('select * from Z1.nba_games_scores', con=engine)
nba_games_scores.to_sql(name = 'nba_games_scores', con=engine2, if_exists='replace',chunksize=1000, index=False)
