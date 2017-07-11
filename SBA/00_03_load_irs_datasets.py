"""
Load IRS Zip code data
Source of data:
https://www.irs.gov/uac/soi-tax-stats-individual-income-tax-statistics-2014-zip-code-data-soi

Documentation of columns:
https://www.irs.gov/pub/irs-soi/14zpdoc.doc
"""

import argparse

import pandas as pd

from utilities.db_manager import DBManager


def get_args():
    """Use argparse to parse command line arguments."""
    parser = argparse.ArgumentParser(description='Runner for tasks')
    parser.add_argument('--db_url', help='Database url string to the db.', required=True)
    return parser.parse_args()


def load_irs_data(dbm, direc):
    """Load census datasets
    Keyword Args:
        dbm: DBManager object
        direc: Directory where files are
    """
    df = pd.read_csv(direc + '14_irs_zip.csv')
    dbm.write_df_table(
        df, table_name='irs_zip_data', schema='data_ingest')


def main():
    """Execute Stuff"""
    print('Parsing Census datasets')
    args = get_args()
    dbm = DBManager(db_url=args.db_url)
    directory = 'C:/Users/User/Dropbox/Documents/Analytics/Analyses/SBA/data/'
    load_irs_data(dbm, directory)


if __name__ == '__main__':
    """See https://stackoverflow.com/questions/419163/what-does-if-name-main-do"""
    main()