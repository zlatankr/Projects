"""Utility functions for working with module sql files."""

import os

SQL_PATH = os.path.join(os.path.dirname(__file__))


def get_sql_as_string(sql_basename):
    """Given a basename X read the file sql/X.sql and return the contents as a string."""
    sql_filename = '{}.sql'.format(sql_basename)
    sql_filepath = os.path.join(SQL_PATH, sql_filename)
    with open(sql_filepath, 'r') as sql_file:
        return sql_file.read()


def process_sql_template(sql_basename, **kwargs):
    """Given a basename X read and process sql/X.template.sql and return contents as string.

    Arguments:
        sql_basename: str, the base name for the sql file. X -> sql/X.template.sql
        kwargs: dict, keyword arguments to be passed to .format on the template.
    """
    sql_template_basename = '{}.template'.format(sql_basename)
    sql_template = get_sql_as_string(sql_template_basename)
    return sql_template.format(**kwargs)
