import time
from pathlib import Path
import sqlite3
from functools import wraps


def sleep_exec(func):

    @wraps(func)
    def wrapper(*args):

        attempt = 0
        max_attempts = 50

        while True:

            if attempt >= max_attempts:
                break

            try:
                func(*args)
                break
            except sqlite3.OperationalError:
                time.sleep(5)

            attempt += 1

        return func(*args)

    return wrapper


class TuyauDataBase(object):

    def __init__(self, database_file):
        self.database_file = database_file

    def is_file(self):
        return Path(self.database_file).is_file()

    def has_data(self):
        filesize = Path(self.database_file).stat().st_size
        return filesize > 0
        
    def remove(self):

        if self.is_file():
            Path(self.database_file).unlink()

    def view(self):

        with sqlite3.connect(self.database_file) as conn:

            c = conn.cursor()

            for row in c.execute('SELECT * FROM images ORDER BY grid'):
                print(row)

    def view_grid(self, grid):

        with sqlite3.connect(self.database_file) as conn:

            c = conn.cursor()

            for row in c.execute("SELECT * FROM images WHERE grid = ? ORDER BY grid", (str(grid),)):
                print(row)

    @property
    def table_exists(self):

        def check_func(db_file):

            with sqlite3.connect(db_file) as conn:

                c = conn.cursor()

                c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='images'")

                if c.fetchone()[0] == 1:
                    exists_ = True
                else:
                    exists_ = False

            return exists_

        return check_func(self.database_file)

    def create(self, exists_ok=False):

        """
        Creates a database table
        """

        if not exists_ok and self.is_file():
            raise OSError('The database file already exists.')

        if not self.table_exists:

            def commit_func(db_file):

                with sqlite3.connect(db_file) as conn:

                    c = conn.cursor()
                    c.execute('''CREATE TABLE images (grid text, step text, status text)''')

            commit_func(self.database_file)

    def insert(self, grid, force_step=None):
        
        """
        Inserts rows for a grid
        """

        def commit_func(db_file, proc_steps):

            with sqlite3.connect(db_file) as conn:

                c = conn.cursor()
                c.executemany('INSERT INTO images VALUES (?,?,?)', proc_steps)
                conn.commit()

        if force_step:

            steps = [(str(grid), force_step, 'n'),]

            commit_func(self.database_file, steps)

        else:

            with sqlite3.connect(self.database_file) as conn:

                c = conn.cursor()

                c.execute("SELECT step,status FROM images WHERE grid = ?", (str(grid),))
                data = c.fetchall()

            steps = [(str(grid), step, 'n') for i, step in enumerate(['preprocess',
                                                                      'mask',
                                                                      'reconstruct',
                                                                      'reindex',
                                                                      'classify',
                                                                      'segment']) if not data or step not in data[i]]

            if steps:
                commit_func(self.database_file, steps)

    def update(self, grid, step):

        """
        Updates a row for a grid and step
        """

        def commit_func(db_file, proc_grid, proc_step):

            with sqlite3.connect(db_file) as conn:

                c = conn.cursor()
                c.execute("UPDATE images SET status = 'y' WHERE grid = ? AND step = ?", (str(proc_grid), proc_step))

        commit_func(self.database_file, grid, step)

    def reset(self, grid, step):

        """
        Resets a row for a grid and step
        """

        def commit_func(db_file, proc_grid, proc_step):

            with sqlite3.connect(db_file) as conn:

                c = conn.cursor()
                c.execute("UPDATE images SET status = 'n' WHERE grid = ? AND step = ?", (str(proc_grid), proc_step))

        commit_func(self.database_file, grid, step)

    def eosvault_is_complete(self, satellite, asset):

        """
        Check for eosvault downloads
        """

        with sqlite3.connect(self.database_file) as conn:

            c = conn.cursor()

            c.execute("SELECT status FROM images WHERE satellite = ? AND asset = ?", (satellite, asset))
            status = c.fetchone()

        if status:
            return True if status[0] == 'y' else False
        else:
            return False

    def is_complete(self, grid, step):

        """
        Gets the status of a grid and step
        """

        with sqlite3.connect(self.database_file) as conn:

            c = conn.cursor()

            c.execute("SELECT status FROM images WHERE grid = ? AND step = ?", (str(grid), step))
            status = c.fetchone()

        if status:
            return True if status[0] == 'y' else False
        else:
            return False
