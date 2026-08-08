import pymysql
from contextlib import contextmanager
from app.config import settings

@contextmanager
def get_db_connection():
    connection = pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def execute_query(query: str, params: tuple = None, fetch: bool = True):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return cursor.rowcount

def execute_many(query: str, args_list: list):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, args_list)
            return cursor.rowcount
