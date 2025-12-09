import os, sys
import pandas as pd
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pandas only supports SQLAlchemy connectable.*"
)

from dotenv import load_dotenv
import mysql.connector

# загружаем .env из корня проекта
from dotenv import load_dotenv
load_dotenv()

def getPath():
    curdir = os.getcwd()+ '/'  # текущая дериктория
    list_dir = curdir.replace(os.getenv("BASE_DIR"), '|').split('|')
    root_path = os.path.join(list_dir[0], os.getenv("BASE_DIR") + '/')
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if curdir not in sys.path:
        sys.path.insert(0, curdir)

    return root_path, curdir

def sysInsertPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_path, curdir = getPath()
root_in, root_out = root_path + 'data_in/', root_path + 'data_out/'




########################################




class SqlFunctions():

    def __init__(self):
        pass

    def getSqlConnection(self, database=False):

        db = database or os.getenv("MYSQL_DB")

        sql_conn = mysql.connector.connect(
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=db,
            auth_plugin=os.getenv("MYSQL_AUTH_PLUGIN", "mysql_native_password"),
            connection_timeout=int(os.getenv("MYSQL_CONNECT_TIMEOUT", "600")),
            autocommit=os.getenv("MYSQL_AUTOCOMMIT", "true").lower() == "true",
        )


        return sql_conn


    #======= dataframe

    def uploadSqlTab(self, name_sql_tab, sqlite_connection = False, one_connection = True, print_report = False):
        # загрузка всех данных без фильтрации из SQL
        df, report = 0, 'Upload Error'

        try:
            if not sqlite_connection:
                sqlite_connection = self.getSqlConnection()
                print(sqlite_connection)
            postgresql_select_query = 'select * from ' + name_sql_tab
            SQL_Query = pd.read_sql_query(postgresql_select_query, sqlite_connection)



            df = pd.DataFrame(SQL_Query)
            report = 'Upload Success_' + postgresql_select_query + '_' + str(len(df))

            if print_report: print(report)


        except Exception as e:
            print(f'\n Error uploadAllSqlTab = {e} \n')


        finally:
            if (sqlite_connection)and(one_connection):
                sqlite_connection.close()


        return df

    def dfSqlFreeInsert(self, name_sql_tab, df, sqlite_connection=False, one_connection=True):
        '''
        Добавляет DataFrame - все столбцы. Если столбцы с SQL не совпадает - выдаст исключение
        '''
        try:
            columns = list(df.columns)
            col_line = ''
            for col in columns:
                col_line = col_line + ', ' + col
            col_line = col_line[2:]

            if not sqlite_connection:
                sqlite_connection = self.getSqlConnection()
            cursor = sqlite_connection.cursor()

            var_before = ''
            for i in range(len(df)):  # ['log_ID'].count()
                val_line = ''
                for col in columns:
                    val = df.loc[i, col]
                    val_line = val_line + ', ' + '\'' + str(val) + '\''
                val_line = val_line[2:]
                var_before = var_before + f', ({val_line})'
                if i == 0:
                    var_before = var_before[2:]


            sql_insert_query = 'INSERT INTO ' + name_sql_tab + ' (' + col_line + ') VALUES ' + var_before + ';'
            # вставляем данные
            # выполняем запрос к базе

            cursor.execute(sql_insert_query)

            # коммит изменений
            sqlite_connection.commit()
            cursor.close()


        except Exception as e:
            print(f"Error dfSqlFreeInsert: {e}")

        finally:
            if (sqlite_connection)and(one_connection):
                sqlite_connection.close()

    def stepDfInsert(self, name_sql_tab, df, sqlite_connection=False, one_connection=True):
        df.reset_index(drop=True, inplace=True)
        try:
            if not sqlite_connection:
                sqlite_connection = self.getSqlConnection()

            list_df = self.sliceDf(df, MAX_LEN_DF=500)
            for idf in list_df:
                self.dfSqlFreeInsert(name_sql_tab=name_sql_tab, df=idf, sqlite_connection=sqlite_connection, one_connection=False)

        except Exception as e:
            print(f'ERROR SQL stepDfInsert {e}')

        finally:
            if (sqlite_connection)and(one_connection):
                sqlite_connection.close()


    def sliceDf(self, df, MAX_LEN_DF):
        ''' Нарезка DataFrame на куски MAX_LEN_DF, результат помещаем в список df_list'''
        df_list = []
        for i in range(0, len(df), MAX_LEN_DF):
            idf = df[i:i + MAX_LEN_DF].reset_index(drop=True)
            df_list.append(idf)
        return df_list

        # утилиты

    def updateProxy(self, source = 'proxy.txt'):
        if type(source) == str:
            with open(source, "r", encoding="utf-8") as f:
                proxy_lst = [line.strip() for line in f if line.strip()]

                df = pd.DataFrame()
                df['proxy'] = proxy_lst
                self.stepDfInsert('proxy_base', df)


