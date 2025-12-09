# bootstrap.py

import sys, os
from pathlib import Path
import pandas as pd
import datetime
from selenium.webdriver.common.by import By


# === Расчёт путей ===

# def getPath():
#     curdir = os.getcwd()+ '/'  # текущая дериктория
#     list_dir = curdir.replace('llm_MaxGPT', '|').split('|')
#     root_path = os.path.join(list_dir[0], 'llm_MaxGPT/')
#     if root_path not in sys.path:
#         sys.path.insert(0, root_path)
#     if curdir not in sys.path:
#         sys.path.insert(0, curdir)
#
#     return root_path, curdir
#
# def sysInsertPath(path):
#     if path not in sys.path:
#         sys.path.insert(0, path)
#
# root_path, curdir = getPath()
# root_in, root_out = root_path + 'data_in/', root_path + 'data_out/'

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


# === Инициализация ===
from USEFUL_UTILS.universal_functions import UniversalFunctions
U24 = UniversalFunctions()

from USEFUL_UTILS.parser_function import ParserFunction
PRF = ParserFunction()

from USEFUL_UTILS.sql_functions import SqlFunctions
SQL = SqlFunctions()

from USEFUL_UTILS.brands_and_parts import brandsAndParts
BAP = brandsAndParts()
