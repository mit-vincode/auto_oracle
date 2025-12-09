from pathlib import Path

import re, sys, os
import pandas as pd

def getPath():
    curdir = os.getcwd()+ '/'  # текущая дериктория
    list_dir = curdir.replace('llm_MaxGPT', '|').split('|')
    root_path = os.path.join(list_dir[0], 'llm_MaxGPT/')
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


class brandsAndParts():

    def __init__(self):
        self.dict_brand_to_uni = self.dctBrandsToUni()

    def getOElist(self):  # оригинальные бренды
        df = pd.read_excel(root_path + '/DATA_CATALOGS/KeyWords_Xlsx/brands_to_uni.xlsx')
        df['brand_uni'] = df['brand_uni'].astype('str').str.upper()
        df = df[df['oe_aftermarket'] == 'OE']
        return df['brand_uni'].tolist()


    def dctBrandsToUni(self): #к универсальному бренду по прямому совпадению brand -->brand_uni
        df = pd.read_excel(root_path + '/DATA_CATALOGS/KeyWords_Xlsx/brands_to_uni.xlsx')

        #при дубликатов brand - менее популярные brand_uni должны перезатироваться более популярными
        df['brand_power_index'] = df['brand_power_index'].astype(int)
        df.sort_values(['brand_power_index'], inplace=True)

        df['brand'] = df['key_brand'].astype('str').str.upper()
        df['brand'] = df['brand'].str.replace('[^a-zA-Zа-яА-Я0-9]', '', regex=True)
        df.drop_duplicates(subset='brand', keep='first', inplace=True)
        dict_brand_to_uni = pd.Series(df['brand_uni'].values, index=df['brand'].values).to_dict()

        #для страховки добавляем кросс brand_uni на самого себя
        dict_brand_to_uni.update({re.sub(r'[^a-zA-Zа-яА-Я0-9]', '', str(x)):str(x) for x in df['brand_uni'].values})
        return dict_brand_to_uni


    def brandOrFalse(self, ss):
        ss = U24.clearWaste(ss, type_waste = 'clear_text')
        ss = ss.upper()
        ss = re.sub(r'[^a-zA-Zа-яА-Я0-9]', '', ss)
        if ss not in self.dict_brand_to_uni:
            return False
        return str(self.dict_brand_to_uni[ss]).upper()

    def dfBrandColToUni(self, df, brand_col = 'brand', drop_no_brand = True,  rename_uni = True):
        """ быстрый поиск универсальных брендов в df  - выбор по точному соответсвию"""
        df[brand_col] = df[brand_col].astype('str').str.upper()
        df['brand_uni'] = df.apply(lambda x: self.brandOrFalse(x[brand_col]), axis = 1)

        if drop_no_brand:
            df = df[df['brand_uni'] != False].reset_index(drop=True)

        if rename_uni:
            del df[brand_col]
            df.rename(columns={'brand_uni': brand_col}, inplace=True)


        return df