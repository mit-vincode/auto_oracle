import os, datetime, chardet, csv, string, random, re, time, requests, io
import pandas as pd
from ftplib import FTP
import psutil, math, sys
import numpy as np
from pathlib import Path


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



class UniversalFunctions():

    def __init__(self):
        ### обработка текста - буквы ##########
        self.ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

        self.rusalp_str = 'абвгдеёжзийклмнопрстуфхцшщъыьэюя'
        self.rusalp_lower = [x for x in self.rusalp_str]
        self.rusalp_upper = [x for x in self.rusalp_str.upper()]

        ######## Alphabet  - алфавиты + цифры ####
        self.alphabet_rus_lower = ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п',
                                   'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
        self.alphabet_rus_upper = [l.upper() for l in self.alphabet_rus_lower]

        # English - буквы
        self.alphabet_eng_lower = list(chr(i) for i in range(97, 123))
        self.alphabet_eng_upper = list(chr(i) for i in range(65, 91))
        # цифры
        self.alphabet_dig = list(chr(i) for i in range(48, 58))
        # всё вместе
        self.alphabet_eng_dig_rus = self.alphabet_eng_lower + self.alphabet_eng_upper + self.alphabet_dig + self.alphabet_rus_lower + self.alphabet_rus_upper
        self.alphabet_eng_dig_rus_SMB = self.alphabet_eng_dig_rus + [' ', '.', ',', ';', '-', '₽', '/', '%',  '|', '&', '_', '=',
                                            '*', ')', '(', '+', '!', ':', '[', ']', '>', '<']

        self.car_oil_reject_lst = ['периодическое использование', '80w-90', ' 4t ', ' sae 30 ', '2т для', '4-х тактных',
                      'технологическое', '2т двигател', '2т al-ko', 'бензопил', 'не подлежит классификации по sae',
                      ' 2т ', '2-х тактное', 'снегоход', '2-takt', 'садовов', 'двухтактных',
                      'интенсивное использование', '4-takt', ' для 2-х ', 'motorcycle', ' 4-t ', 'трансмиссионное',
                      'триммер', '4т al-ko', ' 2t ']


        self.sql_crosses_24 = 'crosses_24'
        self.col_crosses_24 = ['root_key', 'cross_bart', 'name', 'cross_tovgruppa', 'cross_global_group', 'price']
        self.col_cr24_brd_art = self.col_crosses_24 + ['brand', 'art']
        self.col_crosses_24_id = ['root_key', 'cross_bart', 'name', 'cross_tovgruppa', 'cross_global_group', 'price', 'id']

        self.col_pars_res = ["task_bart", "cross_bart", "cross_tovgruppa", "cross_global_group", "name", "price", "target", "url", "spare", "ttime"]

        #текстовые реквизиты
        self.cross_bart = 'cross_bart'
        self.root_key = 'root_key'
        self.cross_tovgruppa = 'cross_tovgruppa'
        self.cross_global_group = 'cross_global_group'
        self.typ_id = 'typ_id'
        self.carpark_sql = 'carpark'
        self.control = 'control'


        #популярные символы
        self.PopSmb_rub = '₽'

    def strEngDigRus(self, ss, sep=" ", add_smb_list=False, smb_list=False, punctuation_marks = False):
        """smb_list  - основные символы
        alphabet_eng_dig_rus - возвращает неизменённый текст Eng + Rus + Dig разделённый пробелами (sep=" ")"""
        smb_list = self.alphabet_eng_dig_rus.copy() if not smb_list else smb_list
        if type(add_smb_list) is not bool: smb_list += add_smb_list
        if punctuation_marks:
            smb_list += ['.', ',', '-', '/', '_', '(', ")", "+", '"', '—', '–', ':', '°', '\n']

        ss = ss.replace('\xa0', sep).replace('\r', sep).replace('<br>', sep).replace('<br/>', sep).replace('<br />', sep)
        res = ""
        for s in str(ss):
            if s in smb_list:
                res += s
            else:
                res += sep
        res = sep.join(res.split())

        return res

    def strDig(self, ss, sep=""):
        """неизменённый текст Dig 1-2-3 разделённый пробелами"""
        smb_list = self.alphabet_dig
        res = ""
        for s in str(ss):
            if s in smb_list:
                res += s
            else:
                res += sep
        res = sep.join(res.split())

        return res

    def strRus(self, ss, sep=" ", add_smb_list=False):
        """неизменённый текст Rus разделённый пробелами"""
        smb_list = self.alphabet_rus_lower + self.alphabet_rus_upper
        if type(add_smb_list) is not bool: smb_list += add_smb_list
        res = ""
        for s in str(ss):
            if s in smb_list:
                res += s
            else:
                res += sep
        res = sep.join(res.split())

        return res

    def strEng(self, ss, sep=" ", add_smb_list=False):
        """неизменённый текст Eng разделённый пробелами"""
        smb_list = self.alphabet_eng_lower + self.alphabet_eng_upper
        if type(add_smb_list) is not bool: smb_list += add_smb_list
        res = ""
        for s in str(ss):
            if s in smb_list:
                res += s
            else:
                res += sep
        res = sep.join(res.split())

        return res

    def strEngRus(self, ss, sep=" ", add_smb_list=False):
        """неизменённый текст Eng разделённый пробелами"""
        smb_list = self.alphabet_eng_lower + self.alphabet_eng_upper + self.alphabet_rus_lower + self.alphabet_rus_upper
        if type(add_smb_list) is not bool: smb_list += add_smb_list
        res = ""
        for s in str(ss):
            if s in smb_list:
                res += s
            else:
                res += sep
        res = sep.join(res.split())

        return res

    def strDig_one_Only(self, ss):
        """ в строке только 1 число (например, для выделения остатков: Осталось 6 шт)"""
        ss = ss.split()
        ss = [''.join([x for x in xx if x in self.alphabet_dig]) for xx in ss]
        ss = [x for x in ss if x] #чистим ['', '6', '']
        if len(ss) > 1 or not ss: return False
        return ss[0]

    def getStrDateTime(self):
        return '_'.join(str(datetime.datetime.now()).split('.')[0].split(' ')).replace(':', '-') + '_'

    def getRandCol(self, col='xxx'):  # рандомное имя колонки, чтобы исключить конфликт с существующими
        return f"{col}={random.randint(1000, 100000)}{self.tNow()}"

    def mark(self, info=''):
        print(f'#mark {info}') # marker временных строк (для разработки)

    def printList(self, count_data):
        len_data = len(count_data)
        print_list = [i for i in range(len_data) if i % (int(len_data / 20) + 1)  == 0]
        return print_list, len_data

    def printInfo(self, i, print_list, len_data, name_loop):
        print(f'{name_loop} - {int(100 * round(i / len_data, 2))}% пройдено, {self.getStrDateTime()}') if i in print_list else 0


    #########   загрузка файлов данных ################

    def notDataFile(self, f):
        """НЕ - DataFile --> True"""
        if "~$" in f: return True
        if ('.xlsx' in f)or('.txt' in f)or('.csv' in f):
            return False
        return True

    # def import_csv(self, path):
    #     try:
    #         df = pd.read_csv(path, encoding='utf8', sep=';')
    #     except:
    #         try:
    #             df = pd.read_csv(path, encoding='utf8', sep=';', error_bad_lines=False)
    #         except:
    #             rawdata = open(path, 'rb').read()
    #             result = chardet.detect(rawdata)
    #             encoding = result['encoding']
    #             # Определение разделителя (delimiter)
    #             with open(path, 'r', encoding=encoding) as file:
    #                 dialect = csv.Sniffer().sniff(file.read(1024))
    #                 separator = dialect.delimiter
    #             # Чтение файла с определенной кодировкой и разделителем
    #             df = pd.read_csv(path, encoding=encoding, sep=separator, error_bad_lines=False)
    #
    #     return df

    def import_csv(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        # Быстрые попытки — покрывают 99.9% русских CSV
        attempts = [
            ('cp1251', ';'),
            ('utf-8-sig', ';'),
            ('utf-8', ';'),
            ('cp1251', ','),
            ('utf-8-sig', ','),
        ]

        for enc, sep in attempts:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python', on_bad_lines='skip')
                if df.shape[1] > 1:
                    return df
            except:
                continue

        # Если всё плохо — полный поиск (редко нужен)
        raw = path.read_bytes()[:100000]  # первые 100 КБ
        enc = chardet.detect(raw)['encoding'] or 'utf-8'

        for sep in ';\t,|':
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python', on_bad_lines='skip')
                if df.shape[1] > 1:
                    return df
            except:
                continue

        raise ValueError(f"Не удалось прочитать CSV: {path}")

    def export_csv(self, df, to_path):
        if not to_path.endswith('.csv'): to_path += '.csv'
        df.to_csv(to_path, encoding='utf_8_sig', sep=';', index=False)

    def uploadDataFile(self, in_file):
        type_file = in_file.split('.')[-1]
        if type_file == 'xlsx':
            df = pd.read_excel(f'{in_file}')
        elif (type_file == 'txt') or (type_file == 'csv'):
            df = self.import_csv(in_file)
        else:
            df = []
        return df

    #загрузка tmp --> df
    def tmp_uploadIn(self, file_type = 'xlsx'):
        path_from = root_in + f'tmp.{file_type}'
        df = self.uploadDataFile(path_from)
        print(f'IN len(df) = {len(df)}, path_from = {path_from}')
        return df

    def tmp_uploadOut(self, file_type = 'xlsx'):
        path_from = root_out + f'tmp.{file_type}'
        df = self.uploadDataFile(path_from)
        print(f'IN len(df) = {len(df)}, path_from = {path_from}')
        return df

    #сохранение df --> tmp
    def tmp_saveIn(self, df):
        print(f'OUT  len(df) = {len(df)}, path_to = {root_in}')
        return self.xlsxSave(df, root_in)
    def tmp_saveOut(self, df):
        print(f'OUT  len(df) = {len(df)}, path_to = {root_out}')
        return self.xlsxSave(df, root_out)


    def uploadDirectory2Df(self, path, max_len_df = 30*1000*1000, return_file_list = False):
        """ много файлов из директории набирает в 1 df,
        + возвращает лист с именами файлов"""
        files = os.listdir(path)
        file_list, df = [], pd.DataFrame()
        for i, file in enumerate(files):
            if self.notDataFile(file):
                continue
            file_list.append(file)
            if "~$" in file: continue

            print(file)
            try:
                df = pd.concat([df, self.uploadDataFile(path + file)])
            except: continue

            print(f"{i}/{len(files)}, len(df) = {len(df)}, {file}")
            if len(df) > max_len_df:
                df.reset_index(drop=True, inplace=True)
                if return_file_list: return df, file_list
                return df


        df.reset_index(drop=True, inplace=True)
        if return_file_list: return df, file_list
        return df

    def data2Df_upload(self, path, max_len_df=30 * 1000 * 1000, return_file_list=False) -> pd.DataFrame:
        lst = ['.xls', '.csv', '.txt']
        if any([x in path for x in lst]): return self.uploadDataFile(path)
        return self.uploadDirectory2Df(path, max_len_df, return_file_list)

    def dirs2Df(self, dir_base):
        """18/12/23 - директория со вложенными папками в df"""
        folders = [entry.name for entry in os.scandir(dir_base) if entry.is_dir()]
        df = pd.DataFrame()
        for ff in folders:
            df = pd.concat([df, self.data2Df_upload(dir_base + ff +'/')])

        df.reset_index(drop=True, inplace=True)

        return df



    def xlsxSave(self, df, to_path = root_out, sheet_name = 'Sheet1'):
        if (len(df) > 1048570) or (sys.platform == 'linux'):
            print(f"Attention, saved CSV, not xlsx! len df = {len(df)}, max len = 1048570 (1048576)")
            to_path = to_path.replace(".xlsx", '.csv')
            if to_path == root_in or to_path == root_out: to_path += 'tmp.csv'
            if '.csv' not in to_path: to_path += '.csv'
            self.export_csv(df, to_path)
            return
        if to_path == root_in or to_path == root_out: to_path += 'tmp.xlsx'
        if '.xlsx' not in to_path: to_path += '.xlsx'

        # #удаляем спец-символы вида \x16 из-за которых excel вылетает
        # df = df.applymap(lambda x: self.ILLEGAL_CHARACTERS_RE.sub(r'', x) if isinstance(x, str) else x)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].map(lambda x: self.ILLEGAL_CHARACTERS_RE.sub(r'', x) if isinstance(x, str) else x)

        df.to_excel(to_path, sheet_name=sheet_name, index=False)

    def controlPath(self, path):
        """ безопасное создание /  контроль директории"""
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def pathFromRoot(self, path):
        """ безопасное создание /  контроль директории"""
        if not os.path.exists(root_path + path):
            os.mkdir(root_path + path)
        return root_path + path

    def getStart(self, path, sep = '_'):
        files = os.listdir(path)
        extensions = ['.xlsx', '.csv', '.jpg', '.png' ]
        files = [int(f.split(sep)[0]) for f in files if any(ext in f for ext in extensions) and "~$" not in f]
        if not files: return 0
        return max(files) + 1

    def fileList_intervalDays(self, path_from, min_days, max_days):
        """ список файлов в интервале дней"""
        min_sec, max_sec = min_days * 24 * 3600, max_days * 24 * 3600
        files = os.listdir(path_from)
        file_list = []
        for f in files:
            if self.notDataFile(f): continue
            file_date = datetime.datetime.fromtimestamp(os.path.getctime(path_from + f))
            if ((datetime.datetime.now() - file_date).total_seconds() < min_sec) or (
                    (datetime.datetime.now() - file_date).total_seconds() > max_sec):
                continue
            file_list.append(f)

        return file_list

    ###### END загрузка файлов данных ################

    ########## FTP  - загрузка / сохранение

    def longTimeAgo(self, in_file):
        """ контроль достаочного временного лага, чтобы исключить колизии при перезаписи файлов"""
        LAG_SEC = 60
        file_date = datetime.datetime.fromtimestamp(os.path.getctime(in_file))
        if ((datetime.datetime.now() - file_date).total_seconds() < LAG_SEC):
            return False #файл не созрел
        return True #файл созрел



     ############ END FTP - загрузка / сохранение   ################




    def tNow(self):
        return datetime.datetime.now()

    def calcTimeNextLoop(self, step_sec):  # время следующего цикла
        return datetime.datetime.now() + datetime.timedelta(0, step_sec)

    def startTimeControl(self, script, period_start_sec = 1 * 24 * 3600):  # 1 раз в 1 день: 1 * 24 * 3600
        dir_start = self.controlPath(root_path + '/useful_utils/dir_start/')
        path_start_csv = dir_start + f'{script}.csv'

        if not os.path.exists(path_start_csv):
            """ start!"""
            self.export_csv(pd.DataFrame({'dtime': [self.tNow()]}), path_start_csv)
            return True

        else:
            dtime =  self.uploadDataFile(path_start_csv).loc[0, 'dtime']
            dtime = datetime.datetime.strptime(dtime, '%Y-%m-%d %H:%M:%S.%f')

            last_start_sec = (self.tNow() - dtime).total_seconds()
            if last_start_sec < period_start_sec:
                return False
            else:
                """ start!"""
                self.export_csv(pd.DataFrame({'dtime': [self.tNow()]}), path_start_csv)
                return True





    ###### обработка результатов парсинга ###############
    def targetFromStr(self, ss, del_target_list=False):
        """ target из строки. Если найдено несколько target - вернёт False. Можно часть ненужных target отбросить  - del_target_list"""
        target_list = ['emex', 'ixora', 'exist', 'autopiter', 'euroauto', 'rossko', 'autoeuro', 'wildberries',
                       'yamarket', 'ozon', 'avito', "vincodrf", 'auto3n', 'topdetal', 'automig',
                       'autoopt', 'baltkam', 'opex', 'forum-auto', 'autorus', 'smartec', 'berg', 'sparox', 'avtoalfa', 'nitauto', 'drom',
                       'froza', 'autodoc']
        if del_target_list: #чистка ненужных target
            for target in del_target_list:
                if target in target_list: target_list.remove(target)

        ss = ss.lower()
        res_list = []
        for target in target_list:
            if target in ss: res_list.append(target)
        if not res_list: return False  # нет известных target
        if len(res_list) > 1: return False  # 2и более - не понятно, какую выбирать
        return res_list[0]

    def targetFromUrl(self,url):
        target_list = ["emex", "autopiter", 'ixora', 'auto3n', 'exist', 'rossko', 'autoopt', 'baltkam',
                       'sparox', 'avtoalfa', 'nitauto', 'drom', 'froza', 'autodoc']
        for target in target_list:
            if target in url: return target
        return False

    def daysBeforeFromStr(self, ss, dformat="%Y-%m-%d_%H-%M-%S"):
        """кол-во дней от текущей даты до даты в названии файла (назад)
           ss = '2100_avtp2_2023-04-07_22-21-04_res_ixora.xlsx'"""
        for year in ['2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033']:
            if f"{year}-" in ss:
                end_year = ss.find(year) + 3
                if ss[end_year + 1] != "-":
                    ss = ss[end_year + 1:]
                    self.daysBeforeFromStr(ss, dformat)

                ss = ss.replace(year, "|").split("|")[1]
                break
        ss = ss.split("_")
        mday, ttime = ss[0], ss[1]
        dtime = datetime.datetime.strptime(f'{year}{mday}_{ttime}', dformat)
        days_before = (self.tNow() - dtime).days + 1

        return days_before

    def daysBack_24Ss(self, ss, dformat="%Y-%m-%d"):
        """Приемник: daysBeforeFromStr - не учитывает время - только дату (2023-04-07_22-21-04)
        кол-во дней от текущей даты до даты в названии файла (назад)
           ss = '2100_avtp2_2023-04-07_22-21-04_res_ixora.xlsx'"""

        for year in ['2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033']:
            if f"{year}-" in ss:
                end_year = ss.find(year) + 3
                if ss[end_year + 1] != "-":
                    ss = ss[end_year + 1:]
                    self.daysBack_24Ss(ss, dformat)

                ss = ss.replace(year, "|").split("|")[1]
                break
        ss = ss.split("_")
        mday = ss[0]
        dtime = datetime.datetime.strptime(f'{year}{mday}', dformat)
        days_before = (self.tNow() - dtime).days + 1

        return days_before


    def dtimeFromStr(self, ss, dformat="%Y-%m-%d_%H-%M-%S"):
        """кол-во дней от текущей даты до даты в названии файла (назад)
           ss = '2100_avtp2_2023-04-07_22-21-04_res_ixora.xlsx'"""
        for year in ['2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033']:
            if f"{year}-" in ss:
                end_year = ss.find(year) + 3
                if ss[end_year + 1] != "-":
                    ss = ss[end_year + 1:]
                    self.dtimeFromStr(ss, dformat)
                ss = ss.replace(year, "|").split("|")[1]
                break
        ss = ss.split(".")[0]
        ss = ss.split("_")
        mday, ttime = ss[0], ss[1]
        return datetime.datetime.strptime(f'{year}{mday}_{ttime}', dformat)


    def deltaTimeProcess(self, t1) -> str:
        """ печатная info для измерения длительнсоти процесса"""
        sec = (datetime.datetime.now() - t1).total_seconds()
        hour = sec // 3600
        minutes = (sec - hour * 3600) // 60
        return f" delta_time = h: {int(hour)}, min: {int(minutes)}, sec: {int(round(sec - hour * 3600 - minutes * 60, 0))}"

    def tProcessProgress(self, info, T1, send_anyway = False):
        """ телеграм - информер прогресса процесса"""
        self.send_telegram(f"{info}, time_now = {str(datetime.datetime.now()).split(' ')[1].split('.')[0]}, deltaTimeProcess = {self.deltaTimeProcess(T1)}", send_anyway=send_anyway)

    ########  чистка мусора    ################

    def clearWaste(self, ss, type_waste=False):
        new_ss, sep, waste_list = '', ',', ['\\n', '/', '\\', ';', '|',
                                            '\'', '@', '&', '_', '=',
                                            '*', ')', '(', '\'', '+', '!', '~', ':', '#',
                                            '[', ']', '•', '>', '<', "\"", "?"]
        if type_waste == 'price':
            waste_list += [' ', '₽']
            sep = ''
        elif type_waste == 'art':
            waste_list += [' ', '.', ',', '-']
            sep = ''

        elif type_waste == 'clear_text':
            waste_list += ['.', ',', '-']
            sep = ' '

        elif type_waste == "bart":
            sep, waste_list = '', ['\\n', '/', '\\', ';',
                                        '\'', '@', '&', '_', '=',
                                        '*', ')', '(', '\'', '+', '!']


        for s in str(ss):
            s = s if s not in waste_list else sep
            new_ss += s
        return new_ss

    def clearSpecSmb(self, df, col, type_waste=False):
        """чистка df от спецсимволов в колонке col"""

        df[col] = df.apply(lambda x: self.clearWaste(x[col], type_waste=type_waste),
                           axis=1)
        return df

    def textClear(self, ss, sep = ''):
        res = ''
        for x in str(ss):
            res = (res + x) if x in self.alphabet_eng_dig_rus_SMB else (res + sep)
        res = ' '.join(res.split())
        return res

    def lstRejectRows_fromDf(self, df, col_control, reject_lst, get_rejected=False):
        """ 30/11/23 из df удаляет или возвращает строки, содержащие хотя бы 1 значение из reject_lst"""

        regex_pattern = '|'.join(map(re.escape, reject_lst))  # re.escape() используется для экранирования спец-символов (таким как +, *, {} и др)

        if not get_rejected: #чистая df
            return df[~df[col_control].astype(str).str.contains(regex_pattern)].reset_index(drop=True)

        return df[df[col_control].astype(str).str.contains(regex_pattern)].reset_index(drop=True)

    def lstReplaceStrSmb_fromDf(self, df, col_to_replace, replace_lst):
        """ 04/12/23 - чистит символы replace_lst из колонки df[col_to_replace], в отличии от lstRejectRows_fromDf - сами строки df не удаляет"""

        #экранируем спец-символы, типа "." или "|"
        escaped_replace_lst = [re.escape(char) for char in replace_lst]
        replace_pattern = '|'.join(escaped_replace_lst)
        df[col_to_replace] = df[col_to_replace].str.replace(replace_pattern, '', regex=True)
        df[col_to_replace] = df[col_to_replace].apply(lambda x: " ".join(x.split()))

        return df

    def statementLst_inSs(self, ss, statement_lst):
        """ проверка вхождения хотя бы 1 элемента из statement_lst в ss
        ss = 'price_mdk_2023-11- за час 28_17-03-13.csv' statement_lst = ['за час', ' часов']"""
        contains_statement = any(statement in ss for statement in statement_lst)
        if contains_statement: return True
        return False


    def wasteSimpleArt(self, ss, threshold_ratio_len=.9000999, max_len_threshold = 18, min_len_threshold = 3, print_ratio=False):
        """Чистка art - ищет мусорные паттерны ('000') в строке ss возвращает True - мусор, False - сложный art, с малой длиной мусорных паттернов
        threshold_ratio_len, напр = .8 - сумма длин подстрок мусора не менее 80% от длины исходной строки ss
        длина мусора по отношению к длине ss – мб больше 1, тк, например в 2345 - 2 паттерна (234 и 345)"""

        def counWastePatterns(ss, waste_list):
            """ возвращает кол-во найденных паттернов мусора + список паттернов
            например, для ss = '00002345qwe' --> (4, ['234', '345', '000', 'qwe'])"""
            count_waste = 0
            found_patterns = []

            for waste in waste_list:
                start = 0
                while True:
                    index = ss.find(waste, start)
                    if index == -1:
                        break
                    count_waste += 1
                    found_patterns.append(waste)
                    start = index + len(waste)

            return count_waste, found_patterns

        def qweList():
            """список из всех последовательностей по 3 буквы латинского алфавита, которые расположены подряд на клавиатуре вида: ["qwe", ..., asd, ..., vbn...]"""
            # Строки с клавиатуры
            keyboard_rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

            sequences = []
            # Проходим по строкам клавиатуры
            for row in keyboard_rows:
                # Проходим по символам в строке
                for i in range(len(row) - 2):
                    # Берем подстроку из трех символов
                    sequence = row[i:i + 3]
                    sequences.append(sequence)
            # Сортируем список
            sequences.sort()
            return sequences

        waste_list = [
            '123', '234', '345', '456', '567', '789', '890', '098', '987', '876', '765', '654', '543', '432', '321',
            '000', '111', '222', '333', '444', '555', '666', '777', '888', '999',
            '12q', '123qw', '23we4', '34er5', '45rt6', '56ty7', '67yu8', '78ui9', '89io0', '90op', '12was',
            '123qase', '23wsdr', '34edtf', '45rfyg', '56tgfh', '67yghj', '78ujhk', '89iklp', '90ol', 'qwsxz', 'awedxz',
            'qwersfxc', 'ertdgxcvb', 'rtyfhvbn', 'tyugjvnmk', 'yuihklnm', 'uijlm', 'ikop', 'asx', 'zsdc', 'xdfv',
            'cfgb',
            'vghn', 'bhjm', 'njkl']

        aaa_bbb_lst = [char * 3 for char in string.ascii_lowercase]  # ['aaa', 'bbb', 'ccc', ...]
        waste_list += qweList() + aaa_bbb_lst

        ss = str(ss).lower()
        ss = self.clearWaste(ss, type_waste='art')
        if (len(ss) > max_len_threshold) or (len(ss) < min_len_threshold): return True #всё равно мусор - длинных артикулов не бывает
        count_waste = counWastePatterns(ss, waste_list)
        ratio_len = len("".join(count_waste[1])) / len(ss)  # длина мусора по отношению к длине ss – мб больше 1, тк, например в 2345 - 2 паттерна (234 и 345)
        if print_ratio: print(f"ratio_len = {ratio_len}, count_waste = {count_waste}")

        if ratio_len > threshold_ratio_len: return True #мусор
        return False #сложный артикул - НЕ-мусор

    def simpleWasteArtDf(self, df, col_art = 'art', col_brand = False, get_waste = False, threshold_ratio_len=.9000999):
        """get_waste = 'return_all' - возвращает все False (хорошие - НЕ-waste)
        get_waste=False - чистые артикулы без мусора (НЕ - мусор), get_waste=True  - только мусор, return_all  -исходный df c колонкой False/True
        df - контроль столбца col_art на мусорные артикулы simpleWasteArt
        col_brand = 'brand' - спасаем некоторые бренды типа Febi, Elring, GKN с короткими артикулами """
        save_df = pd.DataFrame()
        col_simple_waste = self.getRandCol('simple_waste')
        in_col = df.columns

        if col_brand:
            save_brands = ['BREMBO', 'GKN-LOEBRO-SPIDAN', 'TOPRAN', 'MAXGEAR', 'VOLVO', 'REMSA', 'SIDEM', 'LAUTRETTE', 'DIESEL-TECHNIC', 'CORTECO', 'IMPERGOM', 'AC-DELCO',
                           'BERU', 'ATE', 'COFLE', 'DENSO', 'FEBEST', 'LEMFORDER', 'NK', 'METALCAUCHO', 'JAPKO', 'MANN-FILTER', 'ZEKKERT', 'BOSAL', 'ADRIAUTO', 'CHAMPION',
                           '3F-QUALITY', 'GM', 'OSSCA', 'MONROE', 'AVANTECH', 'MS-MARSHAL', 'SACHS', 'ASVA', 'BRISK', 'LPR-AP', 'A.B.S.', 'AUGER', 'FACET', 'FORD', 'MEAT-AND-DORIA',
                           'KAMAZ', 'BOSCH', 'HOLA', 'OPTIMAL', 'RUVILLE', 'MAPCO', 'JP-GROUP', 'JANMOR', 'ELRING', 'PEUGEOT-CITROEN', 'REINZ', 'VALEO', 'SAMPA-FRENOTRUCK', 'KOLBENSCHMIDT',
                           'HC-CARGO', 'LYNX', 'NRF', 'NGK-NTK', 'MEYLE', 'AL-KO', 'NPW', 'NISSENS', 'KYB', 'FIAT-ALFA-LANCIA', 'LADA', 'DELPHI', 'MECAFILTER', 'KNECHT-MAHLE', 'FEBI-BILSTEIN', 'FAE', 'TSN-TSITRON']

            tmp_bart = self.getRandCol("tmp_bart")
            df[tmp_bart] = df[col_brand].astype(str) + "|" + df[col_art].astype(str)
            save_df = df[df[col_brand].isin(save_brands)]
            if len(save_df):
                save_df[col_simple_waste] = False
                df = df[~df[tmp_bart].isin(save_df[tmp_bart])]
            df, save_df = df.drop(tmp_bart, axis=1), save_df.drop(tmp_bart, axis=1)

        df[col_simple_waste] = df[col_art].apply(lambda x: self.wasteSimpleArt(x, threshold_ratio_len=threshold_ratio_len))
        df = pd.concat([df, save_df])
        df.reset_index(drop=True, inplace=True)

        #waste - в любом случае
        waste_anyway = ['11111111111111', 	'00000000000000', 	'00000000000','11111111111', 	'13131313131', 	'33023502090', 	'21083502090', 	'865114Y000', 	'0000000000', 	'86511M0000', 	'60U807221B', 	'620228143R', 	'1111111111', 	'123456789', 	'560408054', 	'000000001', 	'11111111', 	'00000000', 	'77777777', 	'55555555', 	'00000001', 	'12345678', 	'1111111', 	'0000000', 	'7777777', 	'5555555', 	'1703690', 	'0000001', 	'ISO9001', 	'1234567', 	'1500402', 	'5468544', 	'2222222', 	'324234', 	'6СТ100', 	'000001', 	'121212', 	'112233', 	'123456', 	'000000', 	'75D23L', 	'111111', 	'6СТ225', 	'6СТ110', 	'454545', 	'234234', 	'456985', 	'111112', 	'100000', 	'11111', 	'00000', 	'55555', 	'77777', 	'00001', 	'23424', 	'12345', 	'23423', 	'12356', 	'22222', 	'32423', 	'6СТ90', 	'12456', 	'33333', 	'12334', 	'12355', 	'34234', 	'88888', 	'11112', 	'16949', 	'44444', 	'10000', 	'11122', 	'32424', 	'66666', 	'99999', 	'12234', 	'0000', 	'5555', 	'0001', 	'1233', 	'1122', 	'7777', 	'1235', 	'2222', 	'1000', 	'1212', 	'5678', 	'3333', 	'1112', 	'1245', 	'2023', 	'1223', 	'1345', 	'1234', 	'4444', 	'8888', 	'2180', 	'0123', 	'9999', 	'1222', 	'1123', 	'1256', 	'4545', 	'5677', 	'3163', 	'1224', 	'3455', 	'4566', 	'HDK2', 	'4216', 	'0101', 	'1334', 	'1211', 	'1232', 	'1344', 	'3567', 	'6789', 	'1244', 	'2323', 	'2456', 	'4578', 	'4678', 	'6666', 	'3160', 	'0002', 	'2108', 	'1254', 	'1356', 	'2122', 	'2346', 	'3000', 	'3578', 	'5454', 	'5566', 	'9006', 	'0007', 	'1213', 	'2234', 	'2344', 	'5656', 	'5667', 	'60AH', 	'1236', 	'1279', 	'2110', 	'2356', 	'2424', 	'3434', 	'6544', 	'1221', 	'1243', 	'1313', 	'1456', 	'3344']
        df[col_simple_waste][df[col_art].astype(str).isin(waste_anyway)] = True
        if get_waste == 'return_all': return df

        elif get_waste: #мусорные art
            df = df[df[col_simple_waste] == True].reset_index(drop=True)
            return df[in_col]

        #НЕ - мусор == хорошие art
        df = df[df[col_simple_waste] == False].reset_index(drop=True)
        return df[in_col]

    def artIsDigOnly(self, ss):
        """артикул -  ss - проверка, что артикул состоит только из цифр - для поиска подозрительных MANN-FILTER|4011558201500"""
        dig_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        ss = str(ss)
        len_in = len(ss)
        s_list = [s for s in ss if s in dig_list]
        if (len_in - len(s_list)):
            return False #не только цифры

        return True

    def isFloat(self, s):
        """ проверка на число с плавающей точкой (тк '1.2'.isdigit() --> False)"""
        ss = str(s).replace(',', '.').split('.')
        if (len(ss) > 2) or (not all(map(lambda x: x.isdigit(), ss))):
            return False
        return float('.'.join(ss))

    def isArtClearConfidence(self, ss, len_threshold=5, dig_len_threshold = False, add_rus = True):
        """артикул -  ss - неразрывная последовательность ENG букв + dig определенной длины
        dig_len_threshold - длинная последовательность цифр можно и без букв"""
        ss = str(ss).lower()
        s_list = [s for s in ss]
        len_in = len(set(s_list))

        alphabet_list = list(string.ascii_lowercase)
        if add_rus: alphabet_list += self.rusalp_lower
        dig_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        #артикул = длинные цифры - можно

        if (dig_len_threshold)and(self.artIsDigOnly(ss))and(len(ss) > dig_len_threshold):
            return ss.upper()

        #текстовый мусор или длина меньше len_threshold
        if (len_in - len(set(s_list) & set(alphabet_list + dig_list))) or (len(ss) < len_threshold):
            return False
        #в ss нет буквы eng или dig
        if (not len(set(s_list) & set(alphabet_list))) or (not len(set(s_list) & set(dig_list))):
            return False
        return ss.upper()

    def dropSuspiciousArtInDf(self, df, art_col = 'art', brand_col = "brand"):
        """ удаляет подозрительные артикулы, но щадит бренды по списку"""
        df = df.copy()
        in_col = list(df.columns)

        conf_brand_list = ['BREMBO', 'GKN-LOEBRO-SPIDAN', 'TOPRAN', 'MAXGEAR', 'VOLVO', 'REMSA', 'SIDEM', 'LAUTRETTE', 'DIESEL-TECHNIC', 'CORTECO', 'IMPERGOM', 'AC-DELCO',
                           'BERU', 'ATE', 'COFLE', 'DENSO', 'FEBEST', 'LEMFORDER', 'NK', 'METALCAUCHO', 'JAPKO', 'MANN-FILTER', 'ZEKKERT', 'BOSAL', 'ADRIAUTO', 'CHAMPION',
                           '3F-QUALITY', 'GM', 'OSSCA', 'MONROE', 'AVANTECH', 'MS-MARSHAL', 'SACHS', 'ASVA', 'BRISK', 'LPR-AP', 'A.B.S.', 'AUGER', 'FACET', 'FORD', 'MEAT-AND-DORIA',
                           'KAMAZ', 'BOSCH', 'HOLA', 'OPTIMAL', 'RUVILLE', 'MAPCO', 'JP-GROUP', 'JANMOR', 'ELRING', 'PEUGEOT-CITROEN', 'REINZ', 'VALEO', 'SAMPA-FRENOTRUCK', 'KOLBENSCHMIDT',
                           'HC-CARGO', 'LYNX', 'NRF', 'NGK-NTK', 'MEYLE', 'AL-KO', 'NPW', 'NISSENS', 'KYB', 'FIAT-ALFA-LANCIA', 'LADA', 'DELPHI', 'MECAFILTER', 'KNECHT-MAHLE', 'FEBI-BILSTEIN', 'FAE', 'TSN-TSITRON']

        control_col = self.getRandCol("control")
        df[control_col] = df.apply(lambda x: self.isArtClearConfidence(x[art_col], len_threshold=4, dig_len_threshold=3),
                                   axis=1)
        bad_df = df[df[control_col] == False]
        drop_bart_lst = []
        if len(bad_df):
            bad_df = bad_df[~bad_df[brand_col].isin(conf_brand_list)]
            drop_bart_lst = [f"{b}|{a}" for b, a in zip(bad_df[brand_col], bad_df[art_col])]
            df = df[~df[art_col].isin(bad_df[art_col])].reset_index(drop=True)

        del df[control_col]

        return df[in_col], drop_bart_lst

    def clearWaste_inBrands(self, df, col_brand='brand', save_xlsx = False):
        """ 17/07/24 - чистка подозрительных или коротких  оварных групп в брендах для построения тепловых карт и графов"""
        def notGtvg(df, col_brand, save_xlsx = False):
            print(f"IN notGtvg len df = {len(df)}")
            not_dct = {"BOSCH": ['body', 'амортизатор', 'радиаторы', 'подушка двигателя, кпп', 'пружины подвески'],
                       "LYNX": ['body', 'шланги-патрубки'],
                       'POLCAR': ['стартер-генератор', 'свечи зажигания-накала', 'глушитель', 'пружины подвески', ],
                       'TRW-LUCAS': ['прокладки-сальники', 'свечи зажигания-накала'], 'BLUE-PRINT': ['body'],
                       'ASVA': ['электрика-датчики'], 'BREMBO': ['ремни-ролики', 'шланги-патрубки'],
                       'JIKIU': ['радиаторы'],
                       'LUK-INA-FAG': ['колодки тормозные', 'диск-барабан-тормозной', 'шланги-патрубки',
                                       'суппорт тормозной', 'тормозной шланг', 'прокладки-сальники',
                                       'электрика-датчики'],
                       'KNECHT-MAHLE': ['подвеска', 'амортизатор', 'оптика', 'шланги-патрубки'],
                       'GATES': ['подвеска', 'фильтры', 'прокладки-сальники',
                                 'колодки тормозные', 'электрика-датчики', 'амортизатор', 'диск-барабан-тормозной',
                                 'шрус-привод-пыльник-крестовина',
                                 'двигатель', 'вентиляторы-моторы', 'тормозной шланг', 'свечи зажигания-накала'],
                       'SKF': ['фильтры', 'электрика-датчики'],
                       'SANGSIN-BRAKE':['двигатель'],
                       'METALCAUCHO':['колодки тормозные'], 'SASIC':['колодки тормозные'], 'PARTRA':['амортизатор'], 'CENTO':['колодки тормозные'],
                       'SACHS':['подвеска', 'колодки тормозные'], 'HESSA':['колодки тормозные']}

            bad_df = pd.DataFrame()
            for brand in not_dct:
                tdf = df[df[col_brand] == brand]
                bad_df = pd.concat([bad_df, tdf[tdf[self.cross_global_group].isin(not_dct[brand])]])
                print(f"brand = {brand}, len bad_df = {len(bad_df)}")

            if save_xlsx: self.xlsxSave(bad_df, root_out + 'bad_df_notGtvg')

            df = df[~df[self.cross_bart].isin(bad_df[self.cross_bart])]
            print(f"IN notGtvg len df = {len(df)}")
            return df

        def onlyGtvg(df, col_brand='brand', save_xlsx = False):
            only_dct = {"CTR": ['подвеска', 'колодки тормозные', 'амортизатор', 'подушка двигателя, кпп']}
            print(f"IN onlyGtvg len df = {len(df)}, len only_dct = {len(only_dct)}")

            df_only = df[df[col_brand].isin(only_dct)]
            if save_xlsx:
                self.xlsxSave(df_only, root_out  + 'df_only')
            df_clear = df[~df[col_brand].isin(only_dct)]

            print(f"isin(only_dct) len df_only = {len(df_only)}, clear_df = {len(df_clear)}")

            new_df = pd.DataFrame()
            for brand in only_dct:
                tmp_df = df_only[(df_only[col_brand] == brand) & (df_only[self.cross_global_group].isin(only_dct[brand]))]
                new_df = pd.concat([new_df, tmp_df])
                print(f"onlyGtvg brand = {brand}, len tmp_df = {len(tmp_df)}, len new_df = {len(new_df)}")


            clear_df = pd.concat([df_clear, new_df])

            if save_xlsx:
                bad_df = df[~df[self.cross_bart].isin(clear_df[self.cross_bart])]
                self.xlsxSave(new_df, root_out + 'new_onlyGtvg')
                self.xlsxSave(bad_df, root_out + 'bad_onlyGtvg')


            print(f"OUT onlyGtvg len df = {len(clear_df)}")

            return clear_df


        in_col = df.columns
        if col_brand not in df.columns:
            col_brand = self.getRandCol('brand')
            col_art = self.getRandCol('art')
            df = self.splitBartCol(df, art_col=col_art, brand_col=col_brand)

        df = notGtvg(df, col_brand, save_xlsx = save_xlsx)
        df = onlyGtvg(df, col_brand, save_xlsx=save_xlsx)

        return df[in_col]


    ########  END чистка артикулов +   брендов ################


    def headNormalize(self, df, col_pcs, cfnt_head=1 / 1000):
        """ нормализуем выбросы кол-ва в шапке
        cfnt_head = 1/1000 -- чем больше делитель, тем меньше строк в шапке нормализуем
        Пример: 2млн * (1 / 1000) = 2000 строк"""

        def pcsCfnt(old_pcs, new_pcs):
            if new_pcs < old_pcs: return new_pcs
            return old_pcs

        df = df.copy()
        in_col = df.columns
        df[col_pcs] = df[col_pcs].astype(int)
        df = self.sort_values_fast(df, sort_col=col_pcs, ascending=False)

        len_head = int(cfnt_head * len(df)) + 1
        head_df = df[:len_head]
        df = df[len_head:].reset_index(drop=True)
        min_pcs = int(df.loc[0, col_pcs] * 1.003)  # добавляем 0,3% к минимальному
        max_pcs = int(min_pcs * 1.2)
        step_pcs = (max_pcs - min_pcs) // len_head
        lst = [min_pcs + i * step_pcs for i in range(0, len_head)]
        lst.sort(reverse=True)
        head_df = pd.concat([head_df, pd.DataFrame(lst, columns=['new_pcs'])], axis=1)

        head_df[col_pcs] = head_df.apply(lambda x: pcsCfnt(old_pcs=x[col_pcs], new_pcs=x['new_pcs']), axis=1)

        df = pd.concat([head_df, df])
        df.reset_index(drop=True, inplace=True)

        return df[in_col]




    ######## UNITE ROOT BY CROSSES #########################
    #######################################################

    def dictList_fromDf(self, df, col_2key, col_2list):
        """ 04/12/23 - словарь и сключами из col_key к которым прикреплены списки из col_list
        df = pd.DataFrame({'key_col':[1, 1, 1, 2, 2], 'val_col':[11, 111, 1111, 22, 222]})
        print(self.dictList_fromDf(df, col_2key='key_col', col_2list='val_col'))
        –> {1: [11, 111, 1111], 2: [22, 222]}
        """
        return pd.Series(df[col_2list].values,index=df[col_2key]).groupby(level=0).agg(list).to_dict()

    def longTailDuplicates(self, duplicates_root_list):
        print('start longTailDuplicates = объединение/чистка duplicates_root_list')
        j = 0
        while True:
            not_duplicates = True  # дубликатов нет
            unite_duplicates_list = []
            print(f"{j} len(duplicates_root_list) = {len(duplicates_root_list)}")
            # duplicates_root_list - лист зашел сверху
            for lst_root in duplicates_root_list:
                new_flag = True
                # unite_duplicates_list - лист формируется в текущем блоке
                for i, lst_unite in enumerate(unite_duplicates_list):
                    cross_len = list(set(lst_root) & set(lst_unite))  # пересечение root_key
                    if cross_len:
                        new_flag = False
                        not_duplicates = False  # дубликаты есть!
                        unite_duplicates_list[i] = list(set(lst_unite + lst_root))
                        """множественное пересечение root_key внутри dup_root_lst"""
                        break
                if new_flag:
                    unite_duplicates_list.append(lst_root)

            if not_duplicates:  # закончена = объединение/чистка  duplicates_root_list!
                break

            else:
                duplicates_root_list = unite_duplicates_list.copy()
                j += 1

        return unite_duplicates_list


    def uniteDuplicates(self, root_df, col_root='root_key', col_cross='cross_bart'):
        in_col = root_df.columns
        print(f"START mainUniteDuplicates,  len root_df = {len(root_df)}")

        ########### uniteRootsByCrosses    #########################
        def uniteRootsByCrosses(root_df, col_root, col_cross):
            in_col = root_df.columns
            """объединяет блоки root_key по признаку дублей cross_col
            возвращает из root_df 1) DataFrame c объединенными дубликатами united_duplicate_df и изменёнными ключами root_key
            2)duplicates_root_list  - список устаревших ключей root_key для удаления"""

            def getIndices(lst, el):
                return [i for i in range(len(lst)) if lst[i] == el]

            def getRootsIdx(long_root, idx_list):
                return list(set([long_root[idx] for idx in idx_list]))

            def calcDuplicates(root_df):
                count_col = self.getRandCol('count_col')
                root_df[count_col] = root_df.groupby(col_cross)[col_cross].transform('count')
                duplicate_df = root_df[root_df[count_col] != 1]
                if not len(duplicate_df):
                    return []
                duplicate_df.sort_values(by=count_col, ascending=False,
                                         inplace=True)  # самые задублированные ключи - вверху
                duplicate_df.reset_index(inplace=True, drop=True)
                duplicate_df.drop_duplicates(subset=col_cross, keep='first', inplace=True)
                duplicate_list = list(duplicate_df[col_cross].values)

                return duplicate_list

            print(f"IN uniteDuplicates len_df = {len(root_df)}")
            duplicate_list = calcDuplicates(root_df)
            if not len(duplicate_list):
                print('OK! No_duplicates')
                return [], []

            print_list, len_data = self.printList(count_data=duplicate_list)

            duplicates_root_list = []  # группировка - список списков пересекающихся root_key
            long_root, long_bart = list(root_df[col_root].values), list(root_df[col_cross].values)
            for j, duplicate in enumerate(duplicate_list):

                self.printInfo(j, print_list, len_data, name_loop=f"1) группировка root_key -- uniteDuplicates")

                idx_list = getIndices(long_bart, duplicate)  # индексы строк root_df, куда входит данный дубль
                short_root = getRootsIdx(long_root, idx_list)
                new_flag = True
                for i, dup_root_lst in enumerate(duplicates_root_list):
                    cross_len = list(set(dup_root_lst) & set(short_root))  # пересечение root_key
                    if cross_len:
                        new_flag = False
                        duplicates_root_list[i] = list(set(dup_root_lst + short_root))
                        """множественное пересечение root_key внутри dup_root_lst"""
                        break
                if new_flag:
                    duplicates_root_list.append(short_root)

            """внутри - duplicates_root_list - всё равно остались дубли. Почему?:
                1) Допустим, прошло 2 петли for j, duplicate in … -->  и добавились 2 непересекающиеся коробочки (группы root_key)
                duplicates_root_list = [[rk1, rk2, rk3 ], [rk44, rk55, rk66]]
                заходим в 3ю петлю ==>  j = 2 (0 –> 1  2 == 3я петля for j, duplicate in enumerate)
                short_root_j2 = [rk1, rk66 ]     
                выход после итерраций j = 0 - 1 - 2 ==> duplicates_root_list = [[rk1, rk66, rk2, rk3 ], [rk44, rk55, rk66]]
                (тк сработал breack short_root_2 склеился с 0й коробочкой, до следующей 1й = [rk44, rk55, rk66] он не дошел)
                2) Дальше стартует start create united_duplicate_df -- конструируем новый dataframe, объединяя сарык root_key с пересечениями
                for j, duplicate in enumerate(duplicates_root_list):
                    idf = root_df[root_df['root_key'].isin(duplicate)]

                    j = 0 >> idf_0 ([rk1, rk66, rk2, rk3 ])  new_root_A
                    j = 1 >> idf_1 ([rk44, rk55, rk66])   new_root_B ==> задвоение данных rk66 – итоговая 	таблица sql_root_df – может даже увеличиться         
            РЕЗЮМЕ: чистим/объединяем duplicates_root_list до победного!"""

            duplicates_root_list = self.longTailDuplicates(duplicates_root_list)

            print("start create united_duplicate_df")
            # группировка дубликатов в Dataframe
            print_list, len_data = self.printList(count_data=duplicates_root_list)

            united_duplicate_df = pd.DataFrame({n: [] for n in in_col})
            for j, duplicate in enumerate(duplicates_root_list):
                self.printInfo(j, print_list, len_data,
                              name_loop=f"3) группировка дубликатов в Dataframe-- uniteDuplicates")

                idf = root_df[root_df[col_root].isin(duplicate)]
                idf[col_root] = duplicate[0]
                idf.drop_duplicates(subset=col_cross, keep='first', inplace=True)
                united_duplicate_df = pd.concat([united_duplicate_df, idf])
                united_duplicate_df.reset_index(drop=True, inplace=True)

            duplicates_root_list = sum(duplicates_root_list, [])  # разглаживаем

            return united_duplicate_df, duplicates_root_list

        ##########     END uniteRootsByCrosses   ####################


        k = 0
        while True:
            # вычисляем дубликаты cross_brand_art_uni
            united_df, duplicate_list = uniteRootsByCrosses(root_df[in_col], col_root, col_cross)
            print(f'\n k = {k} кол-во групп дубликатов: {len(duplicate_list)}')
            print(f"k = {k} всего строк united_df = {len(united_df)}")
            if not len(duplicate_list):
                break

            root_df = root_df[~root_df[col_root].isin(duplicate_list)]
            root_df = pd.concat([root_df, united_df])

            k += 1
            print(f"next_k = {k}")

        print(f"END mainUniteDuplicates, len root_df = {len(root_df)}")

        return root_df[in_col]

    ############# END UNITE ROOT BY CROSSES ##################


    ######       TovCruppa    #################
    def findTovgruppa(self, df, column_to_scan="name", name_KeyWords='mgpt_KeyWords_PARTS.xlsx', external_KeyWords=False,
                          contact_how='inner',
                          rename_tovgruppa=False, rename_global=False, del_global=False, del_rule=True,
                          only_new_rules=False, add_cols = False, print_info=True, global_list_only = False):
        if print_info: print('start = ', time.ctime())
        df.reset_index(drop=True, inplace=True)

        '''
        возвращает 'tovgruppa' - столбец расшифровки простой поиск ттоварной группы по ключевым словам с исключениями
        упрощенный алгоритм поиска текста по ключевым словам

        # name_Keywords передаём как параметр функции
        'KeyWords_PARTS.csv'  # Запчасти
        # KeyWords_Brands_ALL.csv # бренды
        # 'KeyWords_CARS.csv'  #Автомобили подробно
        #'KeyWords_MAKE.csv'  # марки авто
        # 'KeyWords_Oil_Brand.csv'    # Масла
        # 'KeyWords_Autochemestry.csv' #Автохимия
        '''

        if print_info: print('высота на входе', len(df))
        # загружаем ключевые слова
        if type(external_KeyWords) is bool:
            catalog_KeyWords = root_path + 'DATA_CATALOGS/KeyWords_Xlsx/'
            Katalog_TovGruppa = pd.read_excel(catalog_KeyWords + '/' + name_KeyWords)

        else:  # внешняя таблица ключевых слов
            external_KeyWords.reset_index(drop=True, inplace=True)
            Katalog_TovGruppa = external_KeyWords

        if only_new_rules:
            Katalog_TovGruppa = Katalog_TovGruppa[Katalog_TovGruppa['the_rule'].str.contains("new")]
            Katalog_TovGruppa.reset_index(drop=True, inplace=True)

        if global_list_only:
            Katalog_TovGruppa = Katalog_TovGruppa[Katalog_TovGruppa['global_group'].isin(global_list_only)].reset_index(drop=True)

        df.dropna(subset=[column_to_scan], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df[column_to_scan] = df[column_to_scan].astype(str)
        df = df.sort_values(by=column_to_scan, ascending=True)
        df['index'] = df.index

        if print_info: print('Старт цикла:', time.ctime())

        old_column_names = [column_to_scan, 'index']
        new_column_names = ['cross_tovgruppa', 'cross_global_group', 'Number_the_rule']
        if add_cols: #дополнительные колонки из _KeyWords
            new_column_names += add_cols
        new_columns = {n: [] for n in old_column_names + new_column_names}
        # read katalog
        katalog = []
        for j in range(Katalog_TovGruppa['KeysYes'].count()):
            keys_yes = str(Katalog_TovGruppa.loc[j, 'KeysYes']).split('|')
            keys_no = str(Katalog_TovGruppa.loc[j, 'KeysNo']).split('|')
            new_vals = {
                'cross_tovgruppa': Katalog_TovGruppa.loc[j, 'tovgruppa'],
                'cross_global_group': Katalog_TovGruppa.loc[j, 'global_group'],
                'Number_the_rule': Katalog_TovGruppa.loc[j, 'the_rule']
            }
            if add_cols:
                new_vals.update({c:Katalog_TovGruppa.loc[j, c] for c in add_cols})
            katalog.append((keys_yes, keys_no, new_vals))

        print_list = [i for i in range(len(df)) if i % (int((len(df) / 20)) + 1) == 0]

        for i in range(df[column_to_scan].count()):
            if i in print_list:
                if print_info: print(f"{str(datetime.datetime.now()).split('.')[0]} -- {int(100 * i / len(df))} % пройдено, {name_KeyWords}")
            name = ' '.join(df.loc[i, column_to_scan].lower().split())
            if (len(new_columns[column_to_scan])) and (new_columns[column_to_scan][-1] == name):  #

                for n in old_column_names:
                    if n == 'index':
                        continue
                    new_columns[n].append(df.loc[i, n])

                new_columns['index'].append(df.loc[i, 'index'])
                for n in new_column_names:
                    new_columns[n].append(new_columns[n][-1])

            else:
                for keys_yes, keys_no, new_vals in katalog:
                    # Здесь проверяем 2 условия Yes-Yes и 3 No-No-No:
                    if all(k in name + ' ' for k in keys_yes):
                        if not any(k in name for k in keys_no):
                            for n in old_column_names:
                                new_columns[n].append(df.loc[i, n])
                            for n in new_column_names:
                                new_columns[n].append(new_vals[n])
                            break

        df_new = pd.DataFrame(new_columns)
        col = ['index', 'cross_tovgruppa', 'cross_global_group', 'Number_the_rule']
        if add_cols: col += add_cols
        df_new = df_new[col]

        df = pd.merge(df, df_new, how=contact_how, on='index')
        del df['index']

        if rename_tovgruppa:
            df.rename(columns={'cross_tovgruppa': rename_tovgruppa}, inplace=True)
        if rename_global:
            df.rename(columns={'cross_global_group': rename_global}, inplace=True)
        if (not rename_global) and (del_global):
            del df['cross_global_group']
        if del_rule:
            del df['Number_the_rule']

        if print_info:  print('Высота таблицы = ', df[column_to_scan].count())
        df = df.reset_index(drop=True)
        return df


    def findNameByEtalonTab(self, df, col_to_scan_df, df_etalon, col_etalon_df, col_res_names, mach_threshold = 0):
        """ сравнивает 2 таблицы, ищет максимальное соответсвие текста
        (!) Но это не точно
        (!!!) Должны быть разные col_to_scan_df, , col_etalon_df"""
        def lstUniLowerGap(s):
            # list - только уникальные слова, маленькие
            s = self.clearWaste(ss=s, type_waste='clear_text')
            s = s.lower()
            s = s.split()
            return ' '.join(list(set(s))).split()

        def compareNameAndEtalons(name, etalon_list, etalon_names_list, e_list):
            """etalon_llist - список списков, где фразы разбиты на слова и вложены в list
            etalon_names_list - исходные фразы, вложенные в список (из которого создали etalon_llist)"""
            name_lst = lstUniLowerGap(name)  # ['1l', 'a3/b4', 'titanium', '0w30', 'sae', 'castrol', 'edge']
            if not len(name_lst):
                return 0, 0
            mach_list = list(map(lambda x: len(list(set(name_lst) & set(x))) / len(name_lst), etalon_list))
            if not mach_list:
                return 0, 0

            max_mach = max(mach_list)
            name_mach = etalon_names_list[mach_list.index(max_mach)]
            name_etalon = e_list[mach_list.index(max_mach)]

            return name_mach, max_mach, name_etalon

        e_list = list(df_etalon[col_etalon_df].values) #с чем сравниваем
        names_list = list(df_etalon[col_res_names].values) #что подставляем на выходе
        etalon_list = list(map(lstUniLowerGap, e_list))  # список списков с уникальными словами

        scan_list = set(df[col_to_scan_df].values)
        print_list, len_data = self.printList(count_data=scan_list)

        res_dict = {col_to_scan_df: [], col_res_names: [], 'mach_cfnt': []}
        if col_res_names != col_etalon_df:
            res_dict.update({col_etalon_df:[]})
        for i, name in enumerate(scan_list):
            self.printInfo(i, print_list, len_data, name_loop="findNameByEtalonTab")
            name_mach, max_mach, name_etalon = compareNameAndEtalons(name=name, etalon_list=etalon_list, etalon_names_list=names_list, e_list=e_list)
            if not name_mach:
                continue
            res_dict[col_to_scan_df].append(name)
            res_dict[col_res_names].append(name_mach)
            res_dict['mach_cfnt'].append(max_mach)
            if col_res_names != col_etalon_df:
                res_dict[col_etalon_df].append(name_etalon)


        res_df = pd.DataFrame(res_dict)
        if mach_threshold:
            res_df = res_df[res_df['mach_cfnt'] > mach_threshold].reset_index(drop=True)
        return pd.merge(df, res_df, on=col_to_scan_df, how='inner')

    def litersFind(self, df, col_cross_bart=False, column_to_scan='name', col_global_group=False, keep_rule=False):
        out_col = list(df.columns) + ['liters']

        if not col_cross_bart:
            col_cross_bart = self.getRandCol()
            df[col_cross_bart] = df.index

        if keep_rule:
            out_col += ['Number_the_rule']

        if not col_global_group:  # нужны global_group, чтобы потом не чистить мусор

            df_n = self.minLenNameDf(df, col_name=column_to_scan)  # предварительная чистка базы - пустые строки name
            df_n = self.findTovgruppa(df_n, column_to_scan=column_to_scan, contact_how='inner',
                                      rename_tovgruppa='cross_tovgruppa',
                                      rename_global='cross_global_group')


        group_filter = ['масло моторное', 'масла-жидкости', 'масло акпп/мкпп/вариатор', 'антифриз', 'auto chemistry']
        df_n = df_n[df_n['cross_global_group'].isin(group_filter)] if not col_global_group else df[
            df['cross_global_group'].isin(group_filter)]


        liters_keys = self.uploadDataFile(root_path + "/DATA_CATALOGS/KeyWords_Xlsx/mgpt_KeyWords_LitersKg.xlsx")
        df_n[column_to_scan] = " " + df_n[column_to_scan].astype(
            str) + " "  # в начало и конец " " для расширения срабатывания keywords

        del df_n['cross_global_group']
        if 'cross_tovgruppa' in df_n.columns: del df_n['cross_tovgruppa']

        df_n = self.findTovgruppa(df_n, column_to_scan=column_to_scan, contact_how='inner',
                                  rename_tovgruppa='liters',
                                  del_global=True, external_KeyWords=liters_keys, del_rule=False)

        df_n.drop_duplicates(subset=col_cross_bart, keep="first", inplace=True)

        merge_col = [col_cross_bart, 'liters']
        if keep_rule: merge_col += ['Number_the_rule']
        df = pd.merge(df, df_n[merge_col], on=col_cross_bart, how='left')

        df.fillna('', inplace=True)
        return df[out_col]

    def raitingTovgruppa(self, s):
        def pessimizeRating(s):
            if "(+части)" in s: return -100
            return 0

        """ оцифровка рейтинга строки tovgruppa длина + кол-во опций "[": амортизатор [задний][левый и правый]"""
        len_opt = list(s).count('[')
        raiting_tovgruppa = len(s) + 3 * len_opt
        return raiting_tovgruppa + pessimizeRating(s)

    def bestTvgInRootKey(self, df, col_root_key, col_tovgruppa = "cross_tovgruppa", col_global_group = "cross_global_group", col_name = "name"):
        """ выбираем лучшую tvg внутри группы строк, объединенных одинаковыми ключами root_key"""
        in_col = df.columns
        df_merge = df.copy()
        len_col = f"len{random.randint(1000,10000)}" #случайное название колонки, чтобы не было конфликта при merge
        df_merge[len_col] = df_merge.apply(lambda x: self.raitingTovgruppa(x[col_tovgruppa]), axis=1)
        df_merge.sort_values([len_col], axis=0, ascending=False, inplace=True)
        df_merge.drop_duplicates(subset=col_root_key, keep='first', inplace=True)
        df.drop(columns=[col_tovgruppa, col_global_group, col_name], inplace=True)
        df = pd.merge(df, df_merge[[col_root_key, col_tovgruppa, col_global_group, col_name]],
                      on=col_root_key, how='inner')

        return df[in_col]

    def mostPopularTvgInRootKey(self, df, col_root_key, col_bart='cross_bart', col_tovgruppa="cross_tovgruppa",
                                col_global_group="cross_global_group", col_name='name'):
        """Самая частотная Tvg внутри root_key.
        В отличии от bestTvgInRootKey считаем не самую "крутую" raitingTovgruppa, а самую частотную == повторяющуюся"""
        df = df.copy() #удаляет колонки у df
        in_col = df.columns
        df_merge = df.copy()
        # оставляем уникальные внутри root_key + col_bart + col_tovgruppa (чтобы дубли bart не накручивали результат)
        control_col = self.getRandCol(col='control')  # случайное название колонки, чтобы не было конфликта при merge
        df_merge[control_col] = df_merge[col_root_key].astype(str) + df_merge[col_bart].astype(str) + df_merge[
            col_tovgruppa]
        df_merge.drop_duplicates(subset=control_col, keep='first', inplace=True)

        count_col = self.getRandCol(col='count')
        df_merge[count_col] = df_merge.groupby(col_root_key)[col_tovgruppa].transform('count')
        df_merge.sort_values([count_col], ascending=False, inplace=True)
        df_merge.drop_duplicates(subset=col_root_key, keep='first', inplace=True)

        # заменяем в df колонки на самые частотные значения из merge
        df.drop(columns=[col_tovgruppa, col_global_group, col_name], inplace=True)
        df = pd.merge(df, df_merge[[col_root_key, col_tovgruppa, col_global_group, col_name]], on=col_root_key,
                      how='inner')
        return df[in_col]


    def minLenNameDf(self, df, col_name="name", type_waste="clear_text", min_len=4):
        """ контроль минимальной длины текста в колонке col_name"""
        in_col = df.columns
        new_col = self.getRandCol('new_col')
        df[new_col] = df[col_name].copy()
        df = self.clearSpecSmb(df, col=new_col, type_waste=type_waste)
        df[new_col] = df[new_col].apply(lambda x: " ".join(str(x).split()))
        count_col = self.getRandCol("count_col")
        df[count_col] = df[new_col].astype("str").str.len()
        df = df[df[count_col] >= min_len].reset_index(drop=True)

        return df[in_col]

    def getViscosity(self, df, column_to_scan='name', col_cross_bart = 'cross_bart', with_viscosity_only = False):
        """ выделяет масло из cross_tovgruppa только для моторного масла, остальные - 0
        col_cross_bart = cross_bart - у одинаковых bart может быть разные name, то viscosity ищем внутри совпадающих cross_bart, иначе - ищем внутри name (column_to_scan)"""
        if 'cross_tovgruppa' not in df.columns:
            df = self.findTovgruppa(df, column_to_scan=column_to_scan)
            if not len(df): return df
        df = df[df['cross_global_group'] == 'масло моторное']
        vdf = df[(df["cross_tovgruppa"].str.contains("w-")) & (df["cross_tovgruppa"].str.contains("масло моторное "))]
        vdf['viscosity'] = df["cross_tovgruppa"].str.replace('масло моторное ', '').str.upper()

        if with_viscosity_only: merge_how = 'inner'
        else: merge_how = 'left'

        if col_cross_bart:
            vdf.drop_duplicates(subset=col_cross_bart, keep='first', inplace=True)
            df = pd.merge(df, vdf[[col_cross_bart, 'viscosity']], on=col_cross_bart, how=merge_how)

        else:
            vdf.drop_duplicates(subset=column_to_scan, keep='first', inplace=True)
            df = pd.merge(df, vdf[[column_to_scan, 'viscosity']], on=column_to_scan, how=merge_how)

        return df


    def oilApiFuelTypeFind(self, df, column_to_scan='name', col_cross_bart=False, col_global_group=False, keep_rule=False):
        def mostPopularApiFuel(df, col_cross_bart):
            df = df.copy()

            control_col = self.getRandCol('control_col')
            df[control_col] = df[col_cross_bart].astype(str) + df['api']
            count_col = self.getRandCol(col='count')
            df[count_col] = df.groupby(control_col)[control_col].transform('count')
            df.sort_values([count_col], ascending=False, inplace=True)
            df.drop_duplicates(subset=col_cross_bart, keep='first', inplace=True)

            return df


        out_col = list(df.columns) + ['api', 'fuel']

        if not col_cross_bart:
            col_cross_bart = self.getRandCol()
            df[col_cross_bart] = df.index

        if keep_rule:
            out_col += ['Number_the_rule']
        if not col_global_group:  # нужны global_group, чтобы потом не чистить мусор
            df_n = self.minLenNameDf(df, col_name=column_to_scan)  # предварительная чистка базы - пустые строки name
            df_n = self.findTovgruppa(df_n, column_to_scan=column_to_scan, contact_how='inner',
                                      rename_tovgruppa='cross_tovgruppa',
                                      rename_global='cross_global_group')

        group_filter = ['масло моторное']
        df_n = df_n[df_n['cross_global_group'].isin(group_filter)] if not col_global_group else df[
            df[col_global_group].isin(group_filter)]

        api_keys = self.uploadDataFile(root_path + "/dataCatalogs/KeyWords_Xlsx/KeyWords_API_fuelType.xlsx")

        del df_n['cross_global_group']
        if 'cross_tovgruppa' in df_n.columns: del df_n['cross_tovgruppa']

        df_n = self.findTovgruppa(df_n, column_to_scan=column_to_scan, contact_how='inner',
                                  rename_tovgruppa='api', rename_global='fuel', external_KeyWords=api_keys, del_rule=False)

        if col_cross_bart:
            df_n = mostPopularApiFuel(df_n, col_cross_bart)
            cols = [col_cross_bart, 'api', 'fuel']
            merge_col = col_cross_bart
        else:
            cols = [column_to_scan, 'api', 'fuel']
            merge_col = column_to_scan

        if keep_rule: cols += ['Number_the_rule']
        df = pd.merge(df, df_n[cols], on=merge_col, how='left')

        return df[out_col]

    ###########    END CrossTovgruppa    #####################


    def sort_values_fast(self, df, sort_col, ascending=True):
        x = 1 if ascending else -1
        df = df.copy()
        sorted_indices = np.argsort(x*df[sort_col].values)
        df = df.iloc[sorted_indices]
        df.reset_index(drop=True, inplace=True)
        return df

    def roundIntUp(self, df, col_round, n_round =1):
        """округление в большую сторону до int m_round"""
        df[col_round] = df[col_round].apply(lambda x: max(math.ceil(x), n_round)) #0 и 0000.1 округляем в большую сторону до 1
        return df

    def isPyCharm(self,):
        # Проверка переменных окружения, установленных PyCharm
        if 'PYCHARM_HOSTED' in os.environ:
            return True

        # Проверка аргументов командной строки, специфичных для PyCharm
        for arg in sys.argv:
            if 'pycharm' in arg.lower():
                return True

        return False






