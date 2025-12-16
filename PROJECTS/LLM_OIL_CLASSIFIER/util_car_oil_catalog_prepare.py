import pandas as pd

from bootstrap import *
import re
import numpy as np
path_from_to = root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/'


def recomendDel(df):

    df['Спецификация масла'] = df['Спецификация масла'].str.split(' Мы рекомендуем: ', n=1, expand=True)[0].str.strip()
    df['Спецификация масла'] = df['Спецификация масла'].str.split(' мы рекомендуем: ', n=1, expand=True)[0].str.strip()
    df['Спецификация масла'] = df['Спецификация масла'].str.split(', но мы рекоменд', n=1, expand=True)[0].str.strip()
    df['Спецификация масла'] = df['Спецификация масла'].str.split(' - мы рекоменд', n=1, expand=True)[0].str.strip()
    df['Спецификация масла'] = df['Спецификация масла'].str.split(' мы рекомендуем', n=1, expand=True)[0].str.strip()
    df['Спецификация масла'] = df['Спецификация масла'].str.split(' Мы рекомендуем', n=1, expand=True)[0].str.strip()

    return df

def cluePetterns(df, col):
    pattern = r'(?<!\s)([А-ЯA-Z][^ ]*)'

    # Извлекаем всё, что начинается с "приклеенной" заглавной буквы

    for c in col:
        matches = df[c].astype(str).str.findall(pattern).explode().dropna()

        # Убираем дубликаты
        print(f"{c}:")
        unique_words = sorted(set(matches))

        print(unique_words)


def v1(ss):
    lst = ['н.в.', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for l in lst:
        ss = ss.replace(str(l)+'Объём', f"{l}. Объём")

    ss = ss.replace('Подбор масла для ', '')

    def insert_space_before_words(text, words):
        # Создаём регулярное выражение: (?<=[^\s])(Кузов:|Год выпуска:|Двигатель:|...)
        pattern = r'(?<=[^\s])(' + '|'.join(re.escape(word) for word in words) + r')'
        return re.sub(pattern, r'. \1', text)

    words_to_separate = ["Кузов:", "Год выпуска:", "Поколение:"]
    ss = insert_space_before_words(ss, words_to_separate)


    return ss

def v2(ss, url):
    url = url.replace('https://podbormasla.ru/', '').replace('gen', ' generation ').replace('_', ' ')
    url = url.split('/')
    url = [' '.join(x.split()) for x in url]


    for u in url:
        if u not in ss.lower():
            ss += f" {u.upper()}"

    return ss

# def fluid_Type(df):
#     lst = df['fluid_type'].values.tolist()
#     lst = [x.split(' Модель:')[0] for x in lst]
#     print(list(set(lst)))

def fluid_type(ss):
    dct_replace = {'МАСЛО в ДВИГАТЕЛЬ':"МАСЛО в ДВИГАТЕЛЬ (масло моторное)", 'МАСЛО в РАЗДАТОЧНУЮ КОРОБКУ':'МАСЛО в РАЗДАТОЧНУЮ КОРОБКУ (масло раздатки)',
                   'МАСЛО в ТОРМОЗНУЮ СИСТЕМУ':'МАСЛО в ТОРМОЗНУЮ СИСТЕМУ (жидкость тормозная)', 'МАСЛО в АКПП': "МАСЛО в АКПП (коробка автомат)", 'МКПП':'МКПП (механическая коробка)'}
    for k, v in dct_replace.items():
        ss = ss.replace(k, v)

    return ss

def remove_trailing_model(text):
    if isinstance(text, str):
        # Убираем " Модель:", "Модель:", " модель:", " Модель." и т.п. — только если в конце
        return re.sub(r'\s*Модель[:.]?\s*$', '', text, flags=re.IGNORECASE)
    return text

def textReplace(df):

    df["vehicle"] = df["vehicle"].str.replace('годаОбъём', 'года. Объём')
    df["vehicle"] = df["vehicle"].str.replace('Масла для ', '')

    return df

def collectParsedData():
    path_from =  root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/OIL_SELECTION_PARSER/data_out/'
    df = U24.data2Df_upload(path_from)


    df['vehicle'] = df['vehicle'].apply(v1)
    df = textReplace(df)
    df['vehicle'] = df.apply(lambda row: v2(row['vehicle'], row['url']), axis=1)


    df['fluid_type'] = df['fluid_type'].apply(remove_trailing_model)
    df['fluid_type'] = df['fluid_type'].apply(fluid_type)

    df["Q_text"] = df['fluid_type'] + '\n\nДля автомобиля: ' + df['vehicle']

    df = recomendDel(df)

    df["A_text"] = df['fluid_type'] + '\n\nДля автомобиля: ' + df['vehicle'] + '\n\nОбъём заливки: ' + df['Объём заливки'] + '\n\nСпецификация масла/жидкости: ' + df['Спецификация масла']

    df.dropna(subset=['Q_text', 'A_text'], inplace=True)


    for c in ['Рекомендация', 'Моторное масло', 'Масло КПП', 'Трансмиссионное масло']:
        if c in df.columns: del df[c]


    U24.xlsxSave(df, path_from_to + "car_oil_catalog.xlsx")

    return df

def paramClasifier(df):

    def car_from_URL(url):
        url = url.replace('https://podbormasla.ru/', '').replace('gen', '').replace('_', ' ')
        url = url.split('/')
        url = ' '.join(url)

        return url

    def extract_fuel(ss):
        x = ss.lower()
        if ('электр' in x) or ('батарея:' in x): return 'электричество'
        if 'гибрид' in x: return 'гибрид'
        if any(y in x for y in ['бенз. ', 'бензин']): return 'бензин'
        if any(y in x for y in ['диз. ', 'tdi']): return 'дизель'


        return None

    def extract_SAE(ss):
        if (not 'масло моторное ' in ss ) or (not 'w-' in ss): return ''
        return ss.split()[-1]

    def extract_year(ss):
        if not 'Год выпуска' in ss: return ''
        if 'Год выпуска: -Объём двигателя' in ss: return ""
        return ss.split('Год выпуска: ')[-1].split('. Объём двигателя')[0].split('Батарея: ')[0].split(' Объём двигателя:')[0].split('Объём двигателя:')[0].split(' GENERATION')[0].split(' г.Объём двигател')[0].split(' годаОбъём двигателя:')[0].split(' MOSKVITCH')[0]

        #Объём двигателя:


    df = U24.findTovgruppa(df, column_to_scan='fluid_type')
    df.rename(columns={'cross_tovgruppa': 'part_tovgruppa', 'cross_global_group':'part_global_group'}, inplace=True)
    df.fillna('', inplace=True)
    df['sae'] = df['part_tovgruppa'].apply(extract_SAE)

    df['car_url'] = df['url'].apply(car_from_URL)
    df = U24.findTovgruppa(df, column_to_scan='car_url', name_KeyWords='mgpt_KeyWords_CARS.xlsx', add_cols=['vehicle_type'],contact_how='left')
    df.rename(columns={'cross_tovgruppa': 'car', 'cross_global_group': 'make'}, inplace=True)

    df['brand_own'] = df['make'].copy()
    df = BAP.dfBrandColToUni(df, brand_col='brand_own', drop_no_brand=False)

    df['fuel'] = df['vehicle'].apply(extract_fuel)
    df['year'] = df['vehicle'].apply(extract_year)

    def yearStartStop(ss):
        if not ss: return 0, 0

        ss = ss.split('-')
        start, stop = ss
        stop = stop.replace(' г.', '')

        start = ''.join(start.split())
        stop = ''.join(stop.split())


        stop = stop if (not 'н' in stop) and stop else '2031'
        stop = stop if '.' not in stop else stop.split('.')[-1]
        stop = stop.replace('года', '')
        start = start if '.' not in start else start.split('.')[-1]

        start = ''.join(start.split())
        stop = ''.join(stop.split())

        return int(start), int(stop)

    df[['year_start', 'year_stop']] = df['year'].apply(lambda x: pd.Series(yearStartStop(x)))
    df['year_interval'] = df.apply(
        lambda row: '|'.join(str(y) for y in range(row['year_start'], row['year_stop'] + 1)),
        axis=1
    )

    def extract_volume(ss):
        if ' л.' not in ss: return ''
        return ss.split(' л.')[0].split(' ')[-1].replace('Plus)', '')

    df['volume'] = df['vehicle'].apply(extract_volume)

    df.rename(columns={'part_tovgruppa': 'tovgruppa', 'part_global_group':"global_group"}, inplace=True)

    bad_list = ['АКБ', 'свечи зажигания-накала', 'двигатель', 'фильтры','etc', 'агрегаты', 'радиаторы',
                'body', 'рейка рулевая и гур', 'блоки-модули', 'насосы-помпы', 'стартер-генератор', 'trucks-and-special']

    df = df[~df['global_group'].isin(bad_list)]


    df = recomendDel(df)



    from util_oil_assortment_prepare import specificationCol
    df = specificationCol(df)

    df['specification'] = df['specification'].replace(['', 'nan', 'None'], np.nan)
    df['specification'] = df['specification'].fillna('')
    df = df[df['specification'] != '']


    U24.xlsxSave(df, path_from_to + "_car_oil_catalog_params.xlsx")

def paramClassifier_1():
    path_out = root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/OIL_SELECTION_PARSER/data_out_mmarket/'
    df = U24.data2Df_upload(path_out)
    #
    #

    # df = U24.data2Df_upload(root_out + '_tmp.xlsx')
    # col = ['Вязкость SAE', 'API', 'ACEA']

    df['vehicle'] = (df['make'].astype(str) + ' ' + df['model'].astype(str) + ' ' + df['modification'].astype(str) + ', ' + df['modification_1'].astype(str) +
                     ', ' + df['fuelType'].astype(str) + ' ' + df['horsePower'].astype(str) + ' л.c.')

    df['fluid_type'] = 'масло моторное'
    df['tovgruppa'] = 'масло моторное'
    df['global_group'] = 'масло моторное'

    c = 'oil_capacity'
    df[c] = df[c].replace(['', 'nan', 'None'], np.nan)
    df[c] = df[c].fillna('')


    df['Объём заливки'] = df['oil_capacity'].apply(lambda x: U24.strEngDigRus(x, punctuation_marks=True))
    df['Объём заливки'] = df['Объём заливки'].str.replace(' Заправочный объём ', '. Заправочный объём ')

    df.rename(columns={'Вязкость SAE':'sae'}, inplace=True)

    col = ['sae', 'API', 'ACEA', 'ILSAC', 'OEM']

    for c in col:

        df[c] = df[c].replace(['', 'nan', 'None'], np.nan)
        df[c] = df[c].fillna('')


    df['Спецификация масла'] = ''
    def v1(x, y, col):
        if not x and y: return f'{col.upper()} {y}'
        if y: return x + f', {col.upper()} {y}'

        return x

    for c in col:
        df['Спецификация масла'] = df.apply(lambda x: v1(x['Спецификация масла'], x[c], c), axis=1)

    df["Q_text"] = 'МАСЛО в ДВИГАТЕЛЬ (масло моторное)' + '\n\nДля автомобиля: ' + df['vehicle']


    df["A_text"] = 'МАСЛО в ДВИГАТЕЛЬ (масло моторное)' +  '\n\nДля автомобиля: ' + df['vehicle'] + '\n\nОбъём заливки: ' + df[
        'Объём заливки'] + '\n\nСпецификация масла/жидкости: ' + df['Спецификация масла']

    df['car_url'] = df['make'].astype(str)  + ' ' + df['model'].astype(str)

    df.drop(columns=['make', 'model'], inplace=True)
    df = U24.findTovgruppa(df, column_to_scan='car_url', name_KeyWords='mgpt_KeyWords_CARS.xlsx',
                           add_cols=['vehicle_type'], contact_how='left')
    df.rename(columns={'cross_tovgruppa': 'car', 'cross_global_group': 'make', 'fuelType':'fuel'}, inplace=True)

    print(f"Cadillac ATS  + Ravon Nexia R3...")

    df['brand_own'] = df['make'].copy()
    df = BAP.dfBrandColToUni(df, brand_col='brand_own', drop_no_brand=False)

    df['fuel'] = df['fuel'].str.lower()


    def extract_year(ss):
        if '2012 - 2019' in ss:
            return '2012 - 2019'
        if '2000 - 2010' in ss:
            return '2000 - 2010'
        if ss[-1] == ')':
            return ss.split('(')[-1].replace(')', '')

        if ss[-1] == '.':
            return ss.split(')')[-1]

        return ss.split(' ')[-1]

    df['year'] = df['modification'].apply(extract_year)
    df['year'][df['car_url'] == 'Sollers Argo'] = '2016 – 2017'
    df['year'][df['car_url'] == 'Sollers Atlant'] = '2021 - Н.В.'


    def yearStartStop_1(ss):
        if not ss: return 0, 0
        ss = ss.lower()
        if '-' in ss:
            ss = ss.split('-')

        else:
            ss = ss.split('–')
        start, stop = ss
        stop = stop.replace(' г.', '')

        start = ''.join(start.split())
        stop = ''.join(stop.split())


        stop = stop if (not 'н' in stop) and stop else '2031'
        stop = stop if '.' not in stop else stop.split('.')[-1]
        stop = stop.replace('года', '')
        start = start if '.' not in start else start.split('.')[-1]

        start = ''.join(start.split())
        stop = ''.join(stop.split())

        return int(start), int(stop)


    df[['year_start', 'year_stop']] = df['year'].apply(lambda x: pd.Series(yearStartStop_1(x)))
    df['year_interval'] = df.apply(
        lambda row: '|'.join(str(y) for y in range(row['year_start'], row['year_stop'] + 1)),
        axis=1
    )

    df['volume'] = df['volume'].astype(str).str.replace(',', '.')

    def specification_construct(sae, api, acea, ilsac, oem):
        oem = oem.upper()
        api = api.upper()
        all_sae = 0
        all_api = 0
        all_acea = 0
        all_ilsac = 0
        all_oem = 0
        if sae:
            sae_1 = sae.split(' / ')
            sae_2 = [x.replace('-', '') for x in sae_1]

            sae_3 = ['SAE ' + x for x in sae_1]
            sae_4 = ['SAE' + x for x in sae_1]
            sae_5 = ['SAE ' + x for x in sae_2]
            sae_6 = ['SAE' + x for x in sae_2]

            all_sae = sae_1 + sae_2 + sae_3 + sae_4 + sae_5 + sae_6

        if api:
            api_1 = api.split(' / ')
            api_2 = [x.replace('-', '') for x in api_1]

            api_3 = ['API ' + x for x in api_1]
            api_4 = ['API' + x for x in api_1]
            api_5 = ['API ' + x for x in api_2]
            api_6 = ['API' + x for x in api_2]

            all_api = api_1 + api_2 + api_3 + api_4 + api_5 + api_6

        if ilsac:
            ilsac_1 = ilsac.split(' / ')
            ilsac_2 = [x.replace('-', '') for x in ilsac_1]

            ilsac_3 = ['ILSAC ' + x for x in ilsac_1]
            ilsac_4 = ['ILSAC' + x for x in ilsac_1]
            ilsac_5 = ['ILSAC ' + x for x in ilsac_2]
            ilsac_6 = ['ILSAC' + x for x in ilsac_2]

            all_ilsac = ilsac_1 + ilsac_2 + ilsac_3 + ilsac_4 + ilsac_5 + ilsac_6

        if acea:
            acea_1 = acea.split(' / ')
            acea_2 = [x.replace('-', '') for x in acea_1]

            acea_3 = ['ACEA ' + x for x in acea_1]
            acea_4 = ['ACEA' + x for x in acea_1]
            acea_5 = ['ACEA ' + x for x in acea_2]
            acea_6 = ['ACEA' + x for x in acea_2]

            all_acea = acea_1 + acea_2 + acea_3 + acea_4 + acea_5 + acea_6

        if oem and 'DEXOS' in oem:
            oem_1 = oem.split(' / ')
            oem_2 = [x.replace('-', '') for x in oem_1]

            oem_3 = ['GM ' + x for x in oem_1]
            oem_4 = ['GM' + x for x in oem_1]
            oem_5 = ['GM ' + x for x in oem_2]
            oem_6 = ['GM' + x for x in oem_2]

            all_oem = oem_1 + oem_2 + oem_3 + oem_4 + oem_5 + oem_6

        res_lst = [all_sae, all_api, all_acea, all_ilsac, all_oem]
        result = []
        for x in res_lst:
            if not x: continue
            result += x

        result = set(result)


        return '|'.join(result)

    df['specification'] = df.apply(lambda x: specification_construct(x['sae'], x['API'], x['ACEA'], x['ILSAC'], x['OEM']), axis=1)

    col = U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/car_oil_catalog_params.xlsx').columns

    U24.tmp_saveOut(df[col])


if __name__ == '__main__':
    # df = collectParsedData()
    # df = U24.data2Df_upload(root_out + "oil_classifier.xlsx")

    # paramClasifier(df)

    paramClassifier_1()