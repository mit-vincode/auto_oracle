from bootstrap import *
import re
import numpy as np

classification_stop_list = ['5', '-', '--', 'MOBIL', 'OEM', 'API', 'SHELL', 'FORD']
brands_df = U24.data2Df_upload(root_path + '/DATA_CATALOGS/KeyWords_Xlsx/brands_to_uni.xlsx')
brands_oe = (list(brands_df[brands_df['oe_aftermarket'] == 'OE']['brand_uni'].unique() )
             + list(brands_df[brands_df['oe_aftermarket'] == 'OE']['key_brand'].unique()))
classification_stop_list += brands_oe

brand_oil_list = ['3TON','ВМПАВТО','ВОЛГА-ОЙЛ','ДЕВОН','ЗУБР','ЛАКИРИС','ЛУКОЙЛ (LUKOIL)','НТК-NORD-OIL','ОЙЛРАЙТ','ТОТЕК','ABRO','AC-DELCO','ADDINOL','AIMOL','AISIN','AKROSS','ALPHA-S','ALPINE','AMALIE','AMSOIL','API','ARAL','ARCTIC-CAT','ARDECA','ARECA','AREOL','AREON','ARIAL','ASTRON','ATLAS-OIL','AUTOBACS','AUTOCOMPONENT','AVENO','AVISTA','AVOL-STREICHERT','AVT-GROUP','BALLU','BARDAHL','BENDIX','BIBICARE','BIZOL','BLUE-BLOOD','BLUE-PRINT','BMW','BOMBARDIER','BP','BRAIT','C.N.R.G.','CAM2','CARLSON','CARTENDER','CASTROL','CEPSA','CHAMPION','CHAMPION-OIL','CHANGAN','CHEMPIOIL','CHERY','CHEVRON','CHRYSLER-DODGE-JEEP','CLAAS','CNRG','COMMA','CONSOL','COUNTRY','CUPPER','CWORKS','CYCLON','DAF','DAIHATSU','DATSUN','DCEC','DDE','DEXTRIM','DIVINOL','DRAGON','EKKA','ELF','ELISSA','ENEOS','ENI-AGIP','ENOC','ESSO','EUROL','EUROLUB','EURONOL','EUROPART','EUROREPAR','EUROTEC','EVEREST','EXTREME-A.M.G.','EXTREME-LUBRICANTS','FANFARO','FARET','FASTROIL','FAVORIT','FEBEST','FEBI','FEBI-BILSTEIN','FELIX','FIAT-ALFA-LANCIA','FILLINN','FMY','FORD','FORSAN','FQ','FUBAG','FUCHS','FURO','G-ENERGY','GAZ','GAZPROMNEFT','GENERAL MOTORS','GLANZ','GM','GNV','GRACE','GREAT-WALL-HAVAL','GT','GT-CRUISER','GT-OIL','GULF','GUNK','HANSE','HAVEN','HECK','HENGDA','HENGST','HESSOL','HI-GEAR','HIGHWAY','HINO','HONDA-ACURA','HUTER','HYUNDAI-GLOVIS','HYUNDAI-KIA-MOBIS','IDEMITSU','IGOL','IPONE','ISUZU','IVECO','JB-GERMAN-OIL','JTC','JURID','KAMAZ','KANSLER','KAWASAKI','KENDALL','KIXX','KMD','KORSON','KRAFTMAX','KROON-OIL','KUTTENKEULER','LADA','LAND-ROVER-JAGUAR','LAVR','LIFAN','LIQUI MOLY','LIQUI-MOLY','LIVCAR','LOTOS','LPR-AP','LUBEX','LUBREX','LUBRIGARD','LUKOIL','LUXE','MAG-1','MAG1','MAN','MANDO','MANNOL','MAPETROL','MARSHAL','MARSHALL','MAZDA','MECHANICAL BROTHERS','MEGUIN','MERCEDES-BENZ','MERCURY','MICKING','MIRAX','MITASU','MITSUBISHI','MOBIL','MOL','MOLY-GREEN','MOPAR','MOTORCRAFT','MOTOREX','MOTRIO','MOTUL','MOZER','MPM','NAFTAN','NAVIGATOR','NESTE','NGK-NTK','NGN','NISSAN-INFINITI','NOMAD','NORD-OIL','NORD-STREAM','NORDBERG','NOVONOL','OEM-OIL','OIL-RIGHT','OILOK','OILRIGHT','OILWAY','OLEX','OLIPES','ONZOIL','OPET','ORLEN-OIL','OSCAR','PATRON','PEAK','PEMCO','PENNASOL','PENNZOIL','PENRAY','PENTOSIN','PETRO-CANADA','PETROFER','PETROL-OFISI','PETROLL','PETRONAS','PEUGEOT-CITROEN','PHILLIPS-66','PILOTS','POLYMERIUM','PRISTA-OIL','PROFESSIONAL-HUNDERT','PROFI-CAR','PROFIX','Q8','QUAKER-STATE','QUICKSILVER','QVPRO','RAVENOL','RED-LINE','REDSKIN','REINWELL','RENAULT','REPSOL','RHEINOL','RINKAI','RINNOL','RIXX','ROLF','ROSNEFT','ROWE','RUSEFF','RUSSIA','S-OIL','SAMURAI','SAMURAI-GT','SCT','SEIKEN','SELENIA','SHELL','SHIKANA','SINTEC','SINTOIL','SONATEX','SPECTROL','SRS','SSANG-YONG','STARKRAFT','STATOIL','STELLOX','STEP-UP','STRONG-OIL','SUBARU','SUMICO','SUPROTEC','SUZUKI','SWAG','SWD-RHEINOL','SYNTIUM','SYNTIX','TAIF','TAKAYAMA','TAKUMI','TALER','TAMASHI','TANECO','TATNEFT','TCL','TEBOIL','TECHNO-POWER','TESLA','TEXACO','TEXOIL','THE-BEAST','TITAN','TNK','TOPCOOL','TOPRAN','TOTACHI','TOTAL','TOYOTA-LEXUS','TUTELA','UAZ','UNIL','UNITED','UNITED-OIL','UNIX','URANIA','VAG','VALVOLINE','VAPSOIL','VENDOR','VENOL','VERITY','VERTON','VITEX','VMPAUTO','VOLGA-OIL','VOLVO','WEGO','WEZZER','WOLF','WURTH','X-OIL','XADO','XENOL','XENUM','XIM','XPRO2','YACCO','YAMAHA','YOKKI','ZF','ZIC',]
classification_stop_list += brand_oil_list

def combineAPI(ss):
    ss = ss.split('|')
    lst = []
    for s in ss:
        lst.append(s)
        if not re.search(r'API[A-Za-z]{2}', s): continue
        lst.append(s.replace('API', "API "))
        lst.append(s.replace('API', ""))

    return '|'.join(set(lst))

def specificationCol(df):
    df['specification'] = df['Спецификация масла'].str.split(' Периодичность ', n=1, expand=True)[0].str.strip()
    sep_list = [' Норм. ', ' Если T°C', ' По регламенту:', ' подходящее для', ' Контроль: ', ' Проверка: ', ' Замена:', ' тяж. условия']
    for sep in sep_list:
        df['specification'] = df['specification'].str.split(sep, n=1, expand=True)[0].str.strip()



    ###
    # sep_list = ['выбор: ',]
    # for sep in sep_list:
    #     print(sep)
    #     # try:
    #     df['specification'] = df['specification'].str.split(sep, n=1, expand=True)[1].str.strip()
        # except:
        #     df['specification'] = df['specification'].replace(sep, '').str.strip()

    rep_1_dct = {'(Зелёный)':'Зелёный', '(зелёный)':'зелёный', '(Красный)':'Красный', 'красный':'красный'}
    for r in rep_1_dct:
        df['specification'] = df['specification'].str.replace(r, rep_1_dct[r])



    replace_list = ['Для регионов, где T опускается ниже – 20 °С:', 'При температуре от -30°C до 30°C:', 'При температуре менее -30°C: ','ИНФОРМАЦИЯ УТОЧНЯЕТСЯ', 'для регинов средней полосы', 'для северных и арктических регионов','для моделей с МКПП ', 'для моделей с АКПП ', 'не выше -38 °C','В КПП ', 'для моделей с 21.06.2010 ', 'для моделей до 21.06.2010 ','Для моделей до 2012 года: ', 'Для моделей 2015-2017 года выпуска ', 'Для моделей 2011-2015 года выпуска ','Для моделей 2021 - н.в. ','Для моделей 2018 - 2021 года ','Для моделей до 2012 года ',' или аналогичная ',' для алюминиевых радиаторов ', '- Периодичность замены: -','не требуется', 'Аналог: ', ' и выше для ', ' или ', ' для ', '(предпочтительно) ', 'Ниже -30 °C: ', ' Выше -30°C: ', ' -- -- -- -- --', 'На заводе: ',
                    'Для стран, с дизельным топливом с содержанием серы ниже 350 ppm Лучший выбор: ', 'Для моделей с фильтром DPF', 'Для моделей без фильтра DPF', 'выше',
                    ' Если T°C опускается ниже -30: ', ' Альтернатива: ', ' Если температура в регионе опускается ниже -30°C: ', 'Рекомендовано: ', 'С завода: ',
                    'Для моделей с сажевым фильтром DPF ', ' - - - - - - - - - - - - - -', 'Для моделей без сажевого фильтра DPF ', 'Лучший выбор: ',
                    'для моделей с сажевым фильтром CDPF ', 'для моделей без сажевого фильтра CDPF ', ' Альтернативы:',
                    'моделей без сажевого фильтра CDPF ', 'Для моделей с фильтром C.P.F. ', 'Для моделей без фильтра C.P.F. ', ' -- -- -- --', 'Для моделей с фильтром CPF ',
                    'Для моделей без фильтра CPF ', 'Ниже -30°C: ', ' Выше -30 °C: ', ' и', 'Требования: ', ' -40°C', ' Допуск: ', ' OEM: ', ' Аналоги:', ' - -',
                    ' (аналог ', ') ', 'норм. условия', 'Для очень низких температур: ', 'для КПП с двойным сцеплением)', 'он же ', 'Цвет:', 'Информация уточняется',
                    'OEM:', 'для', 'класс', 'Для мехатроника КПП: ', 'Для КПП:', 'информация уточняется', 'Для', 'С сажевым фильтром (DPF):', 'Без сажевого фильтра|DPF):',
                    '(Старый):', '(Новый):']
    replace_list += [f"Для регионов, где T опускается ниже – {x} °С: " for x in range(10, 50, 5)]
    replace_list += [f'Бензин АИ {x}' for x in range(75, 110, 5)]
    for r in replace_list:
        df['specification'] = df['specification'].str.replace(r, '|')

    rep_d_list = ['FUCHS', 'ESSO','Fuchs','TOTAL', 'ACURA ATF ', 'NISSAN ATF ','TOYOTA ATF ', 'BMW ATF ', 'BMW MTF', 'ACURA ATF', 'VAG ', 'VW ', 'HONDA ATF', 'BMW ', 'Castrol ', 'AISIN ', 'API ',
                  'CVTF ', 'MERCON ATF', 'HYUNDAI ATF ', ' MERCON ULV', ' Motorcraft']
    makes_list = ['ACURA','AUDI','BMW','CHANGAN','CHERY','CHEVROLET','DAEWOO','DAIHATSU','DATSUN','EXEED','FAW','FORD','GAC','GEELY','GREAT WALL','HAVAL','HONDA','HYUNDAI','JAC','JAECOO','JETOUR','KIA','LADA','LEXUS','LIFAN','LIXIANG','MAZDA','MERCEDES','MITSUBISHI','NISSAN','OMODA','OPEL','RENAULT','SKODA','SSANGYONG','SUBARU','SUZUKI','TANK','TOYOTA','UAZ','VOLKSWAGEN','VOYAH','ZEEKR',]
    rep_d_list += makes_list

    rep_d_dct = {x:f"|{x}" for x in rep_d_list}
    replace_dct = {'(G12++)':'|G12++|','(G12+)':'|G12+|',', API ': '|API' , ', SAE ':'|SAE ', ' SAE ':'|SAE ', ' API ':'|API ', ', ACEA ':'|ACEA ', 'Honda ULTRA ':'|Honda ULTRA ', "| ":'|', " |":'|',
                   ' Ford ':'|Ford ',
                   ' ILSAC ':'|ILSAC ', ' GM ': '|GM ', "|||||":'|', "||||":'|', "|||":'|', "||":'|', 'Литиевая смазка|шасси':'Литиевая смазка для шасси',
                   ' Литиевая смазка':'|Литиевая смазка'}
    rep_d_dct.update(replace_dct)
    for r in rep_d_dct:
        df['specification'] = df['specification'].str.replace(r, rep_d_dct[r])


    rep_2_list = [' (', ') ']
    rep_2_dct = {x: f"|" for x in rep_2_list}
    for r in rep_2_dct:
        df['specification'] = df['specification'].str.replace(r, rep_2_dct[r])
    df['specification'] = df['specification'].str.strip()

    def bracketDrop(ss):

        pattern = r"\([^)]*\)"  # всё между "(" и ")" включая пробелы

        return re.sub(pattern, "", ss)

    def yearInrevalDrop(ss):
        pattern = r"Для моделей\s+[0-9\-–]+\s+года выпуска"

        return re.sub(pattern, "", ss)

    def pattern3(ss):
        pattern_list = [r"моделей до\s+(?:\d{1,2}\.\d{1,2}\.\d{4}|\d{4})", r"[Мм]одел(?:и|ей)\s+с\s+\d{4}\s+года",
                        r"Для регионов, где T опускается ниже\s*[–\-]?\s*\d+\s*°С\s*:?", r'до\s*-\s*\d+\.?\d*',
                        r'модели\s+с\s+\d{4}\s*г\b', r'для\s+[A-Z0-9]+?\)', r'от\s*-\d+', r'до\s*-\d+', r'Для\s+моделей\s+\d{4}\s*-\s*\d{4}\s+года',
                        r'Для\s+моделей\s+\w+\s+года', r'Для\s+моделей\s+до\s+\d{4}\s+года:', r'Для\s+моделей\s+с\s+\d{4}\s+года:',
                        r'моделей\s+\d{4}\s*-\s*\d{4}\s+года', r'моделей\s+до\s+\d{4}\s+года:', r'моделей\s+до\s+\d{2}\.\d{2}\.\d{4}']
        for pattern in pattern_list:
            re.sub(pattern, "", ss)
        return ss

    df['specification'] = df['specification'].apply(bracketDrop)
    df['specification'] = df['specification'].apply(yearInrevalDrop)
    df['specification'] = df['specification'].apply(pattern3)
    df['specification'] = df['specification'].str.replace('|5|', '')


    ####  масла - жидкости

    rep_list = ['DOT 4', 'DOT 3', 'Liqui Moly', 'Pentosin', 'Sinopec']
    rep_dct = {x: f"|{x}" for x in rep_list}
    for r in rep_dct:
        df['specification'] = df['specification'].str.replace(r, rep_dct[r])

    df['specification'] = df['specification'].str.upper()

    df['specification'][(df['specification'] == "-") | (df['specification'] == '|')] = ''

    def normalize(ss):
        if not ss: return ''

        ss = ss.replace(' / ', '|')
        ss = ss.replace(',', '|')
        ss = ss.replace('|   ', '|')
        ss = ss.replace('|  ', '|')
        ss = ss.replace('| ', '|')
        ss = ss.replace('   |', '|')
        ss = ss.replace('  |', '|')
        ss = ss.replace(' |', '|')
        ss = ss.replace(')', '|')
        ss = ss.split('|')
        return '|'.join([s for s in ss if s])

    df['specification'] = df['specification'].apply(normalize)

    def combine(ss):
        ss = ss.split("|")
        lst = []
        for s in ss:
            if not s: continue

            if s in classification_stop_list:
                continue
            lst.append(s)
            s1 = U24.strEngDigRus(s, sep='', add_smb_list=['+'])
            lst.append(s1)
            if re.search(r'\d+W', s):
                if "-" not in s:
                    s1 = s.replace('W', 'W-')
                    lst.append(s1)
                if 'SAE' not in s:
                    s1 = 'SAE ' + s
                    s1 = ' '.join(s1.split())
                    lst.append(s1)
                else:
                    s1 = s.replace('SAE', '')
                    s1 = ' '.join(s1.split())
                    lst.append(s1)


        ss = []
        for s in lst:
            ss.append(s)
            s1 = U24.strEngDigRus(s, sep='')
            ss.append(s1)
        ss = list(set(ss))

        return '|'.join(ss)


    df['specification'] = df['specification'].apply(combine)
    df['specification'] = df['specification'].apply(combineAPI)



    return df


def df_2_Specifications(df):
    spec_df = U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/car_oil_catalog_params.xlsx')
    spec_unique = spec_df['specification'].unique()
    sae_lst = ['0W-20','0W-30','0W-40','10W-30','10W-40','10W-50','10W-60','15W-40','15W-50','20W-20','20W-40','20W-50','20W-60','25W-40','5W-16','5W-20','5W-30','5W-40','5W-50',]
    spec_unique = list(spec_unique) + sae_lst

    spec_lst = []

    for sp in spec_unique:
        for s in str(sp).split("|"):
            if not s or s in classification_stop_list: continue
            spec_lst.append(s.replace('/ ', '').replace('(', '').replace(' /', ''))

    spec_lst = list(set(spec_lst))


    def findSpec(ss):
        ss = " " + str(ss) + " " + U24.strEngDigRus(str(ss), sep='', add_smb_list=['+']) + " "
        ss = ss.upper()
        lst = []
        for spec in spec_lst:
            spec_x = spec if len(spec) > 3 else f" {spec} "
            if spec_x in ss:
                lst.append(spec)


        dct_remove = {'10W30':'0W30', '10W-30':'0W-30', '15W40':'5W40', '15W-40':'5W-40', '15W-50':'5W-50', '15W50':'5W50'}
        for x, y in dct_remove.items():
            if x in lst and y in lst:
                lst.remove(y)

        lst = set(lst)
        lst_1 = []
        for s in lst:
            lst_1.append(s)
            if re.search(r'\d+W', s):
                if "-" not in s:
                    s1 = s.replace('W', 'W-')
                    lst_1.append(s1)
                if 'SAE' not in s:
                    s1 = 'SAE ' + s
                    s1 = ' '.join(s1.split())
                    lst_1.append(s1)
                else:
                    s1 = s.replace('SAE', '')
                    s1 = ' '.join(s1.split())
                    lst_1.append(s1)


        lst = set(lst_1)
        return '|'.join(lst)

    df = U24.findTovgruppa(df, column_to_scan='control', contact_how='left')
    df = U24.litersFind(df, col_cross_bart=False, col_global_group='cross_global_group')
    df['liters'] = df['liters'].str.replace(",", '.')
    df['liters'] = df['liters'].str.split(' ', n=1, expand=True)[0]

    # Заменяем пустые строки и 'nan' на NaN
    df['liters'] = df['liters'].replace(['', 'nan', 'None'], np.nan)

    # Заполняем из колонки "Объем упаковки, л"
    df['liters'] = df['liters'].fillna(df['Объем упаковки, л'])

    # Теперь безопасно приводим к float
    df['liters'] = pd.to_numeric(df['liters'], errors='coerce')

    # Опционально: если остались NaN — можно заполнить 0
    df['liters'] = df['liters'].fillna(0)

    df['liters'][df['liters'] == ''] = 0

    df['liters'] = df['liters'].astype(float)
    df = df[df['liters'] <=30].reset_index(drop=True)



    df['control'] = df['control'] + df['cross_tovgruppa']

    df['specifications_1'] = df['control'].apply(findSpec)
    df['specifications_1'] = df['specifications_1'].apply(combineAPI)
    df = df[df['specifications_1'] != '']

    df['len'] = 95 - df['name'].str.len()
    df['len'][df['len'] < 13] = 0
    df['len'][df['specifications'] == ''] = 0

    df['name'][df['len'] > 0] = df['name'] + " | " + df.apply(lambda x: x['specifications'][:x['len']].split('.')[0], axis=1)


    U24.xlsxSave(df, root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/all_oil_assortment.xlsx')

    # U24.tmp_saveOut(df)



if __name__ == '__main__':
    # df = U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/car_oil_catalog_params.xlsx')
    # specificationCol(df)

    df = U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/input_all_oil_goods.xlsx')
    df.fillna('', inplace=True)

    df['control'] = df['name'].astype(str) + ' ' + df['Вязкость'].astype(str)
    df['len'] = 200 - df['name'].str.len()
    df['len'][df['len'] < 7] = 0

    df['control'] += " " + df.apply(lambda x: x['specifications'][:x['len']], axis=1)



    df_2_Specifications(df)

