import pandas as pd

from bootstrap import *
BONUS_TVG = 1
LITERS_LIMIT = 10

def get_assortmentDf():
    df_assortment = U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/all_oil_assortment.xlsx')
    df_assortment['len'] = df_assortment['specifications_1'].str.len()
    df_assortment.sort_values(['len'], ascending=False, inplace=True)
    df_assortment.reset_index(drop=True, inplace=True)
    df_assortment['name'] = "– " + df_assortment['brand'] + " | " + df_assortment['name']
    df_assortment = df_assortment[['brand', 'name', 'cross_tovgruppa', 'cross_global_group', 'specifications_1', 'liters']]
    df_assortment['liters'] = df_assortment['liters'].astype(int)
    df_assortment['name'][df_assortment['liters'] != 0] = df_assortment['name'] + " | Объём L: " + df_assortment['liters'].astype(str)
    df_assortment.sort_values(['name'], ascending=False, inplace=True)
    df_assortment.reset_index(drop=True, inplace=True)

    return df_assortment

def get_dct_brand_Bonus():
    df = U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/brand_oil_bonus.xlsx')
    return pd.Series(df['bonus'].values, index=df['brand'].values).to_dict()

df_assortment = get_assortmentDf()
dct_brand_bonus = get_dct_brand_Bonus()
OE_list = BAP.getOElist()

def spec_rating(x, specification, x_tovgruppa, tovgruppa, brand, brand_own):
    rating = len(set(set(x.split('|')) & set(specification.split('|'))))
    if not rating: return 0

    if (rating > 2) and (brand in dct_brand_bonus):
        rating += dct_brand_bonus[brand]


    if brand == brand_own:
        rating += 1

    elif brand in OE_list:
        rating -= 1
        rating = rating if rating >= 0 else 0


    if x_tovgruppa == tovgruppa:
        rating += BONUS_TVG

    return rating

def select_Oil_assortment(param_df, limit_row = 4):
    param_df.drop_duplicates(subset=['specification'], keep='first', inplace = True)
    param_df.reset_index(drop=True, inplace=True)

    res_df = pd.DataFrame()

    prev_tovgruppa = 0
    for i, specification in enumerate(param_df['specification']):

        global_group = param_df.loc[i, 'global_group']
        tovgruppa = param_df.loc[i, 'tovgruppa']
        vehicle_type = param_df.loc[i, 'vehicle_type']
        brand_own = param_df.loc[i, 'brand_own']

        tdf = df_assortment[df_assortment['cross_global_group'] == global_group].reset_index(drop=True)
        if not len(tdf):
            continue


        #LITERS_LIMIT
        if vehicle_type in ['passenger', 'LCV']:
            filtered_tdf = tdf[tdf['liters'] <= LITERS_LIMIT]
            if len(filtered_tdf):
                tdf = filtered_tdf.copy()

        tdf['rating'] = tdf.apply(lambda x: spec_rating(x['specifications_1'], specification,
                                                        x['cross_tovgruppa'], tovgruppa,
                                                        x['brand'], brand_own), axis = 1)
        tdf = tdf[tdf['rating'] >0]
        if not len(tdf):
            continue
        tdf.drop_duplicates(subset=['brand', 'specifications_1'], keep='first', inplace=True)
        tdf.sort_values(['rating'], ascending=False, inplace=True)


        # заголовки блоков спецификаций: масло моторное / антифриз / жидкость тормозная
        tdf['cross_tovgruppa'] = tovgruppa
        if prev_tovgruppa != tovgruppa:
            res_df = pd.concat([res_df, pd.DataFrame({'name':["\n" + tovgruppa + ':'], 'cross_tovgruppa':['']}),tdf[:limit_row]])
        else:
            res_df = pd.concat([res_df, tdf[:limit_row]])


    if not len(res_df):
        return ''


    res_df.drop_duplicates(subset=['name', 'cross_tovgruppa'], inplace=True)
    res_lst = res_df['name'].tolist()

    return '\n'.join(res_lst)