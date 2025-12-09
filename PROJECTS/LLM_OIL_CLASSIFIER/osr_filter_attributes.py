# osr_filter_attributes.py
# Скрипт для формирования RAG + SFT датасетов
# из файла 1_paramClasifier.xlsx


import pandas as pd
import json
from pathlib import Path

from bootstrap import *

path_from = root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/car_oil_catalog_params.xlsx'
df_base = U24.data2Df_upload(path_from)


# ======================= УТИЛИТЫ =======================
def matches_any(value, composite_str, sep='|'):
    """Проверка: есть ли value хотя бы в одном элементе строки, разделённой sep"""
    if pd.isna(composite_str) or str(composite_str).strip() in ['0', 'nan', '', 'none']:
        return True
    parts = [p.strip() for p in str(composite_str).split(sep) if p.strip()]
    return str(value).strip().lower() in [p.lower() for p in parts]

# ======================= ОСНОВНАЯ ФУНКЦИЯ СВЁРТКИ =======================
def filter_car_params(params: dict):
    """
    Последовательная фильтрация с откатом при обнулении.
    """

    df = df_base.copy()
    applied_filters = []

    filter_order = [
        ('global_group', lambda v: df['global_group'].str.contains(str(v).lower(), na=False)),
        ('tovgruppa', lambda v: df['tovgruppa'].str.contains(str(v).lower(), na=False)),
        ('car',       lambda v: df['car'] == str(v).lower()),
        ('make',      lambda v: df['make'] == str(v).lower()),
        ('fuel',      lambda v: df['fuel'] == str(v).lower()),
        ('year',      lambda v: df['year_interval'].apply(lambda x: matches_any(v, x))),
        ('volume',    lambda v: (
            pd.to_numeric(df['volume'], errors='coerce').between(float(v)-0.1, float(v)+0.1)
            if str(v).replace('.','').isdigit() else
            (df['volume'] == str(v))
        )),
        ('sae',       lambda v: df['sae'].apply(lambda x: matches_any(v.replace('w','W'), x.upper()))),
    ]


    for attr_name, mask_func in filter_order:
        val = params.get(attr_name)
        if val in [None, '', 'none']:
            continue

        try:
            mask = mask_func(val)
        except:
            continue  # если что-то сломалось — просто пропускаем

        filtered = df[mask]

        if len(filtered) == 0:
            # Откатываем фильтр — он слишком жёсткий
            continue
        else:
            df = filtered
            applied_filters.append(attr_name)

    return df, applied_filters
