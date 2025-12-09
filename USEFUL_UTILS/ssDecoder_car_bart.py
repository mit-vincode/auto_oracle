import time, string
import pandas as pd
import re

from bootstrap import *


class ss_2_CarParams():


    def __init__(self, ):
        self.RESULT_ATTRIBUTES = {
            'tovgruppa', 'tovgruppa_rule', 'global_group',
            'car', 'make', 'year', 'fuel', 'volume', 'sae',
        }

        self._reset_results()

        self.kw_cars = U24.data2Df_upload(root_path + '/DATA_CATALOGS/KeyWords_Xlsx/mgpt_KeyWords_CARS.xlsx')
        self.kw_makes = U24.data2Df_upload(root_path + '/DATA_CATALOGS/KeyWords_Xlsx/mgpt_KeyWords_MAKE.xlsx')
        self.kw_parts = U24.data2Df_upload(root_path + '/DATA_CATALOGS/KeyWords_Xlsx/mgpt_KeyWords_PARTS.xlsx')

        self.list_volume = ['0.7', '0.8', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '11.0',
                       '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.2', '3.3', '3.4',
                       '3.5', '3.6', '3.7', '3.8', '4.0', '4.2', '4.4', '4.5', '4.6', '4.7', '4.8', '5.0', '5.4', '5.5',
                       '5.6', '5.7', '6.0', '6.2', '8.6']
        self.list_volume += [x.replace('.', ',') for x in self.list_volume]
        self.list_volume += [f'{x} лит' for x in range(1, 12)]
        self.dct_volume = {x: x.split(' ')[0] for x in self.list_volume}


    def _reset_results(self):
        for field in self.RESULT_ATTRIBUTES:
            setattr(self, field, None)



    def extract_year(self, ss):
        match = re.search(r'(?<!\d)(199[8-9]|20[0-3][0-5])(?!\d)', str(ss) or '')
        return int(match.group(0)) if match else None

    def extract_fuel(self, ss):
        x = ss.lower()
        if 'электр' in x: return 'электричество'
        if any(y in x for y in ['бенз. ', 'бензин']): return 'бензин'
        if any(y in x for y in ['диз. ', 'tdi']): return 'дизель'

        return None

    def extract_volume(self, ss):


        for v in self.list_volume:
            if v in ss.lower():
                return self.dct_volume[v]

        return None




    def ssDecoder(self, ss):

        self._reset_results()


        df = pd.DataFrame({'name': [ss]})

        if len(tovgruppa := U24.findTovgruppa(df[['name']], external_KeyWords=self.kw_parts, del_rule=False, print_info=False)):
            self.tovgruppa = tovgruppa.loc[0, 'cross_tovgruppa']
            self.global_group = tovgruppa.loc[0, 'cross_global_group']
            if self.global_group == 'масло моторное' and 'w-' in self.tovgruppa:
                self.sae = self.tovgruppa.split(' ')[-1]


            self.tovgruppa_rule = tovgruppa.loc[0, 'Number_the_rule']

        if len(car := U24.findTovgruppa(df[['name']], external_KeyWords=self.kw_cars, print_info=False)):
            self.car = car.loc[0, 'cross_tovgruppa']
            self.make = car.loc[0, 'cross_global_group']

        elif len(make := U24.findTovgruppa(df[['name']], external_KeyWords=self.kw_makes, print_info=False)):
            self.make = make.loc[0, 'cross_tovgruppa']


        self.year = self.extract_year(ss)
        self.fuel = self.extract_fuel(ss)
        self.volume = self.extract_volume(ss)



if __name__ == '__main__':
    ss = 'какое масло моторное 5w30 для  джили монджаро 2.3 бензиновый 2013 года порекомендуете?'
    CarAttr = ss_2_CarParams()

    CarAttr.ssDecoder(ss)
    print("\n-----  -----  ----- result  -----  ----- -----")

    print(f"\ntovgruppa = {CarAttr.tovgruppa}, tovgruppa_rule = {CarAttr.tovgruppa_rule}")
    print(f"\ncar = {CarAttr.car}")
    print(f"\nmake = {CarAttr.make}")
    print(f"\nfuel = {CarAttr.fuel}")
    print(f"\nvolume = {CarAttr.volume}")
    print(f"\nyear = {CarAttr.year}")
    print(f"\nsae = {CarAttr.sae}")


    ""


