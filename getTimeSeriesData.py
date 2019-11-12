import json
import requests
import csv
import pandas as pd


# get the past n days worth of time series data from the S&P 100, store in json
def get_n_days_time_series(n):
    sp100 = pd.read_csv('sp100.csv')
    sp100_symbols = sp100['Symbol'].tolist()
    sp100_names = sp100['Name'].tolist()
    sp100_common_names= sp100['CommonName'].tolist()

    request_base = 'https://api.worldtradingdata.com/api/v1/history?symbol='
    request_tail = '&sort=newest&api_token='
    api_token='2nEq2CAjKkyA7Y60aqSYvr0oieBinSEhwRRTF3lq890gBy2uITqZXboIsgS6'

    symbol_to_request_dict = {}

    first_day = ''
    last_day = ''
    for symbol in sp100_symbols:
        url = request_base + symbol + request_tail + api_token
        r = requests.get(url)
        data = r.json()
        past_n_days = {}
        count = 0
        for k,v in data['history'].items():
            if(count == 0):
                first_day = str(k)
            past_n_days[k] = v
            count += 1
            if(count == n):
                last_day = str(k)
                break

        symbol_to_request_dict[symbol] = past_n_days

    with open('time_series_data_from=' + last_day + 'to=' + first_day + '.json', 'w') as fp:
        json.dump(symbol_to_request_dict, fp)

n = 90
get_n_days_time_series(n)