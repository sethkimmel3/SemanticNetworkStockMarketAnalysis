import json
import requests
import csv
import pandas as pd

def get_market_caps():
    sp100 = pd.read_csv('sp100.csv')
    sp100_symbols = sp100['Symbol'].tolist()

    request_base = 'https://api.worldtradingdata.com/api/v1/stock?symbol='
    request_tail = '&api_token='
    api_token = '2nEq2CAjKkyA7Y60aqSYvr0oieBinSEhwRRTF3lq890gBy2uITqZXboIsgS6'

    symbol_to_market_cap_dict = {}

    for symbol in sp100_symbols:
        url = request_base + symbol + request_tail + api_token
        r = requests.get(url)
        data = r.json()

        symbol_to_market_cap_dict[symbol] = data['data'][0]['market_cap']

    with open('market_cap_data_11_20_19.json', 'w') as fp:
        json.dump(symbol_to_market_cap_dict, fp)

get_market_caps()