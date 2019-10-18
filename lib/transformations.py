import datetime
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
from tqdm import tqdm
from lib import postgres as pg

def serie_fft(serie, frequency, negative_freq):
    
    fft = sp.fftpack.fft(serie)
    psd = np.abs(fft) ** 2
    fft_freq = sp.fftpack.fftfreq(len(psd), frequency)
    
    if negative_freq:
        return fft_freq, 10 * np.log10(psd)
    else:
        i = fft_freq > 0 
        return fft_freq[i], 10 * np.log10(psd[i])

def window_fft(source_schema, source_table, source_column, window_size, frequency, negative_freq = False):
    
    ids_sql = """
        SELECT
            _id
        FROM 
            {}.{}
        WHERE
            _id >= {}
        ORDER BY 1 ASC
        ;
        """.format(source_schema, source_table, window_size)
    ids = pg.load_query_to_df(ids_sql)
    
    features = []
    for _id in tqdm(ids['_id'].values):
        row = {}
        
        serie_sql = """
            SELECT
                {}
            FROM 
                {}.{}
            WHERE
                _id <= {}
            ORDER BY 1 DESC
            LIMIT {}
            ;
            """.format(source_column, source_schema, source_table, _id, window_size)
        df = pg.load_query_to_df(serie_sql)
        
        serie = df[source_column]
        fft_freq, psd_db = serie_fft(serie, frequency, negative_freq)
        
        row['_id'] = _id
        for j in range(1,len(fft_freq)):
            row['f_'+str(fft_freq[j]).replace('.','_')] = psd_db[j]
            
        features.append(row)
    
    return pd.DataFrame(features)