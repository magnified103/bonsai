import os

import pandas as pd
import sklearn.model_selection
from tqdm import tqdm

features = ['source_asn', 'dest_asn', 'source_lon', 'source_lat', 'dest_lon', 'dest_lat', 'distance',
            'MeanUpload_src', 'MeanDownload_src', 'MeanUpload_dst', 'MeanDownload_dst', 'hop_p50']
target = ['rtt_p50']


def prepare_data(data_dir, uploads, downloads, traces):
    global features
    global target
    # load aggPings
    # load mlab upload & download
    # load graph_traces
    # join all data and remove rows with NaN
    pings = pd.read_csv(os.path.join(data_dir, 'aggPings.csv'), keep_default_na=False,
                        na_values=['Unknown', '', -1]).dropna()
    # Remove unwanted values
    pings = pings[pings['rtt_count'] >= 30]  # remove measurements with less than 30 pings (not enough data)
    pings = pings[pings['rtt_std'] < (pings['rtt_p50'] / 2)]  # remove measurements with high variance (too much noise)
    pings = pings[((pings['distance'] / pings[
        'rtt_p50']) / 2 < 300)]  # remove impossible values (who has a 300km/ms connection?) rtt is back and forth, so we divide by 2

    # Join Bandwidth data
    pings = pings.merge(uploads[['MeanUpload']], left_on='source', right_index=True)
    pings = pings.merge(downloads[['MeanDownload']], left_on='source', right_index=True)
    pings = pings.rename(columns={'MeanUpload': 'MeanUpload_src', 'MeanDownload': 'MeanDownload_src'})
    pings = pings.merge(uploads[['MeanUpload']], left_on='destination', right_index=True)
    pings = pings.merge(downloads[['MeanDownload']], left_on='destination', right_index=True)
    pings = pings.rename(columns={'MeanUpload': 'MeanUpload_dst', 'MeanDownload': 'MeanDownload_dst'})

    # Join Graph data
    traces = traces[traces['count'] >= 5][['source', 'source_asn', 'destination', 'dest_asn', 'hop_p50']]
    traces = traces[traces['hop_p50'] < 100]
    pings = pings.merge(traces, left_on=['source', 'source_asn', 'destination', 'dest_asn'],
                        right_on=['source', 'source_asn', 'destination', 'dest_asn'])

    return pings[features + target]


def load_data(data_dir, sample_size=-1):
    uploads = pd.read_csv(os.path.join(data_dir, 'mlab-ndt-uploads.csv'), index_col=['CountryCode'],
                          keep_default_na=False, na_values=['']).dropna()
    downloads = pd.read_csv(os.path.join(data_dir, 'mlab-ndt-downloads.csv'), index_col=['CountryCode'],
                            keep_default_na=False, na_values=['']).dropna()
    traces = pd.read_csv(os.path.join(data_dir, 'graphTraces.csv'), keep_default_na=False, na_values=[-1]).dropna()

    pings = prepare_data(data_dir, uploads, downloads, traces)
    if sample_size > 0:
        pings = pings.sample(n=sample_size, random_state=42)
    train, test = sklearn.model_selection.train_test_split(pings, test_size=0.2, random_state=42)
    return (train[features].to_numpy(), train[target].to_numpy()), (test[features].to_numpy(), test[target].to_numpy())


def load_all_data(data_dir, data_file=None, sample_size=-1, return_data=False):
    global features
    global target

    if data_file is None:
        uploads = pd.read_csv(os.path.join(data_dir, 'mlab-ndt-uploads.csv'), index_col=['CountryCode'],
                              keep_default_na=False, na_values=['']).dropna()
        downloads = pd.read_csv(os.path.join(data_dir, 'mlab-ndt-downloads.csv'), index_col=['CountryCode'],
                                keep_default_na=False, na_values=['']).dropna()

        pings_dir = os.path.join(data_dir, 'pings')
        trace_dir = os.path.join(data_dir, 'traceroute')
        data = []
        for day in tqdm(os.listdir(pings_dir), desc='Loading data...'):
            day_dir = os.path.join(pings_dir, day)
            traces_day_dir = os.path.join(trace_dir, day)
            traces = pd.read_csv(os.path.join(traces_day_dir, 'netGraph.csv'), keep_default_na=False,
                                 na_values=[-1]).dropna()

            pings = prepare_data(day_dir, uploads, downloads, traces)
            data.append(pings)

        print('Concatenating data...')
        data = pd.concat(data)

    else:
        data = pd.read_csv(data_file, keep_default_na=False, na_values=['Unknown', '', -1])
        data = data[data['hop_p50'] < 100]

    if sample_size > 0:
        #print('Sampling data...')
        data = data.sample(n=sample_size, random_state=42)
    #print('Splitting data into train and test...')
    dup_data = data[data.duplicated(features, keep=False)]
    if len(dup_data) > 0:
        print('Averaging duplicate data...')
        group = data.groupby(features)
        data = group.mean()
        data = data.reset_index()

    train, test = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=42)
    if return_data:
        return (train[features].to_numpy(), train[target].to_numpy()), (test[features].to_numpy(), test[target].to_numpy()), data
    return (train[features].to_numpy(), train[target].to_numpy()), (test[features].to_numpy(), test[target].to_numpy())


if __name__ == '__main__':
    _, _, data = load_all_data('/Volumes/T7/data/bonsai/ripe-training', return_data=True)
    print(data.head())
    print(data.shape)
    data.sample(n=10000).to_csv('data/small_sample_data.csv.gz', index=False)
    data.sample(frac=0.1).to_csv('data/10_percent_sample_data.csv.gz', index=False)
    data.to_csv('data/all_data.csv.gz', index=False)