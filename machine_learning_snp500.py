import pandas as pd


# Model built on a single company/ticker. Will find the percent change values
# for the next 7 days and take into account all company percent changes that day
def process_data_for_labels(ticker):
    how_days = 7
    df = pd.read_csv('sp500_joined_adj_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, how_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    print(df.head())
    return tickers, df

# Generate our labels, whether we want to buy sell or hold
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = .02
    for col in cols:
        # Buy. 2% positive price change
        if col > requirement:
            return 1
        # Sell. 2% negative price change
        if col < -requirement:
            return -1
    # Hold Price hasn't dropped or raised 2%
    return 0

process_data_for_labels('MMM')