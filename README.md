# wharton-z-score
Wharton Investment Competition Z-score calculator

pip3 install pandas yfinance
mkdir -p data
python3 fetch.py [tickers] --out data/companies.csv
python3 screen.py data/companies.csv --top 20