from flask import Flask, render_template, request, jsonify
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import requests
import re
from transformers import pipeline
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from dash import dcc, html
import plotly.express as px
import json
import plotly
import plotly.graph_objs as go
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import price
import os
import json
import plotly.io as pio
from datetime import datetime
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import re
from transformers import pipeline
import csv


app = Flask(__name__)  # creating the Flask class object
count = 0


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # Energy
        a = tf.nn.softmax(e, axis=1)  # Attention weights
        output = tf.reduce_sum(x * a, axis=1)  # Weighted sum of input features
        return output

# 加载预训练的Keras LSTM模型
model = load_model(f'../models/best_erp_lstm_model_20241015_1810.keras', custom_objects={'Attention': Attention})

def price():
    crypto_currency = 'BTC'
    against_currency = 'USD'
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()
    data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)
    # prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 60
    future_day = 30
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Create Neural Network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    # Testing the model
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
    print(test_data)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        print(x)
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)

    prediction_prices = scaler.inverse_transform(prediction_prices)

    prices = [item for sublist in prediction_prices for item in sublist]

    print(prices)
    df = pd.DataFrame({'prices': prices})

    df.to_csv('result.csv', index=False)
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(prediction_prices, color='green', label='Predicted Prices')
    plt.title(f'{crypto_currency} price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    return prices

def create_plot():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    print("current_time: ", current_time)
    if current_time=="09:00" or current_time=="09:03":
        print("true")
        if os.path.exists("result.csv"):
           os.remove("result.csv")
        price()

    df2 = pd.read_csv("result.csv")
    fig = px.line(df2, y='sales',
              title='trends prediction',template="plotly_dark")
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(width=1100, height=450)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON



def create_output_array(summaries, scores, urls):
    output = []
    for ticker in ['BTC']:
        for counter in range(len(summaries[ticker])):
            output_this = [
                            ticker,
                            summaries[ticker][counter],
                            scores[ticker][counter]['label'],
                            scores[ticker][counter]['score'],
                            urls[ticker][counter]
                          ]
            output.append(output_this)
    return output



count = 0

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sentiment')
def sentiment():
    results = []
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    print(current_time)
    if current_time == "01:17" or current_time == "01:18":
        if os.path.exists("ethsummaries.csv"):
           os.remove("ethsummaries.csv")
        # sentiment

        # 2. Setup Model
        model_name = "human-centered-summarization/financial-summarization-pegasus"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        # 3. Setup Pipeline
        monitored_tickers = ['BTC']

        # 4.1. Search for Stock News using Google and Yahoo Finance
        print('Searching for stock news for', monitored_tickers)

        def search_for_stock_news_links(ticker):
            search_url = 'https://www.google.com/search?q=yahoo+finance+{}&tbm=nws'.format(ticker)
            r = requests.get(search_url)
            soup = BeautifulSoup(r.text, 'html.parser')
            atags = soup.find_all('a')
            hrefs = [link['href'] for link in atags]
            return hrefs

        raw_urls = {ticker: search_for_stock_news_links(ticker) for ticker in monitored_tickers}

        # 4.2. Strip out unwanted URLs
        print('Cleaning URLs.')
        exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

        def strip_unwanted_urls(urls, exclude_list):
            val = []
            for url in urls:
                if 'https://' in url and not any(exc in url for exc in exclude_list):
                    res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                    val.append(res)
            return list(set(val))

        cleaned_urls = {ticker: strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}

        # 4.3. Search and Scrape Cleaned URLs
        print('Scraping news links.')

        def scrape_and_process(URLs):
            ARTICLES = []
            for url in URLs:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                results = soup.find_all('p')
                text = [res.text for res in results]
                words = ' '.join(text).split(' ')[:350]
                ARTICLE = ' '.join(words)
                ARTICLES.append(ARTICLE)
            return ARTICLES

        articles = {ticker: scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}

        # 4.4. Summarise all Articles
        print('Summarizing articles.')

        def summarize(articles):
            summaries = []
            for article in articles:
                input_ids = tokenizer.encode(article, return_tensors="pt")
                output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
                summary = tokenizer.decode(output[0], skip_special_tokens=True)
                summaries.append(summary)
            return summaries

        summaries = {ticker: summarize(articles[ticker]) for ticker in monitored_tickers}

        # 5. Adding Sentiment Analysis
        print('Calculating sentiment.')
        sentiment = pipeline("sentiment-analysis")
        scores = {ticker: sentiment(summaries[ticker]) for ticker in monitored_tickers}

        # # 6. Exporting Results
        print('Exporting results')
        final_output = create_output_array(summaries, scores, cleaned_urls)
        with open('ethsummaries.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerows(final_output)

    with open("ethsummaries.csv") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:
            results.append(row)
    return render_template('sentiment.html', final_output=results, count=count, length=len(results))


@app.route('/news')
def news():
    news_data = requests.get('https://min-api.cryptocompare.com/data/v2/news/?lang=EN').json()
    news_articles = news_data['Data']
    return render_template('news.html', news_articles=news_articles)


@app.route('/prediction')
def price_prediction():
    plot = create_plot()
    json_data = '''
    [
        {
            "id": "blender123",
            "symbol": "bl123",
            "name": "Blender Pro",
            "image": "https://coin-images.coingecko.com/coins/images/1/large/bitcoin.png?1696501400",
            "current_price": 129.99,
            "market_cap": 3520000,
            "market_cap_rank": 1,
            "fully_diluted_valuation": 5000000,
            "total_volume": 15800,
            "high_24h": 135.50,
            "low_24h": 125.00,
            "price_change_24h": -3.20,
            "price_change_percentage_24h": -2.40,
            "market_cap_change_24h": -85000,
            "market_cap_change_percentage_24h": -2.36,
            "circulating_supply": 27000,
            "total_supply": 30000,
            "max_supply": 30000,
            "ath": 150.00,
            "ath_change_percentage": -13.33,
            "ath_date": "2024-02-15T09:30:00.000Z",
            "atl": 79.99,
            "atl_change_percentage": 62.50,
            "atl_date": "2023-08-10T07:20:00.000Z",
            "roi": null,
            "last_updated": "2024-10-16T06:25:41.652Z"
        },
        {
            "id": "vacuum456",
            "symbol": "vac456",
            "name": "Vacuum Master",
            "image": "https://coin-images.coingecko.com/coins/images/1/large/bitcoin.png?1696501400",
            "current_price": 259.99,
            "market_cap": 6780000,
            "market_cap_rank": 2,
            "fully_diluted_valuation": 8500000,
            "total_volume": 27000,
            "high_24h": 265.00,
            "low_24h": 245.00,
            "price_change_24h": 5.50,
            "price_change_percentage_24h": 2.16,
            "market_cap_change_24h": 120000,
            "market_cap_change_percentage_24h": 1.80,
            "circulating_supply": 35000,
            "total_supply": 40000,
            "max_supply": 40000,
            "ath": 280.00,
            "ath_change_percentage": -7.14,
            "ath_date": "2023-12-25T10:00:00.000Z",
            "atl": 199.99,
            "atl_change_percentage": 30.00,
            "atl_date": "2023-07-20T08:00:00.000Z",
            "roi": null,
            "last_updated": "2024-10-16T06:25:41.652Z"
        },
        {
            "id": "airfryer789",
            "symbol": "af789",
            "name": "Air Fryer Turbo",
            "image": "https://coin-images.coingecko.com/coins/images/1/large/bitcoin.png?1696501400",
            "current_price": 89.99,
            "market_cap": 2135000,
            "market_cap_rank": 3,
            "fully_diluted_valuation": 2500000,
            "total_volume": 12000,
            "high_24h": 95.00,
            "low_24h": 85.00,
            "price_change_24h": -1.50,
            "price_change_percentage_24h": -1.64,
            "market_cap_change_24h": -32000,
            "market_cap_change_percentage_24h": -1.48,
            "circulating_supply": 24000,
            "total_supply": 26000,
            "max_supply": 26000,
            "ath": 100.00,
            "ath_change_percentage": -10.00,
            "ath_date": "2024-01-10T11:45:00.000Z",
            "atl": 49.99,
            "atl_change_percentage": 80.00,
            "atl_date": "2023-09-05T07:15:00.000Z",
            "roi": null,
            "last_updated": "2024-10-16T06:25:41.652Z"
        }
        ]
    '''
    # 将JSON字符串加载为Python对象
    price_data = json.loads(json_data)
    return render_template('trends.html', plot=plot,price_data=price_data)


# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取输入数据
    data = request.get_json(force=True)
    input_data = np.array(data['input']).astype(np.float32)

    # 确保输入数据形状为 (1, 7, 18)
    if input_data.shape != (1, 7, 18):
        return jsonify({'error': 'Invalid input shape, expected (1, 7, 18)'}), 400

    # 进行预测
    prediction = model.predict(input_data)

    # 返回预测结果
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9000)
