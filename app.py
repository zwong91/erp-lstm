from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd

# 初始化Flask应用
app = Flask(__name__)

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l1_l2

# Custom Attention Layer
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
model = load_model(f'models/best_erp_lstm_model_20241015_1613.keras', custom_objects={'Attention': Attention})

# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取输入数据
    data = request.get_json(force=True)
    input_data = np.array(data['input']).astype(np.float32)

    # 确保输入数据形状为 (1, 7, 6)
    if input_data.shape != (1, 7, 6):
        return jsonify({'error': 'Invalid input shape, expected (1, 7, 6)'}), 400

    # 进行预测
    prediction = model.predict(input_data)

    # 返回预测结果
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)