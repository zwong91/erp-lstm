基于时间序列算法，库存预测表预测生产库、销售库存量，预防性进行库存管控分析，提高发运效率

# 工程里面datasets csv 文件是一个公开的demo天气数据集, 替换成真实场景数据

## api

将使用Flask创建一个API来预测Keras LSTM模型。首先，我们需要确保

requirements.txt

中的依赖项已经安装。然后，我们将编写Flask应用来加载模型并处理预测请求。

### 步骤

1. **安装依赖项**：
    - 确保所有依赖项都已安装。

2. **创建Flask应用**：
    - 初始化Flask应用。
    - 定义一个路由来处理预测请求。

3. **加载Keras LSTM模型**：
    - 使用Keras加载预训练的LSTM模型。

4. **处理输入数据**：
    - 从请求中获取输入数据。
    - 对输入数据进行预处理，使其适合模型预测。

5. **进行预测**：
    - 使用模型进行预测。
    - 返回预测结果。

### 安装依赖项

在终端中运行以下命令来安装依赖项：

```bash
pip install -r requirements.txt
```

### Flask应用代码 `app.py`

```python
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd

# 初始化Flask应用
app = Flask(__name__)

# 加载预训练的Keras LSTM模型
model = load_model('path_to_your_lstm_model.h5')

# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取输入数据
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data)
    
    # 假设模型需要的输入形状为 (1, 时间步长, 特征数)
    input_data = input_data.values.reshape(1, input_data.shape[0], input_data.shape[1])

    # 进行预测
    prediction = model.predict(input_data)

    # 返回预测结果
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

### 运行Flask应用

在终端中运行以下命令来启动Flask应用：

```bash
export FLASK_APP=app.py
flask run
```

### 示例请求

你可以使用以下示例JSON数据来测试API：

```json
{
    "input": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
               [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
               [1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
               [2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
               [3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
               [3.7, 3.8, 3.9, 4.0, 4.1, 4.2]]]
}
```

你可以使用`curl`命令或Postman等工具来发送POST请求：

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"input": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6, 1.7, 1.8], [1.9, 2.0, 2.1, 2.2, 2.3, 2.4], [2.5, 2.6, 2.7, 2.8, 2.9, 3.0], [3.1, 3.2, 3.3, 3.4, 3.5, 3.6], [3.7, 3.8, 3.9, 4.0, 4.1, 4.2]]]}'
```

结果有两个值, 分别表示天气压力值, 温度
```
{"prediction":[[1011.3223876953125,20.84238624572754]]}
```
