基于时间序列算法，库存预测表预测生产库、销售库存量，预防性进行库存管控分析，提高发运效率

# 工程里面datasets csv 文件是一个公开的demo库存数据集, 替换成真实场景数据
erp_pdd_2009_2016.csv 中有A-N 14种SKU商品信息
- A
- B
- C
- D
- E
- F
- G
- H
- I
- J
- K
- L
- M
- N
  
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

你可以使用以下示例JSON数据来测试API： 假设输入数据是一个JSON对象，其中包含一个三维数组(过去7个时间步的数据, 预测下一个时间点的数据)

```python
import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('path_to_your_csv_file.csv', parse_dates=['Date Time'], index_col='Date Time')

# 构造 Seconds 列
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)

# 构造 Day sin, Day cos, Year sin, Year cos 列
day = 24 * 60 * 60
year = 365.2425 * day

df['Day sin'] = np.sin(df['Seconds'] * (2 * np.pi / day))
df['Day cos'] = np.cos(df['Seconds'] * (2 * np.pi / day))
df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))

# 选择需要的列
columns_to_add = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
selected_columns = df[columns_to_add]

# 构造输入数据，假设我们取前7行数据
input_data = selected_columns.head(7).values.reshape(1, 7, 18).astype(np.float32)

# 打印输入数据
print(input_data)

```

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
{
    "input": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
               [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
               [3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4],
               [5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2],
               [7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0],
               [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
               [10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6]]]
}

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"input": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6], [3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4], [5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2], [7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0], [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8], [10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6]]]}'

预测结果, 分别表示种类A, 种类B, .... N 14, Day sin
{"prediction":[[0.22278626263141632,-3.6332273483276367,5.187934398651123,121.45440673828125,14.705184936523438,11.681909561157227,3.4519293308258057,10.965514183044434,13.259306907653809,1944.6356201171875,4.166243553161621,6.757382392883301,261.5758361816406,14.526421546936035]]}
```
