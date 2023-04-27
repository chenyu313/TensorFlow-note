## Tensorflow进行数据分析
简单介绍了Tensorflow深度学习框架的运算流程之后，引入一个具体案例，并使用Tensorflow对数据进行 分析。

在数据分类的研究中，普遍存在类别分布不平衡的问题，即某一类别的样本数量远远多于另一类，具有这样特征的数据集视为不平衡。  
我们将使用Kaggle 上托管的 Credit Card Fraud Detection 数据集，目的是从总共 284,807 笔交易中检测出仅有的 492 笔欺诈交易。  
我们需要做的就是定义模型和类权重，从而帮助模型从不平衡数据中学习。

具体流程：
* 使用 Pandas 加载 CSV 文件。
* 创建训练、验证和测试集。
* 训练模型（包括设置类权重）。
* 使用各种指标（包括精确率和召回率）评估模型。
* 尝试使用常见技术来处理不平衡数据，例如：类加权


```python
import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 控制图像属性
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
```

### 数据处理与浏览


```python
# 下载 Kaggle Credit Card Fraud 数据集
file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>...</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>...</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>...</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>-0.894286</td>
      <td>0.286157</td>
      <td>-0.113192</td>
      <td>-0.271526</td>
      <td>2.669599</td>
      <td>3.721818</td>
      <td>0.370145</td>
      <td>0.851084</td>
      <td>-0.392048</td>
      <td>...</td>
      <td>-0.073425</td>
      <td>-0.268092</td>
      <td>-0.204233</td>
      <td>1.011592</td>
      <td>0.373205</td>
      <td>-0.384157</td>
      <td>0.011747</td>
      <td>0.142404</td>
      <td>93.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>-0.338262</td>
      <td>1.119593</td>
      <td>1.044367</td>
      <td>-0.222187</td>
      <td>0.499361</td>
      <td>-0.246761</td>
      <td>0.651583</td>
      <td>0.069539</td>
      <td>-0.736727</td>
      <td>...</td>
      <td>-0.246914</td>
      <td>-0.633753</td>
      <td>-0.120794</td>
      <td>-0.385050</td>
      <td>-0.069733</td>
      <td>0.094199</td>
      <td>0.246219</td>
      <td>0.083076</td>
      <td>3.68</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>




```python
# 对数据属性进行描述
raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>1.165980e-15</td>
      <td>3.416908e-16</td>
      <td>-1.373150e-15</td>
      <td>2.086869e-15</td>
      <td>9.604066e-16</td>
      <td>1.687098e-15</td>
      <td>-3.666453e-16</td>
      <td>-1.220404e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 检查数据集的不平衡情况
neg, pos = np.bincount(raw_df['Class']) # 同Class属性对数据进行分析
total = neg + pos #这里我们将欺诈数量作为正样本pos
print('Examples:\n    总计: {}\n    欺诈交易数量: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
```

    Examples:
        总计: 284807
        欺诈交易数量: 492 (0.17% of total)
    
    

### 清理、拆分和归一化数据
原始数据有一些问题。首先，Time 和 Amount 列变化太大，无法直接使用。删除 Time 列（因为不清楚其含义），并获取 Amount 列的日志以缩小其范围。


```python
cleaned_df = raw_df.copy()

# 删除Time列
cleaned_df.pop('Time')

# 获取 Amount 列的日志以缩小其范围
eps=0.001
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)
```

将数据集拆分为训练、验证和测试集。验证集在模型拟合期间使用，用于评估损失和任何指标，判断模型与数据的拟合程度。测试集在训练阶段完全不使用，仅在最后用于评估模型泛化到新数据的能力。这对于不平衡的数据集尤为重要，因为过拟合是缺乏训练数据造成的一个重大问题。


```python
# 拆分为训练、验证和测试集
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# 形成标签和特征的np数组
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
print(train_features)
```

    [[-1.42048887e+00  1.09880362e+00  1.63835847e+00 ... -1.72045796e-01
      -1.39209006e-01  2.54952329e+00]
     [ 1.91977743e+00 -5.15399093e-01 -9.20753660e-01 ... -6.21946913e-02
      -4.38767141e-02  4.19057856e+00]
     [-1.24128622e+00  1.10901088e+00  1.04311024e+00 ...  4.68305582e-01
       1.96301942e-01  3.97031078e+00]
     ...
     [ 1.20486003e+00 -6.64091980e-03 -3.74094819e-01 ... -6.14565309e-03
       1.49813518e-03  4.08934877e+00]
     [ 1.49671679e+00 -6.95956491e-01  1.85339914e-01 ...  1.64507476e-02
       4.50199392e-03  1.79192612e+00]
     [ 1.11645056e+00 -8.78775630e-01 -1.44111798e+00 ...  1.35149531e-02
       4.48778434e-02  5.05650692e+00]]
    


```python
# 使用 sklearn StandardScaler 将输入特征归一化（平均值设置为 0，标准偏差设置为 1）
scaler = StandardScaler()
# 使用 train_features 进行拟合，以确保模型不会窥视验证集或测试集
train_features = scaler.fit_transform(train_features) 

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# np.clip为截取函数，截取大于-5小于5的数
train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)
```

    Training labels shape: (182276,)
    Validation labels shape: (45569,)
    Test labels shape: (56962,)
    Training features shape: (182276, 29)
    Validation features shape: (45569, 29)
    Test features shape: (56962, 29)
    

### 查看数据分布



```python
pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

sns.jointplot(x=pos_df['V5'], y=pos_df['V6'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
plt.suptitle("Positive distribution")

sns.jointplot(x=neg_df['V5'], y=neg_df['V6'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
_ = plt.suptitle("Negative distribution")
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_12_0.png)
    



    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_12_1.png)
    


### 定义模型和指标
定义一个函数，该函数会创建一个简单的神经网络，其中包含一个密集连接的隐藏层、一个用于减少过拟合的随机失活层，以及一个返回欺诈交易概率的输出 Sigmoid 层：


```python
# 指标
METRICS = [
      keras.metrics.TruePositives(name='tp'), # 真正例
      keras.metrics.FalsePositives(name='fp'),  # 假正例
      keras.metrics.TrueNegatives(name='tn'), # 真负例
      keras.metrics.FalseNegatives(name='fn'), # 假负例
      keras.metrics.BinaryAccuracy(name='accuracy'), # 准确率是被正确分类的样本的百分比
      keras.metrics.Precision(name='precision'), # 精确率是被正确分类的预测正例的百分比
      keras.metrics.Recall(name='recall'),  # 召回率是被正确分类的实际正例的百分比
      keras.metrics.AUC(name='auc'), # AUC 是指接收器操作特征曲线中的曲线下方面积 (ROC-AUC)。此指标等于分类器对随机正样本的排序高于随机负样本的概率。
      keras.metrics.AUC(name='prc', curve='PR'), # AUPRC 是指精确率-召回率曲线下方面积。该指标计算不同概率阈值的精度率-召回率对。
]

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)), # 隐藏层
      keras.layers.Dropout(0.5), # 随机失活层
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias), # sigmoid层
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3), # 优化器
      loss=keras.losses.BinaryCrossentropy(), # 损失
      metrics=metrics) # 指标

  return model
```

### 基线模型
注：此模型无法很好地处理类不平衡问题。我们将在本教程的后面部分对此进行改进。


```python
EPOCHS = 100
BATCH_SIZE = 2048

# 早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model()
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 16)                480       
    _________________________________________________________________
    dropout (Dropout)            (None, 16)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 497
    Trainable params: 497
    Non-trainable params: 0
    _________________________________________________________________
    

### 训练模型



```python
model = make_model()
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels))
```

    Epoch 1/100
    90/90 [==============================] - 2s 13ms/step - loss: 1.1795 - tp: 323.0000 - fp: 120843.0000 - tn: 106608.0000 - fn: 71.0000 - accuracy: 0.4693 - precision: 0.0027 - recall: 0.8198 - auc: 0.7785 - prc: 0.0194 - val_loss: 0.6323 - val_tp: 71.0000 - val_fp: 14543.0000 - val_tn: 30948.0000 - val_fn: 7.0000 - val_accuracy: 0.6807 - val_precision: 0.0049 - val_recall: 0.9103 - val_auc: 0.9164 - val_prc: 0.2200
    Epoch 2/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.5103 - tp: 204.0000 - fp: 46936.0000 - tn: 135024.0000 - fn: 112.0000 - accuracy: 0.7419 - precision: 0.0043 - recall: 0.6456 - auc: 0.7448 - prc: 0.0607 - val_loss: 0.2624 - val_tp: 50.0000 - val_fp: 938.0000 - val_tn: 44553.0000 - val_fn: 28.0000 - val_accuracy: 0.9788 - val_precision: 0.0506 - val_recall: 0.6410 - val_auc: 0.8897 - val_prc: 0.3038
    Epoch 3/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.2800 - tp: 160.0000 - fp: 16609.0000 - tn: 165351.0000 - fn: 156.0000 - accuracy: 0.9080 - precision: 0.0095 - recall: 0.5063 - auc: 0.7507 - prc: 0.1316 - val_loss: 0.1306 - val_tp: 42.0000 - val_fp: 356.0000 - val_tn: 45135.0000 - val_fn: 36.0000 - val_accuracy: 0.9914 - val_precision: 0.1055 - val_recall: 0.5385 - val_auc: 0.8890 - val_prc: 0.3768
    Epoch 4/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1853 - tp: 150.0000 - fp: 7119.0000 - tn: 174841.0000 - fn: 166.0000 - accuracy: 0.9600 - precision: 0.0206 - recall: 0.4747 - auc: 0.7876 - prc: 0.1604 - val_loss: 0.0745 - val_tp: 41.0000 - val_fp: 279.0000 - val_tn: 45212.0000 - val_fn: 37.0000 - val_accuracy: 0.9931 - val_precision: 0.1281 - val_recall: 0.5256 - val_auc: 0.8915 - val_prc: 0.4003
    Epoch 5/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1352 - tp: 155.0000 - fp: 3552.0000 - tn: 178408.0000 - fn: 161.0000 - accuracy: 0.9796 - precision: 0.0418 - recall: 0.4905 - auc: 0.8173 - prc: 0.2248 - val_loss: 0.0472 - val_tp: 43.0000 - val_fp: 219.0000 - val_tn: 45272.0000 - val_fn: 35.0000 - val_accuracy: 0.9944 - val_precision: 0.1641 - val_recall: 0.5513 - val_auc: 0.8994 - val_prc: 0.4756
    Epoch 6/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1067 - tp: 165.0000 - fp: 2041.0000 - tn: 179919.0000 - fn: 151.0000 - accuracy: 0.9880 - precision: 0.0748 - recall: 0.5222 - auc: 0.8576 - prc: 0.2986 - val_loss: 0.0322 - val_tp: 48.0000 - val_fp: 91.0000 - val_tn: 45400.0000 - val_fn: 30.0000 - val_accuracy: 0.9973 - val_precision: 0.3453 - val_recall: 0.6154 - val_auc: 0.9072 - val_prc: 0.5331
    Epoch 7/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.0870 - tp: 173.0000 - fp: 1218.0000 - tn: 180742.0000 - fn: 143.0000 - accuracy: 0.9925 - precision: 0.1244 - recall: 0.5475 - auc: 0.8687 - prc: 0.3374 - val_loss: 0.0230 - val_tp: 50.0000 - val_fp: 21.0000 - val_tn: 45470.0000 - val_fn: 28.0000 - val_accuracy: 0.9989 - val_precision: 0.7042 - val_recall: 0.6410 - val_auc: 0.9041 - val_prc: 0.5803
    Epoch 8/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.0734 - tp: 172.0000 - fp: 682.0000 - tn: 181278.0000 - fn: 144.0000 - accuracy: 0.9955 - precision: 0.2014 - recall: 0.5443 - auc: 0.8689 - prc: 0.3459 - val_loss: 0.0171 - val_tp: 51.0000 - val_fp: 12.0000 - val_tn: 45479.0000 - val_fn: 27.0000 - val_accuracy: 0.9991 - val_precision: 0.8095 - val_recall: 0.6538 - val_auc: 0.9082 - val_prc: 0.6144
    Epoch 9/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.0633 - tp: 174.0000 - fp: 457.0000 - tn: 181503.0000 - fn: 142.0000 - accuracy: 0.9967 - precision: 0.2758 - recall: 0.5506 - auc: 0.8812 - prc: 0.3915 - val_loss: 0.0132 - val_tp: 52.0000 - val_fp: 10.0000 - val_tn: 45481.0000 - val_fn: 26.0000 - val_accuracy: 0.9992 - val_precision: 0.8387 - val_recall: 0.6667 - val_auc: 0.9084 - val_prc: 0.6445
    Epoch 10/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.0545 - tp: 199.0000 - fp: 346.0000 - tn: 181614.0000 - fn: 117.0000 - accuracy: 0.9975 - precision: 0.3651 - recall: 0.6297 - auc: 0.8883 - prc: 0.4707 - val_loss: 0.0105 - val_tp: 53.0000 - val_fp: 10.0000 - val_tn: 45481.0000 - val_fn: 25.0000 - val_accuracy: 0.9992 - val_precision: 0.8413 - val_recall: 0.6795 - val_auc: 0.9098 - val_prc: 0.6638
    Epoch 11/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.0478 - tp: 206.0000 - fp: 279.0000 - tn: 181681.0000 - fn: 110.0000 - accuracy: 0.9979 - precision: 0.4247 - recall: 0.6519 - auc: 0.9063 - prc: 0.5210 - val_loss: 0.0087 - val_tp: 55.0000 - val_fp: 10.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8462 - val_recall: 0.7051 - val_auc: 0.9082 - val_prc: 0.6754
    Restoring model weights from the end of the best epoch.
    Epoch 00011: early stopping
    

### 查看训练历史记录
针对训练集和验证集生成模型的准确率和损失绘图。检查是否过拟合。


```python
def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();

# 可视化
plot_metrics(baseline_history)
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_20_0.png)
    



```python
# 评估指标
# 使用混淆矩阵来汇总实际标签与预测标签，其中 X 轴是预测标签，Y 轴是实际标签
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))


baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)
```

    loss :  0.6309481859207153
    tp :  85.0
    fp :  17927.0
    tn :  38937.0
    fn :  13.0
    accuracy :  0.6850531697273254
    precision :  0.004719076212495565
    recall :  0.8673469424247742
    auc :  0.8840129971504211
    prc :  0.15706372261047363
    
    Legitimate Transactions Detected (True Negatives):  38937
    Legitimate Transactions Incorrectly Detected (False Positives):  17927
    Fraudulent Transactions Missed (False Negatives):  13
    Fraudulent Transactions Detected (True Positives):  85
    Total Fraudulent Transactions:  98
    


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_21_1.png)
    


如果模型完美地预测了所有内容，则这是一个对角矩阵，其中偏离主对角线的值（表示不正确的预测）将为零。在这种情况下，矩阵会显示假正例相对较少，这意味着被错误标记的合法交易相对较少。但是，我们可能希望得到更少的假负例，即使这会增加假正例的数量。这种权衡可能更加可取，因为假负例允许进行欺诈交易，而假正例可能导致向客户发送电子邮件，要求他们验证自己的信用卡活动。

### 绘制ROC
ROC绘图非常有用，因为它一目了然地显示了模型只需通过调整输出阈值就能达到的性能范围。


```python
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right');
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_24_0.png)
    


### 绘制 AUPRC
内插精确率-召回率曲线的下方面积，通过为分类阈值的不同值绘制（召回率、精确率）点获得。根据计算方式，PR AUC 可能相当于模型的平均精确率。


```python
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend();
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_26_0.png)
    


假负例（漏掉欺诈交易）可能造成财务损失，而假正例（将交易错误地标记为欺诈）则可能降低用户满意度。

## 类权重
我们的目标是识别欺诈交易，但没有很多可以使用的此类正样本，因此希望分类器提高可用的少数样本的权重。为此，可以使用参数将 Keras 权重传递给每个类。这些权重将使模型“更加关注”来自代表不足的类的样本。


```python
# 按total/2进行缩放有助于将损失保持在相似的量级
# 所有例子的权重之和保持不变
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
```

    Weight for class 0: 0.50
    Weight for class 1: 289.44
    

### 使用类权重训练模型



```python
weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    # 使用 class_weights 会改变损失范围。这可能会影响训练的稳定性，具体取决于优化器
    class_weight=class_weight)
```

    WARNING:tensorflow:From d:\anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\ops\array_ops.py:5043: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
    Instructions for updating:
    The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
    Epoch 1/100
    90/90 [==============================] - 3s 13ms/step - loss: 0.7281 - tp: 380.0000 - fp: 147173.0000 - tn: 91651.0000 - fn: 34.0000 - accuracy: 0.3847 - precision: 0.0026 - recall: 0.9179 - auc: 0.8390 - prc: 0.0371 - val_loss: 0.8372 - val_tp: 75.0000 - val_fp: 24519.0000 - val_tn: 20972.0000 - val_fn: 3.0000 - val_accuracy: 0.4619 - val_precision: 0.0030 - val_recall: 0.9615 - val_auc: 0.9530 - val_prc: 0.4755
    Epoch 2/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.4368 - tp: 293.0000 - fp: 72103.0000 - tn: 109857.0000 - fn: 23.0000 - accuracy: 0.6043 - precision: 0.0040 - recall: 0.9272 - auc: 0.9238 - prc: 0.2344 - val_loss: 0.4927 - val_tp: 71.0000 - val_fp: 7618.0000 - val_tn: 37873.0000 - val_fn: 7.0000 - val_accuracy: 0.8327 - val_precision: 0.0092 - val_recall: 0.9103 - val_auc: 0.9585 - val_prc: 0.6459
    Epoch 3/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.3277 - tp: 293.0000 - fp: 39552.0000 - tn: 142408.0000 - fn: 23.0000 - accuracy: 0.7829 - precision: 0.0074 - recall: 0.9272 - auc: 0.9492 - prc: 0.3652 - val_loss: 0.3374 - val_tp: 70.0000 - val_fp: 3276.0000 - val_tn: 42215.0000 - val_fn: 8.0000 - val_accuracy: 0.9279 - val_precision: 0.0209 - val_recall: 0.8974 - val_auc: 0.9596 - val_prc: 0.6907
    Epoch 4/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.2759 - tp: 287.0000 - fp: 24422.0000 - tn: 157538.0000 - fn: 29.0000 - accuracy: 0.8659 - precision: 0.0116 - recall: 0.9082 - auc: 0.9570 - prc: 0.4336 - val_loss: 0.2538 - val_tp: 70.0000 - val_fp: 1986.0000 - val_tn: 43505.0000 - val_fn: 8.0000 - val_accuracy: 0.9562 - val_precision: 0.0340 - val_recall: 0.8974 - val_auc: 0.9606 - val_prc: 0.7122
    Epoch 5/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.2639 - tp: 285.0000 - fp: 17680.0000 - tn: 164280.0000 - fn: 31.0000 - accuracy: 0.9028 - precision: 0.0159 - recall: 0.9019 - auc: 0.9545 - prc: 0.5172 - val_loss: 0.2187 - val_tp: 70.0000 - val_fp: 1667.0000 - val_tn: 43824.0000 - val_fn: 8.0000 - val_accuracy: 0.9632 - val_precision: 0.0403 - val_recall: 0.8974 - val_auc: 0.9611 - val_prc: 0.7175
    Epoch 6/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.2461 - tp: 285.0000 - fp: 14700.0000 - tn: 167260.0000 - fn: 31.0000 - accuracy: 0.9192 - precision: 0.0190 - recall: 0.9019 - auc: 0.9590 - prc: 0.5112 - val_loss: 0.2032 - val_tp: 70.0000 - val_fp: 1609.0000 - val_tn: 43882.0000 - val_fn: 8.0000 - val_accuracy: 0.9645 - val_precision: 0.0417 - val_recall: 0.8974 - val_auc: 0.9630 - val_prc: 0.7082
    Epoch 7/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.2434 - tp: 279.0000 - fp: 12824.0000 - tn: 169136.0000 - fn: 37.0000 - accuracy: 0.9294 - precision: 0.0213 - recall: 0.8829 - auc: 0.9549 - prc: 0.5290 - val_loss: 0.1912 - val_tp: 70.0000 - val_fp: 1524.0000 - val_tn: 43967.0000 - val_fn: 8.0000 - val_accuracy: 0.9664 - val_precision: 0.0439 - val_recall: 0.8974 - val_auc: 0.9645 - val_prc: 0.7118
    Epoch 8/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.2313 - tp: 282.0000 - fp: 10863.0000 - tn: 171097.0000 - fn: 34.0000 - accuracy: 0.9402 - precision: 0.0253 - recall: 0.8924 - auc: 0.9616 - prc: 0.5115 - val_loss: 0.1743 - val_tp: 69.0000 - val_fp: 1377.0000 - val_tn: 44114.0000 - val_fn: 9.0000 - val_accuracy: 0.9696 - val_precision: 0.0477 - val_recall: 0.8846 - val_auc: 0.9654 - val_prc: 0.7142
    Epoch 9/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1931 - tp: 289.0000 - fp: 9668.0000 - tn: 172292.0000 - fn: 27.0000 - accuracy: 0.9468 - precision: 0.0290 - recall: 0.9146 - auc: 0.9763 - prc: 0.5527 - val_loss: 0.1542 - val_tp: 69.0000 - val_fp: 1232.0000 - val_tn: 44259.0000 - val_fn: 9.0000 - val_accuracy: 0.9728 - val_precision: 0.0530 - val_recall: 0.8846 - val_auc: 0.9654 - val_prc: 0.7158
    Epoch 10/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.2142 - tp: 288.0000 - fp: 8995.0000 - tn: 172965.0000 - fn: 28.0000 - accuracy: 0.9505 - precision: 0.0310 - recall: 0.9114 - auc: 0.9642 - prc: 0.5627 - val_loss: 0.1516 - val_tp: 69.0000 - val_fp: 1232.0000 - val_tn: 44259.0000 - val_fn: 9.0000 - val_accuracy: 0.9728 - val_precision: 0.0530 - val_recall: 0.8846 - val_auc: 0.9655 - val_prc: 0.7061
    Epoch 11/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1840 - tp: 286.0000 - fp: 7980.0000 - tn: 173980.0000 - fn: 30.0000 - accuracy: 0.9561 - precision: 0.0346 - recall: 0.9051 - auc: 0.9774 - prc: 0.5686 - val_loss: 0.1360 - val_tp: 69.0000 - val_fp: 1103.0000 - val_tn: 44388.0000 - val_fn: 9.0000 - val_accuracy: 0.9756 - val_precision: 0.0589 - val_recall: 0.8846 - val_auc: 0.9657 - val_prc: 0.7079
    Epoch 12/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1818 - tp: 289.0000 - fp: 7774.0000 - tn: 174186.0000 - fn: 27.0000 - accuracy: 0.9572 - precision: 0.0358 - recall: 0.9146 - auc: 0.9764 - prc: 0.5722 - val_loss: 0.1298 - val_tp: 69.0000 - val_fp: 1090.0000 - val_tn: 44401.0000 - val_fn: 9.0000 - val_accuracy: 0.9759 - val_precision: 0.0595 - val_recall: 0.8846 - val_auc: 0.9651 - val_prc: 0.7092
    Epoch 13/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1638 - tp: 294.0000 - fp: 6939.0000 - tn: 175021.0000 - fn: 22.0000 - accuracy: 0.9618 - precision: 0.0406 - recall: 0.9304 - auc: 0.9825 - prc: 0.6024 - val_loss: 0.1177 - val_tp: 69.0000 - val_fp: 965.0000 - val_tn: 44526.0000 - val_fn: 9.0000 - val_accuracy: 0.9786 - val_precision: 0.0667 - val_recall: 0.8846 - val_auc: 0.9654 - val_prc: 0.7093
    Epoch 14/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1857 - tp: 288.0000 - fp: 6661.0000 - tn: 175299.0000 - fn: 28.0000 - accuracy: 0.9633 - precision: 0.0414 - recall: 0.9114 - auc: 0.9721 - prc: 0.6086 - val_loss: 0.1170 - val_tp: 69.0000 - val_fp: 962.0000 - val_tn: 44529.0000 - val_fn: 9.0000 - val_accuracy: 0.9787 - val_precision: 0.0669 - val_recall: 0.8846 - val_auc: 0.9655 - val_prc: 0.7093
    Epoch 15/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1766 - tp: 287.0000 - fp: 6279.0000 - tn: 175681.0000 - fn: 29.0000 - accuracy: 0.9654 - precision: 0.0437 - recall: 0.9082 - auc: 0.9767 - prc: 0.6228 - val_loss: 0.1145 - val_tp: 69.0000 - val_fp: 968.0000 - val_tn: 44523.0000 - val_fn: 9.0000 - val_accuracy: 0.9786 - val_precision: 0.0665 - val_recall: 0.8846 - val_auc: 0.9655 - val_prc: 0.7097
    Epoch 16/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1811 - tp: 287.0000 - fp: 6581.0000 - tn: 175379.0000 - fn: 29.0000 - accuracy: 0.9637 - precision: 0.0418 - recall: 0.9082 - auc: 0.9755 - prc: 0.6066 - val_loss: 0.1161 - val_tp: 69.0000 - val_fp: 1033.0000 - val_tn: 44458.0000 - val_fn: 9.0000 - val_accuracy: 0.9771 - val_precision: 0.0626 - val_recall: 0.8846 - val_auc: 0.9662 - val_prc: 0.7118
    Epoch 17/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1791 - tp: 289.0000 - fp: 6589.0000 - tn: 175371.0000 - fn: 27.0000 - accuracy: 0.9637 - precision: 0.0420 - recall: 0.9146 - auc: 0.9765 - prc: 0.5573 - val_loss: 0.1142 - val_tp: 69.0000 - val_fp: 1035.0000 - val_tn: 44456.0000 - val_fn: 9.0000 - val_accuracy: 0.9771 - val_precision: 0.0625 - val_recall: 0.8846 - val_auc: 0.9659 - val_prc: 0.6845
    Epoch 18/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1756 - tp: 287.0000 - fp: 6185.0000 - tn: 175775.0000 - fn: 29.0000 - accuracy: 0.9659 - precision: 0.0443 - recall: 0.9082 - auc: 0.9759 - prc: 0.5901 - val_loss: 0.1147 - val_tp: 69.0000 - val_fp: 1040.0000 - val_tn: 44451.0000 - val_fn: 9.0000 - val_accuracy: 0.9770 - val_precision: 0.0622 - val_recall: 0.8846 - val_auc: 0.9664 - val_prc: 0.6942
    Epoch 19/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1700 - tp: 287.0000 - fp: 6135.0000 - tn: 175825.0000 - fn: 29.0000 - accuracy: 0.9662 - precision: 0.0447 - recall: 0.9082 - auc: 0.9781 - prc: 0.6197 - val_loss: 0.1102 - val_tp: 69.0000 - val_fp: 1006.0000 - val_tn: 44485.0000 - val_fn: 9.0000 - val_accuracy: 0.9777 - val_precision: 0.0642 - val_recall: 0.8846 - val_auc: 0.9661 - val_prc: 0.6944
    Epoch 20/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1692 - tp: 290.0000 - fp: 5934.0000 - tn: 176026.0000 - fn: 26.0000 - accuracy: 0.9673 - precision: 0.0466 - recall: 0.9177 - auc: 0.9789 - prc: 0.5983 - val_loss: 0.1128 - val_tp: 69.0000 - val_fp: 1043.0000 - val_tn: 44448.0000 - val_fn: 9.0000 - val_accuracy: 0.9769 - val_precision: 0.0621 - val_recall: 0.8846 - val_auc: 0.9659 - val_prc: 0.6697
    Epoch 21/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1770 - tp: 289.0000 - fp: 6066.0000 - tn: 175894.0000 - fn: 27.0000 - accuracy: 0.9666 - precision: 0.0455 - recall: 0.9146 - auc: 0.9762 - prc: 0.6049 - val_loss: 0.1109 - val_tp: 69.0000 - val_fp: 995.0000 - val_tn: 44496.0000 - val_fn: 9.0000 - val_accuracy: 0.9780 - val_precision: 0.0648 - val_recall: 0.8846 - val_auc: 0.9664 - val_prc: 0.6709
    Epoch 22/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1645 - tp: 290.0000 - fp: 5878.0000 - tn: 176082.0000 - fn: 26.0000 - accuracy: 0.9676 - precision: 0.0470 - recall: 0.9177 - auc: 0.9799 - prc: 0.6341 - val_loss: 0.1066 - val_tp: 69.0000 - val_fp: 967.0000 - val_tn: 44524.0000 - val_fn: 9.0000 - val_accuracy: 0.9786 - val_precision: 0.0666 - val_recall: 0.8846 - val_auc: 0.9664 - val_prc: 0.6818
    Epoch 23/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1434 - tp: 288.0000 - fp: 5330.0000 - tn: 176630.0000 - fn: 28.0000 - accuracy: 0.9706 - precision: 0.0513 - recall: 0.9114 - auc: 0.9871 - prc: 0.6103 - val_loss: 0.1015 - val_tp: 69.0000 - val_fp: 930.0000 - val_tn: 44561.0000 - val_fn: 9.0000 - val_accuracy: 0.9794 - val_precision: 0.0691 - val_recall: 0.8846 - val_auc: 0.9661 - val_prc: 0.6998
    Epoch 24/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1551 - tp: 288.0000 - fp: 5627.0000 - tn: 176333.0000 - fn: 28.0000 - accuracy: 0.9690 - precision: 0.0487 - recall: 0.9114 - auc: 0.9833 - prc: 0.6345 - val_loss: 0.1005 - val_tp: 69.0000 - val_fp: 932.0000 - val_tn: 44559.0000 - val_fn: 9.0000 - val_accuracy: 0.9793 - val_precision: 0.0689 - val_recall: 0.8846 - val_auc: 0.9668 - val_prc: 0.6998
    Epoch 25/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1587 - tp: 291.0000 - fp: 5312.0000 - tn: 176648.0000 - fn: 25.0000 - accuracy: 0.9707 - precision: 0.0519 - recall: 0.9209 - auc: 0.9811 - prc: 0.6435 - val_loss: 0.0972 - val_tp: 69.0000 - val_fp: 866.0000 - val_tn: 44625.0000 - val_fn: 9.0000 - val_accuracy: 0.9808 - val_precision: 0.0738 - val_recall: 0.8846 - val_auc: 0.9668 - val_prc: 0.7005
    Epoch 26/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1528 - tp: 292.0000 - fp: 4997.0000 - tn: 176963.0000 - fn: 24.0000 - accuracy: 0.9725 - precision: 0.0552 - recall: 0.9241 - auc: 0.9830 - prc: 0.6354 - val_loss: 0.0990 - val_tp: 69.0000 - val_fp: 926.0000 - val_tn: 44565.0000 - val_fn: 9.0000 - val_accuracy: 0.9795 - val_precision: 0.0693 - val_recall: 0.8846 - val_auc: 0.9672 - val_prc: 0.7004
    Epoch 27/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1483 - tp: 294.0000 - fp: 5500.0000 - tn: 176460.0000 - fn: 22.0000 - accuracy: 0.9697 - precision: 0.0507 - recall: 0.9304 - auc: 0.9839 - prc: 0.6409 - val_loss: 0.0967 - val_tp: 69.0000 - val_fp: 932.0000 - val_tn: 44559.0000 - val_fn: 9.0000 - val_accuracy: 0.9793 - val_precision: 0.0689 - val_recall: 0.8846 - val_auc: 0.9652 - val_prc: 0.7007
    Epoch 28/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1406 - tp: 291.0000 - fp: 5054.0000 - tn: 176906.0000 - fn: 25.0000 - accuracy: 0.9721 - precision: 0.0544 - recall: 0.9209 - auc: 0.9869 - prc: 0.6415 - val_loss: 0.0914 - val_tp: 69.0000 - val_fp: 865.0000 - val_tn: 44626.0000 - val_fn: 9.0000 - val_accuracy: 0.9808 - val_precision: 0.0739 - val_recall: 0.8846 - val_auc: 0.9653 - val_prc: 0.7016
    Epoch 29/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1546 - tp: 290.0000 - fp: 5102.0000 - tn: 176858.0000 - fn: 26.0000 - accuracy: 0.9719 - precision: 0.0538 - recall: 0.9177 - auc: 0.9826 - prc: 0.6143 - val_loss: 0.0954 - val_tp: 69.0000 - val_fp: 949.0000 - val_tn: 44542.0000 - val_fn: 9.0000 - val_accuracy: 0.9790 - val_precision: 0.0678 - val_recall: 0.8846 - val_auc: 0.9658 - val_prc: 0.7041
    Epoch 30/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1539 - tp: 291.0000 - fp: 5266.0000 - tn: 176694.0000 - fn: 25.0000 - accuracy: 0.9710 - precision: 0.0524 - recall: 0.9209 - auc: 0.9815 - prc: 0.6431 - val_loss: 0.0952 - val_tp: 69.0000 - val_fp: 962.0000 - val_tn: 44529.0000 - val_fn: 9.0000 - val_accuracy: 0.9787 - val_precision: 0.0669 - val_recall: 0.8846 - val_auc: 0.9658 - val_prc: 0.6959
    Epoch 31/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1384 - tp: 295.0000 - fp: 5322.0000 - tn: 176638.0000 - fn: 21.0000 - accuracy: 0.9707 - precision: 0.0525 - recall: 0.9335 - auc: 0.9867 - prc: 0.6392 - val_loss: 0.0955 - val_tp: 69.0000 - val_fp: 983.0000 - val_tn: 44508.0000 - val_fn: 9.0000 - val_accuracy: 0.9782 - val_precision: 0.0656 - val_recall: 0.8846 - val_auc: 0.9663 - val_prc: 0.6955
    Epoch 32/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1470 - tp: 291.0000 - fp: 5037.0000 - tn: 176923.0000 - fn: 25.0000 - accuracy: 0.9722 - precision: 0.0546 - recall: 0.9209 - auc: 0.9839 - prc: 0.6577 - val_loss: 0.0941 - val_tp: 69.0000 - val_fp: 946.0000 - val_tn: 44545.0000 - val_fn: 9.0000 - val_accuracy: 0.9790 - val_precision: 0.0680 - val_recall: 0.8846 - val_auc: 0.9673 - val_prc: 0.7047
    Epoch 33/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1387 - tp: 294.0000 - fp: 5134.0000 - tn: 176826.0000 - fn: 22.0000 - accuracy: 0.9717 - precision: 0.0542 - recall: 0.9304 - auc: 0.9866 - prc: 0.6290 - val_loss: 0.0944 - val_tp: 69.0000 - val_fp: 993.0000 - val_tn: 44498.0000 - val_fn: 9.0000 - val_accuracy: 0.9780 - val_precision: 0.0650 - val_recall: 0.8846 - val_auc: 0.9679 - val_prc: 0.6961
    Epoch 34/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1330 - tp: 291.0000 - fp: 4962.0000 - tn: 176998.0000 - fn: 25.0000 - accuracy: 0.9726 - precision: 0.0554 - recall: 0.9209 - auc: 0.9878 - prc: 0.6378 - val_loss: 0.0908 - val_tp: 69.0000 - val_fp: 960.0000 - val_tn: 44531.0000 - val_fn: 9.0000 - val_accuracy: 0.9787 - val_precision: 0.0671 - val_recall: 0.8846 - val_auc: 0.9689 - val_prc: 0.6964
    Epoch 35/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1442 - tp: 292.0000 - fp: 4851.0000 - tn: 177109.0000 - fn: 24.0000 - accuracy: 0.9733 - precision: 0.0568 - recall: 0.9241 - auc: 0.9840 - prc: 0.6406 - val_loss: 0.0905 - val_tp: 69.0000 - val_fp: 928.0000 - val_tn: 44563.0000 - val_fn: 9.0000 - val_accuracy: 0.9794 - val_precision: 0.0692 - val_recall: 0.8846 - val_auc: 0.9688 - val_prc: 0.6967
    Epoch 36/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1334 - tp: 293.0000 - fp: 4786.0000 - tn: 177174.0000 - fn: 23.0000 - accuracy: 0.9736 - precision: 0.0577 - recall: 0.9272 - auc: 0.9882 - prc: 0.6383 - val_loss: 0.0893 - val_tp: 69.0000 - val_fp: 909.0000 - val_tn: 44582.0000 - val_fn: 9.0000 - val_accuracy: 0.9799 - val_precision: 0.0706 - val_recall: 0.8846 - val_auc: 0.9675 - val_prc: 0.7056
    Epoch 37/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1373 - tp: 292.0000 - fp: 4898.0000 - tn: 177062.0000 - fn: 24.0000 - accuracy: 0.9730 - precision: 0.0563 - recall: 0.9241 - auc: 0.9867 - prc: 0.6386 - val_loss: 0.0883 - val_tp: 69.0000 - val_fp: 881.0000 - val_tn: 44610.0000 - val_fn: 9.0000 - val_accuracy: 0.9805 - val_precision: 0.0726 - val_recall: 0.8846 - val_auc: 0.9676 - val_prc: 0.6972
    Epoch 38/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1272 - tp: 294.0000 - fp: 4748.0000 - tn: 177212.0000 - fn: 22.0000 - accuracy: 0.9738 - precision: 0.0583 - recall: 0.9304 - auc: 0.9898 - prc: 0.6691 - val_loss: 0.0875 - val_tp: 69.0000 - val_fp: 917.0000 - val_tn: 44574.0000 - val_fn: 9.0000 - val_accuracy: 0.9797 - val_precision: 0.0700 - val_recall: 0.8846 - val_auc: 0.9680 - val_prc: 0.6974
    Epoch 39/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1313 - tp: 291.0000 - fp: 4908.0000 - tn: 177052.0000 - fn: 25.0000 - accuracy: 0.9729 - precision: 0.0560 - recall: 0.9209 - auc: 0.9885 - prc: 0.6476 - val_loss: 0.0853 - val_tp: 69.0000 - val_fp: 881.0000 - val_tn: 44610.0000 - val_fn: 9.0000 - val_accuracy: 0.9805 - val_precision: 0.0726 - val_recall: 0.8846 - val_auc: 0.9660 - val_prc: 0.6975
    Epoch 40/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1221 - tp: 293.0000 - fp: 4756.0000 - tn: 177204.0000 - fn: 23.0000 - accuracy: 0.9738 - precision: 0.0580 - recall: 0.9272 - auc: 0.9907 - prc: 0.6578 - val_loss: 0.0826 - val_tp: 69.0000 - val_fp: 848.0000 - val_tn: 44643.0000 - val_fn: 9.0000 - val_accuracy: 0.9812 - val_precision: 0.0752 - val_recall: 0.8846 - val_auc: 0.9664 - val_prc: 0.6979
    Epoch 41/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1425 - tp: 290.0000 - fp: 4726.0000 - tn: 177234.0000 - fn: 26.0000 - accuracy: 0.9739 - precision: 0.0578 - recall: 0.9177 - auc: 0.9859 - prc: 0.6476 - val_loss: 0.0850 - val_tp: 69.0000 - val_fp: 911.0000 - val_tn: 44580.0000 - val_fn: 9.0000 - val_accuracy: 0.9798 - val_precision: 0.0704 - val_recall: 0.8846 - val_auc: 0.9662 - val_prc: 0.6980
    Epoch 42/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1266 - tp: 289.0000 - fp: 4662.0000 - tn: 177298.0000 - fn: 27.0000 - accuracy: 0.9743 - precision: 0.0584 - recall: 0.9146 - auc: 0.9906 - prc: 0.6353 - val_loss: 0.0848 - val_tp: 69.0000 - val_fp: 925.0000 - val_tn: 44566.0000 - val_fn: 9.0000 - val_accuracy: 0.9795 - val_precision: 0.0694 - val_recall: 0.8846 - val_auc: 0.9664 - val_prc: 0.6735
    Epoch 43/100
    90/90 [==============================] - 1s 6ms/step - loss: 0.1327 - tp: 289.0000 - fp: 4539.0000 - tn: 177421.0000 - fn: 27.0000 - accuracy: 0.9750 - precision: 0.0599 - recall: 0.9146 - auc: 0.9882 - prc: 0.6644 - val_loss: 0.0822 - val_tp: 69.0000 - val_fp: 868.0000 - val_tn: 44623.0000 - val_fn: 9.0000 - val_accuracy: 0.9808 - val_precision: 0.0736 - val_recall: 0.8846 - val_auc: 0.9661 - val_prc: 0.6982
    Epoch 44/100
    90/90 [==============================] - 1s 7ms/step - loss: 0.1438 - tp: 288.0000 - fp: 4670.0000 - tn: 177290.0000 - fn: 28.0000 - accuracy: 0.9742 - precision: 0.0581 - recall: 0.9114 - auc: 0.9852 - prc: 0.6182 - val_loss: 0.0844 - val_tp: 69.0000 - val_fp: 902.0000 - val_tn: 44589.0000 - val_fn: 9.0000 - val_accuracy: 0.9800 - val_precision: 0.0711 - val_recall: 0.8846 - val_auc: 0.9663 - val_prc: 0.6653
    Restoring model weights from the end of the best epoch.
    Epoch 00044: early stopping
    


```python
# 查看训练历史记录
plot_metrics(weighted_history)
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_32_0.png)
    



```python
# 评估指标
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)
```

    loss :  0.0886046513915062
    tp :  89.0
    fp :  1128.0
    tn :  55736.0
    fn :  9.0
    accuracy :  0.9800392985343933
    precision :  0.07313065230846405
    recall :  0.9081632494926453
    auc :  0.9855610132217407
    prc :  0.6803525686264038
    
    Legitimate Transactions Detected (True Negatives):  55736
    Legitimate Transactions Incorrectly Detected (False Positives):  1128
    Fraudulent Transactions Missed (False Negatives):  9
    Fraudulent Transactions Detected (True Positives):  89
    Total Fraudulent Transactions:  98
    


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_33_1.png)
    


在这里，我们可以看到，使用类权重时，由于存在更多假正例，准确率和精确率较低，但是相反，由于模型也找到了更多真正例，召回率和 AUC 较高。尽管准确率较低，但是此模型具有较高的召回率（且识别出了更多欺诈交易）。当然，两种类型的错误都有代价（客户也不希望因将过多合法交易标记为欺诈来打扰客户）。请在应用时认真权衡这些不同类型的错误。


```python
# 绘制ROC
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right');
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_35_0.png)
    



```python
# 绘制AUPRC
plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_prc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_prc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend();
```


    
![png](5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_files/5-Tensorflow%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90_36_0.png)
    


## 总结
由于可供学习的样本过少，不平衡数据的分类是固有难题。我们应该始终先从数据开始，尽可能多地收集样本，并充分考虑可能相关的特征，以便模型能够充分利用占少数的类。有时我们的模型可能难以改善且无法获得想要的结果，因此请务必牢记问题的上下文，并在不同类型的错误之间进行权衡。
