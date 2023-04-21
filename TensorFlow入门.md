## 入门



```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
```

    TensorFlow version: 2.5.0
    

### 加载数据集（MNIST数据集）



```python
mnist = tf.keras.datasets.mnist

# 加载数据（训练集和测试集）
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("输出训练集：",x_train[1][0])

# 将样本数据从整数转换为浮点数
x_train, x_test = x_train / 255.0, x_test / 255.0
print("转化为浮点数：",x_train[0][0])

```

    输出训练集： [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    转化为浮点数： [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0.]
    

### 构建机器学习模型
通过堆叠层来构建 tf.keras.Sequential 模型。


```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # 展平输入，不影响批量大小。
  tf.keras.layers.Dense(128, activation='relu'), # 全连接层，activation为激活函数。
  tf.keras.layers.Dropout(0.2), # 防止过拟合
  tf.keras.layers.Dense(10)
])

```

对于每个样本，模型都会返回一个包含 logits 或 log-odds 分数的向量，每个类一个。


```python
predictions = model(x_train[:1]).numpy()
predictions
```




    array([[ 2.7237484e-01,  4.6986026e-01,  4.2428455e-01, -2.3238268e-02,
            -1.2808293e-04, -3.7227139e-02,  7.4417830e-01, -1.5574691e-01,
             1.7269371e-01, -3.1401542e-01]], dtype=float32)



tf.nn.softmax 函数将这些 logits 转换为每个类的概率：


```python
tf.nn.softmax(predictions).numpy()
```




    array([[0.10709219, 0.1304742 , 0.12466121, 0.07968464, 0.08154762,
            0.07857771, 0.17165618, 0.06979541, 0.09693192, 0.0595788 ]],
          dtype=float32)



【说明】：可以将 tf.nn.softmax 烘焙到网络最后一层的激活函数中。虽然这可以使模型输出更易解释，但不建议使用这种方式，因为在使用 softmax 输出时不可能为所有模型提供精确且数值稳定的损失计算。
使用 losses.SparseCategoricalCrossentropy 为训练定义损失函数，它会接受 logits 向量和 True 索引，并为每个样本返回一个标量损失。


```python
# SparseCategoricalCrossentropy用于计算多分类问题的交叉熵。标签应为一个整数，而不是one-hot编码形式。
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

此损失等于 true 类的负对数概率：如果模型确定类正确，则损失为零。

这个未经训练的模型给出的概率接近随机（每个类为 1/10），因此初始损失应该接近 -tf.math.log(1/10) ~= 2.3。


```python
loss_fn(y_train[:1], predictions).numpy()
```




    2.5436673



在开始训练之前，使用 Keras Model.compile 配置和编译模型。将 optimizer 类设置为 adam，将 loss 设置为您之前定义的 loss_fn 函数，并通过将 metrics 参数设置为 accuracy 来指定要为模型评估的指标。


```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

### 训练并评估模型
使用 Model.fit 方法调整您的模型参数并最小化损失：


```python
model.fit(x_train, y_train, epochs=5)
```

    Epoch 1/5
    1875/1875 [==============================] - 2s 754us/step - loss: 0.2973 - accuracy: 0.9128
    Epoch 2/5
    1875/1875 [==============================] - 1s 735us/step - loss: 0.1422 - accuracy: 0.9574
    Epoch 3/5
    1875/1875 [==============================] - 1s 746us/step - loss: 0.1068 - accuracy: 0.9679
    Epoch 4/5
    1875/1875 [==============================] - 1s 748us/step - loss: 0.0879 - accuracy: 0.9729
    Epoch 5/5
    1875/1875 [==============================] - 1s 758us/step - loss: 0.0744 - accuracy: 0.9765
    




    <tensorflow.python.keras.callbacks.History at 0x1c13dc92d30>



Model.evaluate 方法通常在 "Validation-set"（验证集） 或 "Test-set" （测试集）上检查模型性能。


```python
model.evaluate(x_test,  y_test, verbose=2)
```

    313/313 - 0s - loss: 0.0737 - accuracy: 0.9756
    




    [0.07368186861276627, 0.975600004196167]



现在，这个照片分类器的准确度已经达到 98%。

---

## 参考
https://www.tensorflow.org/tutorials/quickstart/beginner?hl=zh-cn
