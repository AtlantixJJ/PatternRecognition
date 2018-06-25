# HW4

2015011313 徐鉴劲 计54

## KNN分类

使用sklearn库的K近邻分类方法`KNeighborsClassifier`，核心代码如下：

```python
def do_KNN(data, label, testdata, testlabel):
    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(data, label)
    predict = knn.predict(testdata)
```

### 实验结果

实验重复了五次，平均正确率是86.93%，标准差是2.91%。

## 复现实验的结果

## Linear

70.67 2.51

69.47 2.22