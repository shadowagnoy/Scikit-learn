from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def standscalar():
    """
    对约会对象数据进行归一化处理
    :return:
    """
    # 读取数据，选择要处理的特征
    dating = pd.read_csv("../../../data/dating.txt")

    data = dating[['milage', 'Liters', 'Consumtime']]

    # 实例化minmaxscaler进行fit_transform

    std = StandardScaler()

    data = std.fit_transform(data)

    print(data)

    return None


def minmaxscalar():
    """
    milage,Liters,Consumtime,target
    40920,8.326976,0.953952,3
    14488,7.153469,1.673904,2
    26052,1.441871,0.805124,1
    75136,13.147394,0.428964,1
    对约会对象数据进行归一化处理
    :return:
    """
    # 读取数据，选择要处理的特征
    dating = pd.read_csv("../../../data/dating.txt")

    data = dating[['milage', 'Liters', 'Consumtime']]

    # 实例化minmaxscaler进行fit_transform

    mm = MinMaxScaler(feature_range=(2, 3))

    data = mm.fit_transform(data)

    print(data)

    return None


def mm():
    """
    对二维数组进行归一化处理
    :return:
    """
    mm = MinMaxScaler(feature_range=(2, 3))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)

    return None


def std():
    std = StandardScaler()

    data = std.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)
    return None


if __name__ == '__main__':
    std()
