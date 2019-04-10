import jieba as jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def count_vec_ex():
    """
    单词列列表：将所有文章的单词统计到一个列表当中（重复的词只当做一次），
    默认会过滤掉单个字母
    单个中文也不行，中文要进行分词统计
    单个字母 或者 汉字对 结果没影响  只有词语才行
    :return:
    """
    # 实例化conunt

    """
    sklearn.feature_extraction.text.CountVectorizer(stop_words=[])
        CountVectorizer.fit_transform(X) X:文本或者包含文本字符串的可迭代对象 返回值：返回sparse矩阵
        CountVectorizer.inverse_transform(X) X:array数组或者sparse矩阵 返回值:转换之前数据格
        CountVectorizer.get_feature_names() 返回值:单词列表
        sklearn.feature_extraction.text.TfidfVectorizer
    :return:返回词频矩阵
    """
    count = CountVectorizer()

    # 对两篇文章进行特征抽取，中文要分词处理（jieba 分词错误）
    data = count.fit_transform(["人生 人生 苦短，我 喜 欢Python", "生 活太 长 久，我不 喜欢P ython"])

    # 内容
    print(count.get_feature_names())
    # 调用fit_transform方法输入数据并转换 （注意返回格式，利用toarray()进行sparse矩阵转换array数组）
    print(data.toarray())
    # print(data)

    return None


def dictvec():
    """
    对字典数据进行特征抽取
    :return:
    """
    # 实例化dictvec，默认返回sparse 矩阵
    # dic = DictVectorizer()
    # 返回数组
    """
    sklearn.feature_extraction.DictVectorizer(sparse=True,…)
        DictVectorizer.fit_transform(X) X:字典或者包含字典的迭代器返回值：返回sparse矩阵
        DictVectorizer.inverse_transform(X) X:array数组或者sparse矩阵 返回值:转换之前数据格式
        DictVectorizer.get_feature_names() 返回类别名称
    """
    dic = DictVectorizer(sparse=False)

    # dictvec调用fit_transform
    # 三个样本的特征数据（字典形式）
    temperature_ = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
                    {'city': '深圳', 'temperature': 30}]
    data = dic.fit_transform(temperature_)

    # 打印特征抽取之后的特征结果
    print(dic.get_feature_names())
    # one-hot 编码：对特征当中有类别的信息做处理理———>one-hot编码
    print(data)

    return None


def cutword():
    """
    进行分词处理
    :return:
    """
    c1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    c2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    c3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 将这三个结果，转换成列表，变成以空格隔开的字符串
    content1 = ' '.join(list(c1))
    content2 = ' '.join(list(c2))
    content3 = ' '.join(list(c3))

    return content1, content2, content3


def chinesevec():
    # 实例化conunt
    count = CountVectorizer(stop_words=['不会', '如果'])

    # 进行三句话的分词
    content1, content2, content3 = cutword()

    # 对两篇文章进行特征抽取
    data = count.fit_transform([content1, content2, content3])

    # 内容
    print(count.get_feature_names())
    print(data.toarray())
    # print(data)

    return None


def tfidfvec():
    """
    TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
    TF-IDF作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
    TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
    TF-IDF作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
    分类机器学习算法进行文章分类中前期数据处理方式
    :return:
    """
    # 实例化conunt
    tfidf = TfidfVectorizer()

    # 进行三句话的分词
    content1, content2, content3 = cutword()

    # 对两篇文章进行特征抽取
    data = tfidf.fit_transform([content1, content2, content3])

    # 内容
    print(tfidf.get_feature_names())
    print(data.toarray())
    # print(data)

    return None


if __name__ == '__main__':
    tfidfvec()
