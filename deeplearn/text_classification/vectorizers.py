
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
"""
提取特征
"""

def tf_idf(xtrain, xvalid, min_df,max_df,stwlist):
    def number_normalizer(tokens):
        """ 将所有数字标记映射为一个占位符（Placeholder）。
        对于许多实际应用场景来说，以数字开头的tokens不是很有用，
        但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
        """
        return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

    class NumberNormalizingVectorizer(TfidfVectorizer):
        def build_tokenizer(self):
            tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
            return lambda doc: list(number_normalizer(tokenize(doc)))

    # 利用刚才创建的NumberNormalizingVectorizer类来提取文本特征看

    tfv = NumberNormalizingVectorizer(min_df=3,
                                      max_df=0.5,
                                      max_features=None,
                                      ngram_range=(1, 2),
                                      use_idf=True,
                                      smooth_idf=True,
                                      stop_words=stwlist)

    # 使用TF-IDF来fit训练集和测试集（半监督学习）
    tfv.fit(list(xtrain) + list(xvalid))
    xtrain_v = tfv.transform(xtrain)
    xvalid_v = tfv.transform(xvalid)
    return xtrain_v, xvalid_v