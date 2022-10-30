import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class Ml:

    def __init__(self,path):
        self.path = path

    # 处理数据（切分，标准化，归一化）
    def makedata(self,test_size=0.3,under=False,smote=False):
        data = pd.read_csv(self.path)
        X = data.iloc[:,:30]
        y = data.iloc[:,-1]
        y_count = pd.value_counts(y)
        print('数据位二分类数据，正负样本差距为：',str(y_count[1]-y_count[0]))
        y_count.plot(kind='bar')
        plt.title('Labels Numbers')
        plt.xlabel('Class')
        plt.ylabel('Nums')
        # 是否启用下采样策略
        if under:
            # 正常样本值的索引值（target = 1 ）
            n_indeices = data[data.target == 1].index
            # 处理异常样本值
            f_len = len(data[data.target == 0])
            f_indeices = data[data.target == 0].index
            # 在正常样本中随机获取和异常样本同样数据量的样本值
            random_n_indeices = np.random.choice(n_indeices,f_len,replace=False)
            # 拼接数据索引
            under_data_indices = np.concatenate([f_indeices,random_n_indeices])
            under_data = data.iloc[under_data_indices,:]
            X_under = under_data.iloc[:,:30]
            y_under = under_data.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=test_size, random_state=42)
            return X_train, X_test, y_train, y_test
        elif smote:
            oversample = SMOTE(random_state=0)
            X_over,y_over = oversample.fit_resample(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=test_size, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            # 没有启用任何数据采样策略
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
            return X_train,X_test,y_train,y_test

    # 分类模型训练
    def modelsclasses(self,data=None,models=None,parms=False,params_poll=None,scoring=None,cv=3):
        # 训练数据处理
        X_train,X_test,y_train,y_test = data[0],data[1],data[2],data[3]
        # 模型训练及参数调节
        if parms == 'random_search':
            rm = RandomizedSearchCV(estimator=models,
                                    param_distributions=params_poll,
                                    scoring=scoring,
                                    cv=cv,
                                    verbose=2,
                                    random_state=42,
                                    n_jobs=-1)
            rm.fit(X_train,y_train)
            best_parms = rm.best_params_
        elif parms == 'grid_serch':
            gm = GridSearchCV(
                estimator=models,
                param_grid=params_poll,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=2,
            )
        else:
            print('默认参数训练')
            models.fit(X_train,y_train)
            pred = models.predict(X_test)
            metrix = confusion_matrix(y_test,pred)
            print(metrix)

    # 聚类模型训练
    def cluster(self):
        pass

if __name__ == '__main__':
    path = './wpbc.csv'
    test = Ml(path)
    data = test.makedata(smote=True)
    models = RandomForestClassifier(n_estimators=1000)
    # 参数池
    parms_dict = {}
    test.modelsclasses(data=data,models=models,parms='random_search',params_poll=parms_dict,scoring='f1',cv=5)
    # print(len(data[0]))
    # # plt.show()