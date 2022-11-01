import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams[u'font.sans-serif'] = 'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False
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
        plt.show()
        # 是否启用下采样策略
        if under:
            print('您已选择下采样策略')
            self.names = '下采样'
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
            self.names = '过采样'
            print('您已选择过采样策略')
            oversample = SMOTE(random_state=0)
            X_over,y_over = oversample.fit_resample(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=test_size, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            self.names = '无采样策略'
            print('您没有启用任何数据采样策略')
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
            return X_train,X_test,y_train,y_test

    # 分类模型训练
    def modelsclasses(self,data=None,models=None,scoring=None):
        scoring = 'f1'
        cv = 3
        # 训练数据处理
        plt.plot([0, 1], [0, 1], 'k--', label='对比线')
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
        # 指标池
        models_list = []
        acc_list = []
        pre_list = []
        f1_list = []
        recall_list = []
        # 文件命名
        self.sample = models['samplename']
        for model,parm in zip(models['models'],models['parms']):
            model_name = str(model).split('(')[0]
            parms = models['parm_type']
            # 模型训练及参数调节
            if parms == 'random_search':
                print('您选择随机参数调参法')
                rm = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=parm,
                    scoring=scoring,
                    cv=cv,
                    # verbose=2,
                    random_state=42,
                    n_jobs=-1
                )
                rm.fit(X_train,y_train)
                best_parms = rm.best_params_
                best_models = rm.best_estimator_
                pred = best_models.predict(X_test)
                # 模型评估指标（数量）
                # 1 混淆矩阵
                metrix = confusion_matrix(y_test,pred)
                # 2 准确率
                acc = accuracy_score(y_test,pred)
                # 3 精度值
                pre = precision_score(y_test,pred)
                # 4 召回值
                recall = recall_score(y_test,pred)
                # 5 f1
                f1 = f1_score(y_test,pred)
                # roc 曲线
                fpr,tpr,thresholds = roc_curve(y_test,pred)
                print('最后的参数为：{}'.format(str(best_parms)))
                print('精度值为：{}'.format(str(f1)))
                title = model_name + '随机调参法' + self.names
                tc = 'r-'
                self.plot_score(fpr=fpr,tpr=tpr,tc=tc,title=title)
                # 添加参数
                models_list.append(title)
                acc_list.append(acc)
                pre_list.append(pre)
                f1_list.append(f1)
                recall_list.append(recall)
            elif parms == 'grid_serch':
                print('您已选择了网络调参法')
                gm = GridSearchCV(
                    estimator=model,
                    param_grid=parm,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=2,
                )

            print('默认参数训练')
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            # 模型评估指标（数量）
            # 1 混淆矩阵
            metrix = confusion_matrix(y_test,pred)
            # 2 准确率
            acc = accuracy_score(y_test,pred)
            # 3 精度值
            pre = precision_score(y_test,pred)
            # 4 召回值
            recall = recall_score(y_test,pred)
            # 5 f1
            f1 = f1_score(y_test,pred)
            # roc 曲线
            title = model_name + '默认参数' + self.names
            tc = 'b-.'
            fpr,tpr,thresholds = roc_curve(y_test,pred)
            self.plot_score(fpr=fpr,tpr=tpr,tc=tc,title=title)
            models_list.append(title)
            acc_list.append(acc)
            pre_list.append(pre)
            f1_list.append(f1)
            recall_list.append(recall)
        result_score = pd.DataFrame(
            {
                '模型名称':models_list,
                '准确率':acc_list,
                '精度值':pre_list,
                '召回率':acc_list,
                'f1精度':f1_list
            })
        result_score.to_csv(self.sample + '指标数据.csv')
        print('已在本地生成评估指标文件')

    # 可视化操作
    def plot_score(self,fpr=None,tpr=None,tc=None,title=None):
        plt.plot(fpr,tpr,label=title)
        plt.title('ROC曲线图')
        plt.ylabel('真正率（召回率）')
        plt.xlabel('假正率')
        plt.legend()
        plt.savefig(self.sample + 'ROC.pdf')
        # plt.show()
        print('已在本地生成ROC曲线图')

    # 聚类模型训练
    def cluster(self):
        pass

if __name__ == '__main__':
    path = './wpbc.csv'
    test = Ml(path)
    # 过采样策略
    data = test.makedata(under=True)
    rf_models = RandomForestClassifier()
    lr_models = LogisticRegression()
    svm_models = SVC()
    models_lists = []
    models_lists.append(rf_models)
    models_lists.append(lr_models)
    models_lists.append(svm_models)
    # 树模型的个数
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # 最大特征选择
    max_features = ['auto', 'sqrt']
    # 树模型的最大深度
    max_depth = [int(x) for x in np.linspace(10, 30, 2)]
    # 节点的最小分裂样本
    min_samples_split = [2, 5, 10]
    # 最小叶子节点数
    min_samples_leaf = [1, 2, 4]
    # 是否选择bootstrap采样方案
    bootstrap = [True, False]
    # 构建参数空间（为RandomizedSearchCV提供随机参数组合样本）
    random_grid_tree = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap, }
    # 逻辑回归的参数空间
    random_grid_lr = {
        'penalty':['l1','l2'],
        'C':[0.0001,0.001,0.01,0.1,1,10,100],
        'solver':['liblinear']
    }
    # 封装参数池
    random_grid_svc = {
        'C':[0.0001,0.001,0.01,0.1,1,10,100]
    }
    # 封装参数字典
    model_parm = {
        'samplename':'下采样',
        'parm_type':'random_search',
        'models':models_lists,
        'parms':[random_grid_tree,random_grid_lr,random_grid_svc]
    }


    print(model_parm['models'])
    test.modelsclasses(data=data,models=model_parm)
    plt.show()