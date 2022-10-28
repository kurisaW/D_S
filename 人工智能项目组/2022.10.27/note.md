### sklearn.svm.SVC中kernel参数说明

- [常用核函数](#_1)
- - [线性核函数kernel='linear'](#kernellinear_2)
  - [多项式核函数kernel='poly'](#kernelpoly_5)
  - [径向基核函数kernel='rbf'](#kernelrbf_10)
  - [sigmod核函数kernel='sigmod'](#sigmodkernelsigmod_14)

# 常用核函数

## 线性核函数kernel=‘linear’

![在这里插入图片描述](https://latex.codecogs.com/svg.latex?%5Cinline%20kernel=%3Cx,x%27%3E)



采用线性核kernel='linear’的效果和使用sklearn.svm.LinearSVC实现的效果一样，但采用线性核时速度较慢，特别是对于大数据集，推荐使用线性核时使用LinearSVC

## 多项式核函数kernel=‘poly’

![在这里插入图片描述](https://latex.codecogs.com/svg.latex?\inline kernel=(\gamma <x,x'>+r)^{d})  

degree代表d，表示多项式的次数 
gamma为多项式的系数，coef0代表r，表示多项式的偏置  
注：coef0是sklearn.svm.SVC中的参数，详情点击[SVC参数说明](https://blog.csdn.net/qq_37007384/article/details/88410998)

## 径向基核函数kernel=‘rbf’

![在这里插入图片描述](https://latex.codecogs.com/svg.latex?kernel=exp(-\gamma \left \| x-x' \right \|^{2}))  

可以将gamma理解为支持向量影响区域半径的倒数，gamma越大，支持向量影响区域越小，决策边界倾向于只包含支持向量，模型复杂度高，容易过拟合；gamma越小，支持向量影响区域越大，决策边界倾向于光滑，模型复杂度低，容易欠拟合；  
gamma的取值非常重要，即不能过小，也不能过大

## sigmod核函数kernel=‘sigmod’

![在这里插入图片描述](https://latex.codecogs.com/svg.latex?\inline kernel=tanh(\gamma <x,x'>+r))  

coef0控制r,sigmod核函数是线性核函数经过tanh函数映射变化