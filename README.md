Kaggle California Housing Prices报告

 

 

一、问题定义

该问题来源于kaggle，主要是通过上世纪90年代的加利福利亚地区的房价的数据来训练模型，最终可以较为准确预测出对应房屋的价格。该问题实际上还是一道回归问题。

从kaggle上获取数据后，为了更好的训练，每个样本的特征属性和标签（即房屋价格）需要进行相关处理（包括取出异常值、填充缺省值、特征转换，数据转化等），然后构建多种误差较小的模型，找到最佳参数后将模型stacking融合，获取更小的误差，达到更好的预测效果，然后将模型保存起来。本次题目使用rmse均方根误差作为衡量模型标准，最终降低到0.10左右。

 

 

二、数据获取

首先将Kaggle上提供的相关的训练集和测试集下载下来，主要是train.csv和test.csv，训练集和测试集都是DataFrame数据结构。

然后查看其数据结构。训练集的shape是（1460,81）表示共有1460个样本，每个样本有81个属性，测试集的shape是（1459,80）表示共有1459个测试样本，每个测试样本有1459个属性。除此以外，通过head()函数，可以查看训练集和测试集的前5行数据，进行直观的认识，如下所示。

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps16.jpg) 

从上面，我们可以发现训练集有81个属性，但第一个Id属性没有意义，之后可以考虑删去。最后一个属性SalePrice是该样本数据的标签，其中有些属性是字符型，有些是数据型。而测试集则没有SalePrice标签属性，因为其不参与训练。

 

 

三、数据研究

该部分主要是通过可视化或者直接打印来查看属性之间的相关性，每个属性的分布情况等，从而为后面数据清洗，找出与标签属性SalePrice最相关的其他属性做准备。

1、作图来显示相关性

corrmat=train_data.corr()

plt.figure(figsize=(12,9))

cols=corrmat.nlargest(10,'SalePrice')['SalePrice'].index

cm=np.corrcoef(train_data[cols].values.T)

sns.set(font_scale=1.25)

hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size': 10},xticklabels=cols.value

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps17.jpg) 

2、直接输出相关度高的属性

利用DataFrame数据类型的函数corr()，并将与SalePrice属性相关度大于0.5的所有属性取出来：

Corr=train_data.corr()

print(Corr[Corr['SalePrice']>0.5])，结果显示如下。

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps18.jpg) 

发现除了SalePrice自身外，还有10个属性与SalePrice的相关程度较高。

 

 

 

四、数据准备

1、去除异常数据

上面找出10种与SalePrice最相关的属性后，由于其对标签影响最大，所以我们就要在这些属性上进行一个处理—除去少数非正常分布的属性的值。可以通过画出每个属性与SalePrice的关系的散点图来查看那些点偏离正常分布，画出的部分图像如下：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps19.jpg) 

观察这些分布，可以删去一些异常点：

train_data.drop(train_data[(train_data['GrLivArea']>4000)&(train_data['SalePrice']<200000)].index,inplace=True)

train_data.reset_index(drop=True, inplace=True)

当然，其他属性也有可以删去的点，但偏差不是太大，而且数据少，所以可以忽略。除此之外，可以通过Ridge和ElasticNet训练了训练集，并对训练集进行预测，找出两个算法中预测效果都不理想的样本作为离群值，然后再删去，效果更好。

 

2、填充缺省数据

观察数据会发现许多的样本的属性值是NaN，这是DataFrame类型数据的缺省的表示形式，对于缺失数据的处理，通常会有以下几种做法：

1）如果缺失的数据过多，可以考虑删除该列特征

2）用平均值、中值、分位数、众数、随机值等替代。但是效果一般，因为等于人为增加了噪声

3）用插值法进行拟合

4）用其他变量做预测模型来算出缺失变量。效果比方法1略好。有一个根本缺陷，如果其他变量和缺失变量无关，则预测的结果无意义

5）最精确的做法，把变量映射到高维空间。比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失。但计算量变大。

将训练集和测试集连接在一起来一起进行缺省值补充和后面的特征工程：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps20.jpg) 

首先将整个数据集的所有属性的值的缺省情况（包括缺省数量和缺省占比）统计出来，进行争对性处理。

count=all_features.isnull().sum().sort_values(ascending=False)

ratio=count/len(all_features)

nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps21.jpg) 

缺省代表的是这个房子没有这种特征，用 “None” 来填补。（字符串类型）

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt","GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]

缺省代表的是这个房子本身没有这种特征，用0来填补。（数字类型）

cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]

最后剩下的属性的缺省用该属性的众数来填充。

cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]

最后还剩余一个属性LotFrontage需要特别处理，LotFrontage这个特征与LotAreaCut和Neighborhood有比较大的关系，所以这里用这两个特征分组后的中位数进行插补。

all_features["LotAreaCut"] = pd.qcut(all_features.LotArea,10)

all_features.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])

\# 因为有些LotArea与Neighborhood 的组合是没有的，所以这里只使用LotArea

all_features['LotFrontage']=all_features.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

至此所有缺省值都填充完毕，但是为了对该场景数据下，缺省值还再可以更加细致琢磨琢磨，训练效果应该会更好。

 

 

五、特征工程

该部分是最麻烦，重要的部分，和前面的数据处理部分一起决定了最终误差可以达到的最小值，后面的模型的选择和融合只能尽可能接近这个最小值。该部分主要的工作包括把特征的类别变量转换成数值，属性分布正态化，提炼构造新属性，分割属性，数据编码等。其中可以用到pipe来，lasso等。

 

1、转化特征的字符类别到数字类别

NumStr=["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]

以上特征是字符特征，转化成数值型特征，所以这样后属性统一，更易于特征训练。

然后这样按照SalePrice的均值和中位值对22个特征的取值进行数值化，再构建成map，其中的一个特征的map构建如下：

all_features["oMSSubClass"] = all_features.MSSubClass.map({'180': 1,

​                        '30': 2, '45': 2, '190': 3, '50': 3, '90': 3,'85': 4, '40': 4,

 '160': 4,'70': 5, '20': 5, '75': 5, '80': 5, '150': 5,'120': 6, '60': 6})

这样就建立了字符与数值的联系，从而实现向数值的转换，除此以外还有上面的NumStr里面的其他元素也都需要的构建字典生成新的特征。具体见代码。

 

2、特征分布正态化以及编码

首先从大到小排序前20个属性的偏值，查看属性是否符合正态分布：np.abs(train_data.skew()).sort_values(ascending=False).head(20)

这里可以使用pipeline来进行特征的自动化处理，pipeline可以看作一个特征处理的堆叠器，每个特征处理的输出作为下一个的处理的输入，半自动化，是非常好用。这里我们定义两个类：labelenc和skew_dummies，分别用于标签的向量化编码和将特征的分布正态化，然后将作为它们作为pipeline的参数，最后使用fit_transform自动进行串行处理上面定义的两种处理。详细见代码。

 

3、添加新特征

将原始特征进行组合通常能产生意想不到的效果，然而这个数据集中原始特征有很多，不可能所有都一一组合，所以这里先用Lasso进行特征筛选，选出较重要的一些特征进行组合。

lasso=Lasso(alpha=0.001)

lasso.fit(X_scaled,y_log)

FI_lasso=pd.DataFrame({"FeatureImportance":lasso.coef_},index=data_pipe.columns)

FI_lasso.sort_values("Feature Importance",ascending=False)

FI_lasso[FI_lasso["FeatureImportance"]!=0].sort_values("FeatureImportance").plot(kind="barh",figsize=(15,25))

plt.xticks(rotation=90)

plt.show()

然后再原来特征的基础上添加新属性，这里还是通过定义一个类add_feature，在其中定义fit()和transform()方法来通过其他属性来组合新属性，例如：

X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]，详细见具体代码。

与前面的向量化处理labelenc和正态化处理skew_dummies一样，都添加进pipeline里面一起进行：

pipe = Pipeline([  ('labenc', labelenc()),

​                   ('add_feature', add_feature(additional=2)),

​                   ('skew_dummies', skew_dummies(skew=1)),])

 

经过pipeline后，我们得到了想要的处理好的特征数据，我们可以将这些数据保存成文件，这样就不用每次运行程序都把数据处理一遍，也方便将文件的拷贝。最后再将训练集和测试集从整个数据集中分割出来，单独处理，它们形状和训练集前五行如下：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps22.jpg) 

除此之外，将训练数据再进行标准/归一化RobustScaler以及PCA来降低自己添加的特征之间过高的相关性，可以进一步优化数据。具体见代码。

 

 

五、研究模型

我们可以有许多机器回归模型来选择，但太多反而不知道那些在这种情况下最优，所以可以先通过循环大致找到一些表现良好的模型，然后着通过gridsearch将这些模型中的参数调到最优，最后通过stacking将这些模型融合在一起，从而达到比任何一个单独模型都好的效果。

 

1、评估标准

定义一个评估标准来衡量模型的优劣，即均方误差RMSE：

def rmse_cv(model,X,y):

​    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))

return rmse

2、模型选择

先遍历以下的模型找出几个的rmse较小的，可以发现Ridge，Lasso，SVR，ElasticNet，KernelRidge，BayesianRidge这六种模型的rmse都在0.11左右，比较好。

models=[LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),      ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), ExtraTreesRegressor(),XGBRegressor()]

然后通过gridsearch来为这六种模型的选择最好的参数，从而实例化这六种模型：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps23.jpg) 

3、模型融合

常见的模型融合方法有四种：

voting（投票法）：选出效果最好的或者；

stacking：先用多个模型来训练得到特征组合在一起，再作为下一层模型的输入来训练，注意模型之间尽可能独立，也不能太多；

bagging：类似于加权平均，主要是来优化泛化误差，例如多个决策树融合成随机森林；

boosting：串行结构，每一个弱预测器修正前一个弱预测期的结果，优化拟合性，例如adaBoost，XGBoost，lightGBM等。

1）voting和stacking可以融合不同的模型，bagging和boosting只能融合同类型模型

2）模型融合方法还可以从另一个角度还以分为平均法（boosting）、投票法（voting）和学习法（staking和bagging）

该题目主要使用stacking进行模型融合，stacking的模型结构如下所示：![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps24.png)

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps25.jpg) 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps26.jpg) 

Stacking模型

 

这里将lasso,ridge,svr,ker,ela,bay模型作为第一层的基回归器，ker用来融合基回归器的元回归器。程序中可以直接调用mlxtend的StackingRegressor，也可以自定义stacking。但是尝试发现使用库中的stacking效果反而不如自定义，选择使用自定义的stacking。最终融合后训练和测试结果如下：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps27.jpg) 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps28.jpg) 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps29.jpg) 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml7000\wps30.jpg) 

经过多次尝试，最终均方误差降低到0.101，此时保存模型，训练结束。

 

六、总结

这算是一个比较完整的kaggle的题目的处理流程和必要手段，包括：

问题定义，下载数据，观察结构，找最相关部分特征；

除去异常值，填充缺省值，部分特征正态化，特征编码向量化，增加新特征；

定义评估标准，选择表现好的模型，模型参数最优化，模型融合。

整个过程最重要的就是特征处理，需要非常仔细和特别关注特征的相关性质，往往需要做不同处理，比如时间数据等。部分属性偏值较大时表明特征的分布缺乏合理性，需要令其符合正态分布。Gridsearch调参和模型融合是基础，不同类型的模型融合的侧重点和应用场景有所不同，注意区分使用。

其中仍然有许多可以改进的地方，例如异常值除了画图观察，还可以使用岭回归训练来找出不合适的值；数据的描述文件没有非常仔细的研究导致部分数据不知道其合适的处理方法；数据观察不够充分，特征工程还有很多技术不懂使用，或者说自己的题目经验还是远远不足；通过SVM支持向量机和生成器构造新属性等。

 

 
