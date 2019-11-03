'''
@version: python3.6
@author: Administrator
@file: kaggle_house_prices.py
@time: 2019/09/25
'''


#具体题目和讲解请见于该地址：http://zh.d2l.ai/chapter_deep-learning-basics/kaggle-house-price.html


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from mlxtend.regressor import StackingRegressor
#1.获取数据并查看
train_data=pd.read_csv('D:/用户目录/我的文档/keras实践/house_prices/train.csv')#返回的是dataframe类型
test_data=pd.read_csv('D:/用户目录/我的文档/keras实践/house_prices/test.csv')
print(train_data.shape,test_data.shape)
# data=train_data.iloc[0:4,[0,1,2,3,-1]]
print(train_data.head())#输出前5行数据
print(test_data.head())
#2、去除异常数据和偏值分析
# 使用下列两种方式都可以查看与SalePrice最相关的10个属性
# #作图来显示相关性
# corrmat=train_data.corr()
# plt.figure(figsize=(12,9))
# cols=corrmat.nlargest(10,'SalePrice')['SalePrice'].index
# cm=np.corrcoef(train_data[cols].values.T)
# sns.set(font_scale=1.25)
# hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size': 10},xticklabels=cols.values,yticklabels=cols.values)
# plt.show()
# 或者不作图，直接输出相关性大于0.5的属性数据
Corr=train_data.corr()
print(Corr[Corr['SalePrice']>0.5])

#已经知道有哪些特征与SalePrice比较相关，加下来绘制散点图来具体看每个属性与SalePrice的关系
# sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'BsmtFullBath', 'YearBuilt'],y_vars=['SalePrice'],data=train_data,dropna=True)
# plt.show()

#根据散点图的显示来去除异常值
train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<200000)].index,inplace=True)
# train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index,inplace=True)
# train_data.drop(train_data[(train_data['YearBuilt']<1900) & (train_data['SalePrice']>400000)].index,inplace=True)
# train_data.drop(train_data[(train_data['BsmtFullBath']<1) & (train_data['SalePrice']>300000)].index,inplace=True)
# train_data.drop(train_data[(train_data['TotalBsmtSF']>6000) & (train_data['SalePrice']<200000)].index,inplace=True)
train_data.reset_index(drop=True, inplace=True)
print(train_data.shape)
#除了Ridge和ElasticNet训练了训练集，并对训练集进行预测，找出两个算法中预测效果都不理想的样本作为离群值


#从大到小排序前20个属性的偏值，查看属性是否符合正态分布，符合的后面特征工程处理
#np.abs(train_data.skew()).sort_values(ascending=False).head(20)

#3、缺省值处理
'''
对于缺失数据的处理，通常会有以下几种做法：
如果缺失的数据过多，可以考虑删除该列特征
用平均值、中值、分位数、众数、随机值等替代。但是效果一般，因为等于人为增加了噪声
用插值法进行拟合
用其他变量做预测模型来算出缺失变量。效果比方法1略好。有一个根本缺陷，如果其他变量和缺失变量无关，则预测的结果无意义
最精确的做法，把变量映射到高维空间。比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失。缺点就是计算量会加大
'''

#将数据集连接组合在一起。等所有的需要的预处理进行完之后，再把他们分隔开。
# all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#测试数据接在训练数据后面。训练集和测试集第一列是序号，无用舍去，测试集没有标签
all_features=pd.concat([train_data,test_data], ignore_index=True)
all_features.drop(['Id'],axis=1, inplace=True)
numeric=all_features.dtypes[all_features.dtypes!='object'].index#取出dataframe所有非字符串类型的列名（特征名）
print(all_features.shape)
print(all_features.head())
# print(numeric)

#倒叙统计每种属性的缺省总数和占总数比重
count=all_features.isnull().sum().sort_values(ascending=False)
ratio=count/len(all_features)
nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
print(nulldata)
#缺失代表的是这个房子本身没有这种特征，用 “None” 来填补。（字符串类型）
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt",
         "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    all_features[col].fillna("None", inplace=True)
##缺失代表的是这个房子本身没有这种特征，用 0 来填补。（数字类型）
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    all_features[col].fillna(0, inplace=True)
#众数填充
cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    all_features[col].fillna(all_features[col].mode()[0], inplace=True)
#LotFrontage这个特征与LotAreaCut和Neighborhood有比较大的关系，所以这里用这两个特征分组后的中位数进行插补。
#use qcut to divide it into 10 parts.
all_features["LotAreaCut"] = pd.qcut(all_features.LotArea,10)
all_features.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])
all_features['LotFrontage']=all_features.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# Since some combinations of LotArea and Neighborhood are not available, so we just LotAreaCut alone.
all_features['LotFrontage']=all_features.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
print(all_features.isnull().sum().sort_values(ascending=False))



#4、特征工程
# 以上特征是字符特征，转化成数值型特征，所以这样后属性统一，更易于特征训练。
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    all_features[col]=all_features[col].astype(str)
#Convert some numerical features into categorical features
#build as many features as possible and trust the model to choose the right features
#groupby SalePrice according to one feature and sort it based on mean and median.


#映射这些值
def map_values():
    all_features["oMSSubClass"] = all_features.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    all_features["oMSZoning"] = all_features.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    all_features["oNeighborhood"] = all_features.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    all_features["oCondition1"] = all_features.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    all_features["oBldgType"] = all_features.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    all_features["oHouseStyle"] = all_features.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    all_features["oExterior1st"] = all_features.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    all_features["oMasVnrType"] = all_features.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    all_features["oExterQual"] = all_features.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    all_features["oFoundation"] = all_features.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    all_features["oBsmtQual"] = all_features.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    all_features["oBsmtExposure"] = all_features.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    all_features["oHeating"] = all_features.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    all_features["oHeatingQC"] = all_features.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    all_features["oKitchenQual"] = all_features.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    all_features["oFunctional"] = all_features.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    all_features["oFireplaceQu"] = all_features.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    all_features["oGarageType"] = all_features.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    all_features["oGarageFinish"] = all_features.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    all_features["oPavedDrive"] = all_features.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    all_features["oSaleType"] = all_features.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    all_features["oSaleCondition"] = all_features.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    return "Done!"

print(map_values())
# drop two unwanted columns
all_features.drop("LotAreaCut",axis=1,inplace=True)
all_features.drop(['SalePrice'],axis=1,inplace=True)

# It's convenient to experiment different feature combinations once you've got a pipeline.

#编码三个有关Year的特征
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X


#使用log1p来正态化偏值较大的特征，并且向量化
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X


#将原始特征进行组合通常能产生意想不到的效果，然而这个数据集中原始特征有很多，不可能所有都一一组合，
# 所以这里先用Lasso进行特征筛选，选出较重要的一些特征进行组合。
# lasso=Lasso(alpha=0.001)
# lasso.fit(X_scaled,y_log)
# FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)
# FI_lasso.sort_values("Feature Importance",ascending=False)
# FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
# plt.xticks(rotation=90)
# plt.show()


#增加新特征
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional == 1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]

            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]

            return X

# 通过pipeline,可以更加快速的实验不同的特征组合
pipe = Pipeline([  ('labenc', labelenc()),
                   ('add_feature', add_feature(additional=2)),
                   ('skew_dummies', skew_dummies(skew=1)),])

#组合不同特征后保存处理好的数据
all_features2 = all_features.copy()
data_pipe = pipe.fit_transform(all_features2)
print(type(data_pipe),data_pipe.shape)
n_train=train_data.shape[0]
X = data_pipe[:n_train]
test_X = data_pipe[n_train:]
y= train_data.SalePrice
train_now=pd.concat([X,y],axis=1)
test_now=test_X
print('特征化后训练集形状：',train_now.shape,'特征化后测试集形状：',test_X.shape,'训练集的标签形状：',y.shape)
print('特征化后训练集前5行：')
print(train_now.head())
# 存储处理好的数据
# train_now.to_csv('house_prices/train_afterchange.csv')
# test_X.to_csv('house_prices/test_afterchange.csv')
# 读取处理好的数据
# train_X=pd.read_csv('house_prices/train_afterchange.csv')
# X=train_X[:-1]
# y=train_X.SalePrice
# test_X=pd.read_csv('house_prices/test_afterchange.csv')



#数据标准/归一化
scaler = RobustScaler()
X_scaled = scaler.fit(X).transform(X)
y_log = np.log(y)
test_X_scaled = scaler.transform(test_X)
print(type(X_scaled),X_scaled.shape)
print(type(test_X_scaled),test_X_scaled.shape)
print(y_log.shape)

#自己添加的特征都是高度相关的，所以使用PCA降低这些相关性
pca = PCA(n_components=426)
X_scaled=pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)

print(X_scaled.shape, test_X_scaled.shape)
print(X)




#5、模型创建和融合————————————————————————————————————————————————————————————————————————————————————————————




#首先定义RMSE的交叉验证评估指标：
# 定义一个评估标准来衡量模型的优劣
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

# #先粗劣找出几个表现较好的模型
# models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
#           ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
#           ExtraTreesRegressor(),XGBRegressor()]
# names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
# for name, model in zip(names, models):
#     score = rmse_cv(model, X_scaled, y_log)
#     print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))



def grid_get(model, X, y, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
    grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
    print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])

# #网格搜索找各个模型的最佳参数
# grid_get(Lasso(),X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})
# grid_get(Ridge(),X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
# grid_get(SVR(),X_scaled,y_log,{'C':[11,12,13,14,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
# param_grid={'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}
# grid_get(KernelRidge(),X_scaled,y_log,param_grid)
# grid_get(ElasticNet(),X_scaled,y_log,{'alpha':[0.0005,0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3,0.5,0.7],'max_iter':[10000]})

# #根据结果实例化最佳模型参数
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()


#使用Stacking进行模型融合
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean
# 直接使用mlxtend的StackingRegressor，但效果反而没有自定义的好
# stack_model =StackingRegressor(regressors=[lasso,ridge,svr,ker,ela,bay], meta_regressor=ker)
# print(rmse_cv(stack_model,X_scaled,y_log))
# print(rmse_cv(stack_model,X_scaled,y_log).mean())


a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
print('训练集误差',rmse_cv(stack_model,a,b))#训练集误差[0.10330339 0.10976285 0.11720582 0.09831198 0.10430719]
print('训练集误差均值',rmse_cv(stack_model,a,b).mean())#训练集上所有模型的平均误差 0.10657824413611261

#第二层：提取从stacking产生的特征，然后和原始特征合成在一起
X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)
# print(X_train_stack.shape, a.shape)# (1458, 6) (1458, 426)
X_train_add = np.hstack((a,X_train_stack))#数组拼接-在水平方向上平铺
X_test_add = np.hstack((test_X_scaled,X_test_stack))
# print(X_train_add.shape, X_test_add.shape)#(1458, 432) (1459, 432)
print('原数据集和特征组成的新数据集误差',rmse_cv(stack_model,X_train_add,b))# [0.09847682 0.10494807 0.11206736 0.09385517 0.09895766]
print('新数据集原数据集和特征组成的新数据集误差均值',rmse_cv(stack_model,X_train_add,b).mean()) #0.10166101589603005


# # #submit
# # # This is the final model I use
# # stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
# # stack_model.fit(a,b)
# # pred = np.exp(stack_model.predict(test_X_scaled))
# # result=pd.DataFrame({'Id':test_data.Id, 'SalePrice':pred})
# # result.to_csv("submission.csv",index=False)