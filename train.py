import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import datetime
import warnings
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import sklearn
import gc

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

def kalman_smooth(x):

    series = [x['sales_0'], x['sales_1'], x['sales_2'], x['sales_3'], x['sales_4'],
                      x['sales_5'], x['sales_6'], x['sales_7'], x['sales_8'], x['sales_9'], x['sales_10'], x['sales_11'],
                      x['sales_12'], x['sales_13'], x['sales_14']]
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=1, initial_state_mean=series[0])
    state_means, state_covariance = kf.smooth(series)
    return state_means.ravel().tolist()

def train_model():
    #读取商品goodsale_modified1.csv，商品销售调整过后的csv文件
    df = pd.read_csv('fusai_data/goodsale_modified1.csv')
    #将时间数据转换为datatime格式
    df['data_date'] = pd.to_datetime(df['data_date'], format='%Y-%m-%d')
    # 数据转换
    # 将逗号转换为空格
    df['goods_price'] = df['goods_price'].map(lambda x: x.replace(',', '') if type(x) == np.str else x)
    # 将商品价格转换为数字
    df['goods_price'] = pd.to_numeric(df['goods_price'])
    # 将逗号转换为空格
    df['orginal_shop_price'] = df['orginal_shop_price'].map(lambda x: x.replace(',', '') if type(x) == np.str else x)
    # 将商品吊牌价格转换为数字
    df['orginal_shop_price'] = pd.to_numeric(df['orginal_shop_price'])

    #读取提交结果样本csv文件
    sub = pd.read_csv('fusai_data/submit_example_2.csv')
    #读取商品信息文件
    info = pd.read_csv('fusai_data/goodsinfo.csv')
    #读取商品sku映射表
    relation = pd.read_csv('fusai_data/goods_sku_relation.csv')
    #拼接商品信息和sku_id
    relation = pd.merge(relation, info, on='goods_id')


    #读取商品daily_modified1文件，调整后的商品在用户的表现数据
    daily = pd.read_csv('fusai_data/daily_modified1.csv')
    #数据转换
        #将时间数据转换为data_date格式
    daily['data_date'] = pd.to_datetime(daily['data_date'], format='%Y-%m-%d')
        #去除重复数据goods_id
    droped = daily.drop_duplicates(subset='goods_id')
        #得到商品时间
    droped['open_date'] = droped.apply(lambda x: x['data_date'] - datetime.timedelta(x['onsale_days']), axis=1)



    #按sku_id和own_week分组并求和,也就是算出每个sku的总销售量，并重构索引
    grouped = df.groupby(['sku_id', 'own_week'])['goods_num'].sum().reset_index()
    #重构数据为sku每周销售量，此处索引为sku_id，列组织为owen_week
    pivot = grouped.pivot(index='sku_id', columns='own_week', values='goods_num')
    new_columns = {}
    for i in list(pivot.columns):
        new_columns[i] = 'sales_' + str(i)
    #pivot表进行列的重命名，格式为sales_xxx,即sku_id每周的销售量
    pivot.rename(columns=new_columns, inplace=True)
    #在空值处填充0
    pivot.fillna(0, inplace=True)

    #按goods_id和own_week分组并对点击量求和，并重置索引
    grouped_daily = daily.groupby(['goods_id', 'own_week'])['goods_click'].sum().reset_index()
    #重构数据为商品每周点击次数，索引为goods_id，列组织为own_week,值为商品点击量
    pivot_daily = grouped_daily.pivot(index='goods_id', columns='own_week', values='goods_click')
    new_columns = {}
    for i in list(pivot_daily.columns):
        new_columns[i] = 'goods_click_' + str(i)
    #pivot_daily表进行列的重命名，格式为sales_xxx,即goods每周的点击量
    pivot_daily.rename(columns=new_columns, inplace=True)
    # 填充缺失值
    pivot_daily.fillna(0, inplace=True)

    #按goods_id和own_week分组并对加购次数求和并重构索引
    grouped_daily_cart = daily.groupby(['goods_id', 'own_week'])['cart_click'].sum().reset_index()
    #重构数据为商品每周加购次数，索引为商品id,列组织为周，值是加购次数
    pivot_daily_cart = grouped_daily_cart.pivot(index='goods_id', columns='own_week', values='cart_click')
    new_columns = {}
    for i in list(pivot_daily_cart.columns):
        new_columns[i] = 'cart_click_' + str(i)
    #pivot_daily_cart进行列的重命名，格式为cart_click_xxx,即商品每周的加购次数
    pivot_daily_cart.rename(columns=new_columns, inplace=True)
    # 填充缺失值
    pivot_daily_cart.fillna(0, inplace=True)

    #按goods_id和own_week分组并对收藏次数求和并重构索引
    grouped_daily_fav = daily.groupby(['goods_id', 'own_week'])['favorites_click'].sum().reset_index()
    #重构数据为商品每周的收藏次数，索引为商品id,列组织为周，值是收藏次数
    pivot_daily_fav = grouped_daily_fav.pivot(index='goods_id', columns='own_week', values='favorites_click')
    new_columns = {}
    for i in list(pivot_daily_fav.columns):
        new_columns[i] = 'favorites_click_' + str(i)
    #pivot_daily_cart进行列的重命名，格式为favorities_click_xxx,即商品每周的收藏次数
    pivot_daily_fav.rename(columns=new_columns, inplace=True)
    # 填充缺失值
    pivot_daily_fav.fillna(0, inplace=True)

    # 按goods_id和own_week分组并对商品购买求和并重构索引
    grouped_daily_uv = daily.groupby(['goods_id', 'own_week'])['sales_uv'].sum().reset_index()
    # 重构数据为商品每周的购买人数，索引是商品id，列是周，值是商品购买人数
    pivot_daily_uv = grouped_daily_uv.pivot(index='goods_id', columns='own_week', values='sales_uv')
    new_columns = {}
    for i in list(pivot_daily_uv.columns):
        new_columns[i] = 'sales_uv_' + str(i)
    #修改索引
    pivot_daily_uv.rename(columns=new_columns, inplace=True)
    #填充缺失值
    pivot_daily_uv.fillna(0, inplace=True)

    #合并表格，即sku_id,商品信息，点击人数，加购人数，收藏人数，购买人数，开售时间，在售时长，平均吊牌价，平均销售价格，折扣价格
    sub = pd.merge(sub, pivot, on='sku_id', how='left')
    sub = pd.merge(sub, relation, on='sku_id', how='left')
    sub = pd.merge(sub, pivot_daily, on='goods_id', how='left')
    sub = pd.merge(sub, pivot_daily_cart, on='goods_id', how='left')
    sub = pd.merge(sub, pivot_daily_fav, on='goods_id', how='left')
    sub = pd.merge(sub, pivot_daily_uv, on='goods_id', how='left')
    sub = pd.merge(sub, droped[['goods_id', 'open_date']], on='goods_id', how='left')
    sub['onsale_train'] = sub['open_date'].map(lambda x: (datetime.datetime(2018, 3, 16) - x).days)
    sub['onsale_test'] = sub['open_date'].map(lambda x: (datetime.datetime(2018, 5, 7) - x).days)
    #合并商品的类目信息
    sub['concat'] = sub.apply(lambda x: str(x['cat_level1_id']) +
                                        '_' + str(x['cat_level2_id']) + '_' + str(x['cat_level3_id'])
                                        + '_' + str(x['cat_level4_id']) + '_' + str(x['cat_level5_id']), axis=1)

    #按sku分组对吊牌价求平均并重构索引，即得到供应链商品的平均吊牌价格
    raw_price = df.groupby('sku_id')['orginal_shop_price'].mean().reset_index()
    #按sku分组对售价求平均并重构索引，即得到供应链商品的平均销售价格
    real_price = df.groupby('sku_id')['goods_price'].mean().reset_index()

    #将平均吊牌价和平均销售价格合并到表格
    sub = pd.merge(sub, raw_price, on='sku_id', how='left')
    sub = pd.merge(sub, real_price, on='sku_id', how='left')
    #生成折扣
    sub['discount'] = sub['orginal_shop_price'] - sub['goods_price']

    print('------------load_data-----------------')
    #进行卡尔曼平滑

    sub['smooth'] = sub.apply(lambda x: kalman_smooth(x), axis=1)
    for i in range(15):
        sub['sales_smo_'+str(i)] = sub.apply(lambda x: x['smooth'][i], axis=1)
    print('------------kalman smooth-----------------')
    #赋予不同的周的销售总量不同的权重
    sub['sales_8'] = sub['sales_8'] * 1.1
    sub['sales_9'] = sub['sales_9'] * 1.2
    sub['sales_10'] = sub['sales_10'] * 1
    sub['sales_11'] = sub['sales_11'] * 0.6
    sub['sales_12'] = sub['sales_12'] * 0.7
    sub['sales_13'] = sub['sales_13'] * 0.8
    sub['sales_14'] = sub['sales_14'] * 0.9



    #训练用到的特征：sku_id，商品id，品牌id，商品季节属性，一级类目id，总类目信息,吊牌价格，商品实际价格，折扣，在售时间
    trian_features = ['sku_id', 'goods_id', 'brand_id', 'goods_season', 'cat_level1_id', 'concat',  'orginal_shop_price', 'goods_price', 'discount']
    trian_features.append('onsale_train')
    #测试用到的特征：sku_id，商品id，品牌id，一级类目id，concat,商品季节属性，吊牌价格，商品实际价格，折扣，在售时间
    test_features = ['sku_id', 'goods_id', 'brand_id', 'cat_level1_id', 'concat', 'goods_season',  'orginal_shop_price', 'goods_price', 'discount']
    test_features.append('onsale_test')
    #test数据集获取0到7周的销售数据，平滑后的销售数据，商品点击数据，加购数据，收藏数据，购买人数数据
    for i in range(0, 8):
        test_features.append('sales_' + str(i))
        test_features.append('sales_smo_'+str(i))
        test_features.append('goods_click_' + str(i))
        test_features.append('cart_click_' + str(i))
        test_features.append('favorites_click_' + str(i))
        test_features.append('sales_uv_'+str(i))
    #训练数据集获取7到14周，商品销售数据，平滑后的销售数据，商品点击数据，加购数据，收藏数据，购买人群数据
    for i in range(7, 15):
        trian_features.append('sales_' + str(i))
        trian_features.append('sales_smo_'+str(i))
        trian_features.append('goods_click_' + str(i))
        trian_features.append('cart_click_' + str(i))
        trian_features.append('favorites_click_' + str(i))
        trian_features.append('sales_uv_'+str(i))
    #获取训练集和测试集
    X_train = sub[trian_features]
    y_train = sub[['sales_0', 'sku_id', 'goods_id']]
    X_test = sub[test_features]

    X_train['onsale'] = X_train['onsale_train']
    X_test['onsale'] = X_test['onsale_test']



    X_train['sales_win_0'] = 0
    X_test['sales_win_0'] = 0
    X_train['click_win_0'] = 0
    X_test['click_win_0'] = 0
    X_train['cart_win_0'] = 0
    X_test['cart_win_0'] = 0
    X_train['favorites_win_0'] = 0
    X_test['favorites_win_0'] = 0
    X_train['uv_win_0'] = 0
    X_test['uv_win_0'] = 0
    all_features = ['guize', 'mean_sale', 'median_sale', 'goods_price', 'discount', 'onsale']

    guize_type = 'sales_smo_'
    #对平滑后的每周销售量加权求和
    X_train['guize'] = (13*X_train[guize_type+'7'] + 7*X_train[guize_type+'8'] + 6*X_train[guize_type+'9'] + 5*X_train[guize_type+'10']+
                        4*X_train[guize_type+'11']+ 3*X_train[guize_type+'12']+2*X_train[guize_type+'12']+X_train[guize_type+'14'])/41
    X_test['guize'] = (13 * X_test[guize_type+'0'] + 7 * X_test[guize_type+'1'] + 6 * X_test[guize_type+'2'] + 5 * X_test[
        guize_type+'3'] +4 * X_test[guize_type+'4'] + 3 * X_test[guize_type+'5'] + 2 * X_test[guize_type+'6'] + X_test[guize_type+'7'])/41

    sales_type = 'sales_'
    #对销售量求均值
    #DataFrame中的apply方法就是将函数应用到由列或行形成的一维数组上，求均值
    X_train['mean_sale'] = X_train.apply(
        lambda x: np.mean([x[sales_type+'7'], x[sales_type+'8'], x[sales_type+'9'], x[sales_type+'10'], x[sales_type+'11'],
                      x[sales_type+'12'], x[sales_type+'13'], x[sales_type+'14']]), axis=1)
    X_test['mean_sale'] = X_test.apply(
        lambda x: np.mean([x[sales_type+'0'], x[sales_type+'1'], x[sales_type+'2'], x[sales_type+'3'], x[sales_type+'4'],
                      x[sales_type+'5'], x[sales_type+'6'], x[sales_type+'7']]), axis=1)
    #求销售量中位数
    X_train['median_sale'] = X_train.apply(
        lambda x: np.median([x[sales_type+'7'], x[sales_type+'8'], x[sales_type+'9'], x[sales_type+'10'], x[sales_type+'11'],
                      x[sales_type+'12'], x[sales_type+'13'], x[sales_type+'14']]), axis=1)
    X_test['median_sale'] = X_test.apply(
        lambda x: np.median([x[sales_type+'0'], x[sales_type+'1'], x[sales_type+'2'], x[sales_type+'3'], x[sales_type+'4'],
                      x[sales_type+'5'], x[sales_type+'6'], x[sales_type+'7']]), axis=1)

    #大量构造叠加特征合值特征
    for i in range(1, 9):
        X_train['sales_win_' + str(i)] = X_train['sales_' + str(i + 6)] + X_train['sales_win_' + str(i - 1)]
        X_train['click_win_' + str(i)] = X_train['goods_click_' + str(i + 6)]
        X_train['cart_win_' + str(i)] = X_train['cart_click_' + str(i + 6)] + X_train['cart_win_' + str(i - 1)]
        X_train['favorites_win_' + str(i)] = X_train['favorites_click_' + str(i + 6)] + X_train['favorites_win_' + str(i - 1)]
        X_train['uv_win_' + str(i)] = X_train['sales_uv_' + str(i + 6)] + X_train['uv_win_' + str(i - 1)]
        #每周销售量/点击量
        X_train['sales/click_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['click_win_' + str(i)]
        #按商品求得的每周销售量和每周点击量
        X_train = X_train.join(X_train.groupby('goods_id')['sales_win_' + str(i)].sum().rename('goods_sales_win_' + str(i)), on='goods_id')
        X_train = X_train.join(X_train.groupby('goods_id')['click_win_' + str(i)].sum().rename('goods_click_win_' + str(i)), on='goods_id')
        #按商品一级类目求得的销售总量，按总类目求得的商品销售总量和平均销售量
        X_train = X_train.join(X_train.groupby('cat_level1_id')['sales_win_' + str(i)].sum().rename('cat1_sales_win_' + str(i)), on='cat_level1_id')
        X_train = X_train.join(X_train.groupby('concat')['sales_win_' + str(i)].sum().rename('concat_sales_win_' + str(i)), on='concat')
        X_train = X_train.join(X_train.groupby('concat')['sales_win_' + str(i)].mean().rename('concat_sales_win_' + str(i)+'_mean'), on='concat')
        #按品牌求得商品销售总量
        X_train = X_train.join(X_train.groupby('brand_id')['sales_win_' + str(i)].sum().rename('brand_sales_win_' + str(i)), on='brand_id')

        #每周总销售量/每周商品销售总量，
        X_train['goods/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['goods_sales_win_' + str(i)]
        X_train['brand/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['brand_sales_win_' + str(i)]

        X_train['goods_click/sku_win' + str(i)] = X_train['click_win_' + str(i)] / X_train['goods_click_win_' + str(i)]

        X_train['cat1/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['cat1_sales_win_' + str(i)]
        X_train['concat/sku_win' + str(i)] = X_train['sales_win_' + str(i)] / X_train['concat_sales_win_' + str(i)]
        X_train['concat-sku_win' + str(i)] = X_train['sales_win_' + str(i)] - X_train['concat_sales_win_' + str(i)+'_mean']


        X_train = X_train.join(X_train.groupby('concat')['sales_win_' + str(i)].rank().rename('concat/sku_rank_win' + str(i)))


        print(X_train.head())
        all_features.append('sales_win_' + str(i))


        all_features.append('goods/sku_win' + str(i))

        all_features.append('concat/sku_win'+str(i))



    for i in range(1, 9):
        X_test['sales_win_' + str(i)] = X_test['sales_' + str(i - 1)] + X_test['sales_win_' + str(i - 1)]

        X_test['click_win_' + str(i)] = X_test['goods_click_' + str(i - 1)]
        X_test['cart_win_' + str(i)] = X_test['cart_click_' + str(i - 1)] + X_test['cart_win_' + str(i - 1)]
        X_test['favorites_win_' + str(i)] = X_test['favorites_click_' + str(i - 1)] + X_test['favorites_win_' + str(i - 1)]
        X_test['uv_win_' + str(i)] = X_test['sales_uv_' + str(i - 1)] + X_test['uv_win_' + str(i - 1)]

        X_test['sales/click_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['click_win_' + str(i)]
        X_test = X_test.join(X_test.groupby('goods_id')['sales_win_' + str(i)].sum().rename('goods_sales_win_' + str(i)), on='goods_id')
        X_test = X_test.join(X_test.groupby('goods_id')['click_win_' + str(i)].sum().rename('goods_click_win_' + str(i)), on='goods_id')

        X_test = X_test.join(X_test.groupby('cat_level1_id')['sales_win_' + str(i)].sum().rename('cat1_sales_win_' + str(i)), on='cat_level1_id')
        X_test = X_test.join(X_test.groupby('concat')['sales_win_' + str(i)].sum().rename('concat_sales_win_' + str(i)), on='concat')
        X_test = X_test.join(X_test.groupby('brand_id')['sales_win_' + str(i)].sum().rename('brand_sales_win_' + str(i)), on='brand_id')
        X_test = X_test.join(X_test.groupby('concat')['sales_win_' + str(i)].mean().rename('concat_sales_win_' + str(i)+'_mean'), on='concat')

        X_test['goods/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['goods_sales_win_' + str(i)]
        X_test['goods_click/sku_win' + str(i)] = X_test['click_win_' + str(i)] / X_test['goods_click_win_' + str(i)]
        X_test['brand/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['brand_sales_win_' + str(i)]

        X_test['cat1/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['cat1_sales_win_' + str(i)]
        X_test['concat/sku_win' + str(i)] = X_test['sales_win_' + str(i)] / X_test['concat_sales_win_' + str(i)]
        X_test['concat-sku_win' + str(i)] = X_test['sales_win_' + str(i)] - X_test['concat_sales_win_' + str(i)+'_mean']

        X_test = X_test.join(X_test.groupby('concat')['sales_win_' + str(i)].rank().rename('concat/sku_rank_win' + str(i)))


    print('------------generate features-----------------')
    # xgb模型#随机种子为666，用70棵树来拟合，即迭代七十次，线程数量为4
    # clf = xgb.XGBRegressor(random_state=666, n_estimators=70, silent=False, n_jobs=4)

    # 随机种子为666，用20棵树来拟合，即迭代七十次，线程数量为4
    # lgb模型clf = lgb.LGBMClassifier(random_state=666, n_estimators=20, silent=False, n_jobs=4)

    #cat模型
    clf = cat.CatBoostRegressor(random_state=666, n_estimators=20, silent=False)
    # 数据传入
    clf.fit(X_train[all_features], y_train['sales_0'])
    # 列出所有特征以及其对应特征重要指数
    print("look here:\n", pd.Series(clf.feature_importances_, all_features))
    # 求得预测值
    y_pred = clf.predict(X_test[all_features])

    #第一季度权值为0.9，第二季度权值为0.9，第三季度权值为0.9，第四季度权值为0.5
    season_x = X_test['goods_season'].map(lambda x: 0.5 if x == 4 else 1)
    y_pred = y_pred * season_x
    season_x = X_test['goods_season'].map(lambda x: 0.9 if x == 2 else 1)
    y_pred = y_pred * season_x
    season_x = X_test['goods_season'].map(lambda x: 0.9 if x == 1 else 1)
    y_pred = y_pred * season_x
    season_x = X_test['goods_season'].map(lambda x: 0.9 if x == 3 else 1)
    y_pred = y_pred * season_x

    sub['week3'] = y_pred * 1.6

    sub['week1'] = sub['week3'].map(lambda x: (x / 1.6) * 1)
    sub['week2'] = sub['week3'].map(lambda x: (x / 1.6) * 1.3)
    sub['week3'] = sub['week3'].map(lambda x: (x / 1.6) * 1.7)
    sub['week4'] = sub['week3'].map(lambda x: (x / 1.6) * 2.1)
    sub['week5'] = sub['week3'].map(lambda x: (x / 1.6) * 0.7)

    print('------------predict-----------------')

    sub[['sku_id', 'week1', 'week2', 'week3', 'week4', 'week5']].to_csv('fusai_data/xgb.csv', index=False)
    print('------------ok!!!!!!!!!!!-----------------')

def pre_process():
    #获取商品销售信息
    sale = pd.read_csv('fusai_data/goodsale.csv')
    #获取提交格式
    sub = pd.read_csv('fusai_data/submit_example_2.csv')
    #获取商品在用户的表现信息，如点击次数，收藏次数，加购次数，购买人数，在售天数等信息
    daily = pd.read_csv('fusai_data/goodsdaily.csv')
    #获取商品与sku_id的对应关系，sku即供应链
    relation = pd.read_csv('fusai_data/goods_sku_relation.csv')
    #将商品销售数据中的时间信息数据格式转换为data_date格式
    sale['data_date'] = pd.to_datetime(sale['data_date'], format='%Y%m%d')
    #生成商品数据距离预测起始点的时间间隔
    sale['own_week'] = sale['data_date'].map(lambda x: (datetime.datetime(2018, 3, 16)-x).days//7)
    #生成调整后的商品销售信息
    sale.to_csv('fusai_data/goodsale_modified1.csv', index=False)
    print('-----------------生成sale数据ok----------------')
    #将商品与sku对应表格与提交表格做拼接
    sub = pd.merge(sub, relation, on='sku_id', how='left')
    #在商品在用户的表现数据中剔除不需要的商品信息
    part = daily[daily['goods_id'].isin(sub['goods_id'].unique())]
    #将时间数据转换为data_date格式
    part['data_date'] = pd.to_datetime(part['data_date'], format='%Y%m%d')
    #生成商品数据距离预测起始点的时间间隔
    part['own_week'] = part['data_date'].map(lambda x: (datetime.datetime(2018, 3, 16) - x).days//7)
    #保存为调整后的商品在用户的表现数据
    part.to_csv('fusai_data/daily_modified1.csv', index=False)
    print('-----------------生成daily数据ok----------------')


if __name__ == '__main__':
    #数据预处理
    pre_process()
    #模型训练
    train_model()
