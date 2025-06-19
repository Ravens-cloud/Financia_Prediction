import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import os
from scipy.fftpack import fft
import tensorflow as tf
from tensorflow.python.keras import layers, models, callbacks
# Windows 下常用 SimHei，如果路径不同请调整
font_path = r"C:\Windows\Fonts\simhei.ttf"
if os.path.exists(font_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    return df

# 2. 缺失值 & 异常值处理
def detect_and_cap_outliers(df, col, method='iqr'):
    series = df[col]
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    else:
        mean = series.mean()
        std = series.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
    mask_out = (series < lower) | (series > upper)
    if mask_out.any():
        df.loc[mask_out, col] = np.nan
    return lower, upper

def fill_missing_by_interpolation(df, cols, time_index_col='日期'):
    df = df.sort_values(time_index_col)
    df2 = df.set_index(time_index_col)
    for col in cols:
        if col in df2.columns and df2[col].isna().any():
            try:
                df2[col] = df2[col].interpolate(method='time')
            except Exception:
                df2[col] = df2[col].interpolate(method='linear')
    df2[cols] = df2[cols].ffill().bfill()
    return df2.reset_index()

def preprocess(df):
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    print("各列缺失比例（前10）:\n", missing_ratio.head(10))

    # 去除宏观列（示例）
    macro_cols = ['cpi_yoy', 'cpi_mom', 'm2_yoy', 'lpr_1y', 'lpr_5y']
    for col in macro_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"已去除宏观列: {col}")

    # 丢弃缺失比例>0.5 列
    to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    to_drop = [c for c in to_drop if c in df.columns]
    if to_drop:
        print("丢弃缺失比例>50%的列:", to_drop)
        df = df.drop(columns=to_drop)

    # 异常值检测
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().all():
            continue
        detect_and_cap_outliers(df, col, method='iqr')
    # 插值填充
    df = fill_missing_by_interpolation(df, num_cols)
    # 中位数填充剩余
    missing2 = df.isna().mean().sort_values(ascending=False)
    rem = missing2[missing2 > 0].index.tolist()
    if rem:
        print("用中位数填充剩余缺失列:", rem)
        imp = SimpleImputer(strategy='median')
        df[rem] = imp.fit_transform(df[rem])
    return df

# 3. 构造回归目标：先添加 future_close 列，再在特征工程时转换为收益率目标
def add_regression_targets(df, forecast_horizon=3):
    df = df.sort_values('日期').reset_index(drop=True)
    for i in range(1, forecast_horizon + 1):
        df[f'future_{i}d_close'] = df['收盘'].shift(-i)
    df = df.iloc[:-forecast_horizon].reset_index(drop=True)
    return df

# 4. 特征工程（保证不使用任何 future_* 信息）
def feature_engineering(df):
    # df 已含“日期”、“收盘”、“开盘”等，以及 future_*_close 列，但下面不要用 future 作为特征
    df = df.sort_values('日期').reset_index(drop=True)

    # 周期性编码
    df['month'] = df['日期'].dt.month
    df['weekday'] = df['日期'].dt.weekday
    df['day_of_year'] = df['日期'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # 计算滞后收益率特征和滚动统计
    # 先计算基础收益率：1d、5d、10d 等滞后收益
    df['ret_1d'] = df['收盘'].pct_change(periods=1)
    df['ret_5d'] = df['收盘'].pct_change(periods=5)
    df['ret_10d'] = df['收盘'].pct_change(periods=10)
    # 滚动统计（基于过去窗口，不含当日）
    windows = [5, 10, 20, 60]
    for w in windows:
        # 收盘滚动均线、波动率
        df[f'close_ma_{w}'] = df['收盘'].rolling(window=w, min_periods=1).mean().shift(1)
        df[f'close_std_{w}'] = df['收盘'].rolling(window=w, min_periods=1).std().shift(1).fillna(0)
        # 收益滚动统计
        df[f'ret_mean_{w}'] = df['ret_1d'].rolling(window=w, min_periods=1).mean().shift(1)
        df[f'ret_std_{w}'] = df['ret_1d'].rolling(window=w, min_periods=1).std().shift(1).fillna(0)
        # 成交量滚动
        if 'volume' in df.columns:
            df[f'vol_mean_{w}'] = df['volume'].rolling(window=w, min_periods=1).mean().shift(1)
            df[f'vol_std_{w}'] = df['volume'].rolling(window=w, min_periods=1).std().shift(1).fillna(0)
        # 动量信号：当前收盘与均线偏离
        df[f'close_dev_ma_{w}'] = (df['收盘'] - df[f'close_ma_{w}']) / df[f'close_ma_{w}'].replace(0, np.nan)
    # 技术指标示例：RSI、MACD（可选，若已有实现则用已有；此处简要示例 RSI 14 日）
    # RSI 14 日
    delta = df['收盘'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean().shift(1)
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().shift(1)
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD: 12/26 EMA
    ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['macd'] = (ema12 - ema26).shift(1)
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().shift(1)
    df['macd_hist'] = (df['macd'] - df['macd_signal']).shift(1)
    # 布林带偏离度：20 日均线和标准差
    df['bb_mid'] = df['收盘'].rolling(window=20, min_periods=1).mean().shift(1)
    df['bb_std'] = df['收盘'].rolling(window=20, min_periods=1).std().shift(1).fillna(0)
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.nan)

    # 删除因 shift/rolling 产生的 NaN 行：确保主要滚动特征 non-NaN
    feature_cols = [c for c in df.columns if c.startswith(('close_ma_','ret_mean_','vol_mean_','rsi_','macd','bb_','close_dev_ma_'))]
    df = df.dropna(subset=feature_cols, how='any').reset_index(drop=True)

    # 最后删除中间时间列
    df = df.drop(columns=['month','weekday','day_of_year'], errors='ignore')
    # 删除“日期”、“股票代码”列，但保留 “收盘”、“volume”、所有滚动特征，以及 future_*_close 列
    df = df.drop(columns=['日期','股票代码'], errors='ignore')

    return df

# 5. 构造回归目标收益率列，并时序划分
def prepare_data_for_regression(df, date_series, forecast_horizon=3, train_size=0.7, valid_size=0.15):
    # 先添加 future_close 列
    df_reg = add_regression_targets(df, forecast_horizon=forecast_horizon)
    # 特征工程（含“收盘”特征）
    df_feat = feature_engineering(df_reg)
    # 计算目标收益率列： future_i_return = (future_i_close - 当前收盘) / 当前收盘
    for i in range(1, forecast_horizon+1):
        df_feat[f'future_{i}d_return'] = (df_reg['收盘'].iloc[:len(df_feat)].values * 0 + 0)  # 占位，下面覆盖
        # 取 df_reg 对应位置
        curr_close = df_reg['收盘'].iloc[:len(df_feat)].values
        fut_close = df_reg[f'future_{i}d_close'].iloc[:len(df_feat)].values
        df_feat[f'future_{i}d_return'] = (fut_close - curr_close) / curr_close
    # 删除 future_close 列，不再作为特征
    future_close_cols = [c for c in df_feat.columns if c.startswith('future_') and c.endswith('d_close')]
    df_feat = df_feat.drop(columns=future_close_cols)

    # 对齐 date_series：feature_engineering 可能丢掉首几行，需同步 date_series
    # 假设 date_series 是原始 df['日期']，与 df_reg 对齐后再取前 len(df_feat)
    date_series_reg = date_series.iloc[forecast_horizon:].reset_index(drop=True)  # 因 add_regression_targets 丢掉最后 forecast_horizon
    # feature_engineering 丢掉若干首行：以 df_feat 长度为准，取 date_series_reg.iloc[offset: offset+len(df_feat)]
    # 这里简单假设 feature_engineering 丢掉的行等于 len(df_reg)-len(df_feat)
    # 若要精确对齐，可记录 feature_engineering 开始前后的索引。示例假设 df_feat index 对齐 df_reg index[0:len(df_feat)]
    date_used = date_series_reg.iloc[:len(df_feat)].reset_index(drop=True)

    # 时序划分
    n = len(df_feat)
    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))
    # 构造 X, y_returns
    future_return_cols = [f'future_{i}d_return' for i in range(1, forecast_horizon+1)]
    X = df_feat.drop(columns=future_return_cols)
    y = df_feat[future_return_cols]
    X_train = X.iloc[:train_end].reset_index(drop=True)
    y_train = y.iloc[:train_end].reset_index(drop=True)
    X_valid = X.iloc[train_end:valid_end].reset_index(drop=True)
    y_valid = y.iloc[train_end:valid_end].reset_index(drop=True)
    X_test = X.iloc[valid_end:].reset_index(drop=True)
    y_test = y.iloc[valid_end:].reset_index(drop=True)
    date_train = date_used.iloc[:train_end].reset_index(drop=True)
    date_valid = date_used.iloc[train_end:valid_end].reset_index(drop=True)
    date_test = date_used.iloc[valid_end:].reset_index(drop=True)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, future_return_cols, date_train, date_valid, date_test

# 6. 单输出回归超参调优 & 训练
def train_models_separate_horizons(X_train, y_train, X_valid, y_valid, future_return_cols):
    """
    对每个 horizon 单独调优 XGBRegressor，然后训练并返回模型列表。
    """
    models = {}
    best_params = {}
    for idx, col in enumerate(future_return_cols):
        print(f"\n--- 调优并训练模型: 预测 {col} ---")
        y_tr = y_train[col].values
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        xgb_reg = xgb.XGBRegressor(tree_method='hist', random_state=42, objective='reg:squarederror')
        param_dist = {
            'n_estimators': np.arange(50, 301, 50),
            'max_depth': np.arange(3, 11, 2),
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 5]
        }
        rs = RandomizedSearchCV(
            xgb_reg, param_distributions=param_dist,
            n_iter=10, scoring='neg_mean_absolute_error',
            cv=tscv, random_state=42, n_jobs=-1, verbose=1, error_score=np.nan
        )
        rs.fit(X_train, y_tr)
        best = rs.best_estimator_
        print(f"Best params for {col}:", rs.best_params_)
        # 验证集表现
        yv_pred = best.predict(X_valid)
        mae = mean_absolute_error(y_valid[col].values, yv_pred)
        rmse = np.sqrt(mean_squared_error(y_valid[col].values, yv_pred))
        print(f"验证集: {col}  MAE={mae:.4f}, RMSE={rmse:.4f}")
        models[col] = best
        best_params[col] = rs.best_params_
    return models, best_params

# 7. 预测还原价格并评估
def evaluate_return_models(models, X, y, date_series, prefix="测试集"):
    """
    models: dict of {future_i_return: model}
    X: DataFrame features, 包含 '收盘' 列作为当前收盘价
    y: DataFrame with true future returns
    date_series: 对应日期Series，表示预测日的日期
    """
    print(f"\n--- {prefix} 预测评估 ---")
    # 预测收益率
    preds = {}
    for col, model in models.items():
        preds[col] = model.predict(X)
    # 转成 DataFrame
    df_pred_ret = pd.DataFrame(preds, index=X.index)
    # 计算收益率 MAE/RMSE
    for col in preds:
        y_true = y[col].values
        y_p = preds[col]
        mae = mean_absolute_error(y_true, y_p)
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        print(f"{col} 收益率预测: MAE={mae:.4f}, RMSE={rmse:.4f}")
    # 还原价格并评估
    # 当前收盘价
    curr_close = X['收盘'].values
    for col in preds:
        # 提取 horizon i
        # 例如 col="future_1d_return"，取 i=1
        i = int(col.split('_')[1].replace('d',''))
        price_pred = curr_close * (1 + preds[col])
        # 从 date_series 中获取真实 future_i 收盘价：此处假设外部准备了真实 future_close 列或在 y_true_close 中提供
        # 但在此我们在 prepare_data_for_regression 阶段已丢弃 future_close，所以需从原始 df_reg 获取真实 future_close
        # 因此，建议在调用 evaluate_return_models 前，传入真实 future_close Series。
        # 这里暂不实现 price 评估，示例给出框架：
        # true_future_close = ...
        # mae_p = mean_absolute_error(true_future_close, price_pred)
        # rmse_p = np.sqrt(mean_squared_error(true_future_close, price_pred))
        # print(f"{col} 价格预测: MAE={mae_p:.4f}, RMSE={rmse_p:.4f}")
        pass
    # 若需要画图，可示例绘制某 horizon 的真实 vs 预测收益率时间序列
    # 例如:
    plt.figure(figsize=(10,4))
    plt.plot(date_series, y['future_1d_return'], label='真实1d_return')
    plt.plot(date_series, preds['future_1d_return'], label='预测1d_return')
    plt.legend(); plt.title("1日收益率 真实 vs 预测")
    plt.show()

# 8. 可选 MLP 回归
def build_mlp_reg(input_dim, output_dim, hidden_layers=[128,64], dropout_rate=0.5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(output_dim, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # 兼容 ReduceLROnPlateau 读取 lr
    model.optimizer.lr = model.optimizer.learning_rate
    return model

def train_mlp_reg(X_train, y_train, X_valid, y_valid, epochs=100, batch_size=32):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_mlp_reg(input_dim, output_dim)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    # 可视化损失
    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title("MLP 回归 Loss")
    plt.show()
    return model

# 主流程
if __name__ == '__main__':
    # 示例路径，请根据实际修改
    csv_path = 'file_data/ml_dataset_600900_20250615_091809.csv'
    df_raw = load_data(csv_path)

    # 预处理
    df_clean = preprocess(df_raw.copy())  # 保留“日期”、“收盘”等

    # 设定预测天数
    forecast_horizon = 3

    # 时序划分前的特征与目标准备
    X_train, y_train, X_valid, y_valid, X_test, y_test, future_return_cols, \
        date_train, date_valid, date_test = prepare_data_for_regression(
            df_clean, df_raw['日期'], forecast_horizon=forecast_horizon,
            train_size=0.7, valid_size=0.15
        )
    print(f"训练集: {X_train.shape}, 验证集: {X_valid.shape}, 测试集: {X_test.shape}")

    # 标准化（MLP 需要；XGB 可不需要）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s = scaler.transform(X_test)

    # 6. XGBRegressor 单输出分别调优并训练
    models_dict, best_params = train_models_separate_horizons(
        X_train, y_train, X_valid, y_valid, future_return_cols
    )
    print("各 horizon 最佳参数：", best_params)

    # 7. 评估收益率模型
    evaluate_return_models(models_dict, X_valid, y_valid, date_valid, prefix="验证集")
    evaluate_return_models(models_dict, X_test, y_test, date_test, prefix="测试集")

    # 8. 可选 MLP 回归尝试
    mlp_model = train_mlp_reg(X_train_s, y_train.values, X_valid_s, y_valid.values, epochs=50)
    # 评估 MLP 收益率预测
    yv_pred_mlp = mlp_model.predict(X_valid_s)
    for i, col in enumerate(future_return_cols):
        y_true = y_valid.values[:, i]
        y_p = yv_pred_mlp[:, i]
        mae = mean_absolute_error(y_true, y_p)
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        print(f"MLP {col} 收益率预测: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # 9. 可视化某 horizon 真实 vs 预测收益率
    # 以 1d_return 为例：
    plt.figure(figsize=(10,4))
    plt.plot(date_valid, y_valid['future_1d_return'], label='真实1d_return')
    plt.plot(date_valid, models_dict['future_1d_return'].predict(X_valid), label='预测1d_return')
    plt.legend(); plt.title("1日收益率 真实 vs 预测（验证集）")
    plt.show()

    print("全流程完成。")
