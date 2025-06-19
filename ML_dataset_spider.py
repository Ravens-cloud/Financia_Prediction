import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import numpy as np
import akshare as ak
import jieba
import jieba.analyse
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
# It's good practice to manage file paths and other constants in one place.
DATA_DIR = 'file_data'
# For reproducibility, you might want to set a seed
random.seed(42)
np.random.seed(42)


class FinancialDataEngine:
    """
    An advanced data engine to fetch, process, and engineer features from various
    financial sources for stock market analysis and machine learning.
    """

    def __init__(self, user_agent=None, timeout=20):
        """Initializes the data engine."""
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        self.timeout = timeout
        self.data_dir = DATA_DIR
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        jieba.initialize()
        print("FinancialDataEngine initialized.")

    def _normalize_date_str(self, date_str):
        """Converts various date string formats to 'YYYYMMDD'."""
        if isinstance(date_str, datetime):
            return date_str.strftime("%Y%m%d")
        try:
            return pd.to_datetime(date_str).strftime("%Y%m%d")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date format: {date_str}. Error: {e}")

    def save_data(self, data, filename_prefix, file_format='json'):
        """Saves data to a timestamped file in the designated data directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.data_dir, f"{filename_prefix}_{timestamp}.{file_format}")

        try:
            if isinstance(data, pd.DataFrame):
                df = data.copy()
                # Convert datetime columns to string for serialization
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, Asia/Shanghai]']).columns:
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

                if file_format == 'json':
                    records = df.to_dict(orient='records')
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(records, f, ensure_ascii=False, indent=4, allow_nan=True)
                elif file_format == 'csv':
                    df.to_csv(filepath, index=False, encoding='utf-8-sig')
                else:
                    raise ValueError(f"Unsupported file format for DataFrame: {file_format}")
            else:
                if file_format == 'json':
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4, allow_nan=True)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(str(data))

            print(f"Data successfully saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Failed to save data to {filepath}. Error: {e}")
            return None

    # --- Section 1: Core Data Fetching ---

    def get_stock_data(self, stock_code, start_date, end_date, adjust='qfq'):
        """Fetches historical stock price data with robust error handling."""
        print(f"Fetching stock data for {stock_code} from {start_date} to {end_date}...")
        try:
            sd = self._normalize_date_str(start_date)
            ed = self._normalize_date_str(end_date)
            df = ak.stock_zh_a_hist(symbol=stock_code, start_date=sd, end_date=ed, adjust=adjust)
            if df is None or df.empty:
                print(f"Warning: No stock data found for {stock_code} in the given period.")
                return None
            df['日期'] = pd.to_datetime(df['日期'])
            # Ensure standard column names
            df = df.rename(columns={'成交量': 'volume', '成交额': 'amount'})
            return df
        except Exception as e:
            print(f"Error fetching stock data for {stock_code}: {e}")
            return None

    def get_market_index_data(self, index_code, start_date, end_date):
        """Fetches historical market index data with robust error handling."""
        print(f"Fetching index data for {index_code}...")
        try:
            # 尝试多种指数接口
            try:
                df = ak.stock_zh_index_daily(symbol=index_code)
                rename_map = {
                    'date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'volume': '成交量'
                }
            except:
                # 备用接口
                df = ak.index_zh_a_hist(symbol=index_code, period="daily")
                rename_map = {
                    '日期': '日期',
                    '开盘': '开盘',
                    '收盘': '收盘',
                    '最高': '最高',
                    '最低': '最低',
                    '成交量': '成交量'
                }

            if df is None or df.empty:
                print(f"Warning: No data found for index {index_code}.")
                return None

            # 标准化列名
            df = df.rename(columns=rename_map)

            # 确保包含所有必要列
            required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # 转换日期格式
            df['日期'] = pd.to_datetime(df['日期'])
            df = df[(df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))]

            # 添加前缀避免列名冲突
            df = df.add_prefix(f"index_{index_code}_")
            df = df.rename(columns={f"index_{index_code}_日期": "日期"})

            return df
        except Exception as e:
            print(f"Error fetching index data for {index_code}: {e}")
            return None

    def get_financial_reports(self, stock_code):
        """Fetches key financial report indicators with robust error handling."""
        print(f"Fetching financial reports for {stock_code}...")
        try:
            # 获取当前年份作为起始查询点
            current_year = datetime.now().year
            start_year = str(current_year - 5)  # 获取最近5年的数据

            # 使用改进的财务指标接口
            df = ak.stock_financial_analysis_indicator(
                symbol=stock_code,
                start_year=start_year
            )

            if df is None or df.empty:
                print(f"Warning: No financial reports found for {stock_code}.")
                return None

            # 标准化日期格式
            df = df.rename(columns={'日期': 'report_date'})
            df['report_date'] = pd.to_datetime(df['report_date'])

            # 选择关键财务指标
            essential_cols = [
                'report_date',
                '净资产收益率(%)',
                '摊薄每股收益(元)',
                '总股本(万股)',
                '净利润(万元)',
                '资产负债率(%)'
            ]

            # 确保所有必要列都存在
            available_cols = [col for col in essential_cols if col in df.columns]
            if len(available_cols) < 3:  # 至少需要日期和2个指标
                print(f"Warning: Insufficient financial columns for {stock_code}")
                return None

            df = df[available_cols]

            # 重命名列并转换单位
            rename_map = {
                '净资产收益率(%)': 'roe',
                '摊薄每股收益(元)': 'eps',
                '总股本(万股)': 'total_shares',
                '净利润(万元)': 'net_profit',
                '资产负债率(%)': 'debt_ratio'
            }
            df = df.rename(columns=rename_map)

            # 转换单位：万股→亿股，万元→元
            if 'total_shares' in df.columns:
                df['total_shares'] = df['total_shares'] / 10000  # 转换为亿股
            if 'net_profit' in df.columns:
                df['net_profit'] = df['net_profit'] * 10000  # 转换为元

            # 清理数据
            df = df.dropna(subset=['report_date'])
            df = df.sort_values('report_date', ascending=False)

            return df
        except Exception as e:
            print(f"Error fetching financial reports for {stock_code}: {e}")
            return None

    def get_macro_economic_indicators(self, start_date, end_date):
        """获取宏观经济指标，增强日期解析能力"""
        print("Fetching macroeconomic indicators...")
        try:
            # 获取CPI数据并处理日期格式
            cpi_df = ak.macro_china_cpi()

            # 改进的日期处理逻辑
            # 处理多种可能的日期格式：2025年05月份 -> 2025-05
            cpi_df['月份'] = cpi_df['月份'].str.replace(r'(\d{4})年(\d{1,2})月份?', r'\1-\2', regex=True)
            cpi_df['月份'] = cpi_df['月份'].str.replace('份', '')  # 移除可能残留的"份"

            # 尝试解析日期
            cpi_df['date'] = pd.to_datetime(cpi_df['月份'], format='%Y-%m', errors='coerce')

            # 处理转换失败的日期
            if cpi_df['date'].isnull().any():
                print("Some CPI dates failed to parse, using fallback method")
                # 备选方案：提取年份和月份数字
                cpi_df['year'] = cpi_df['月份'].str.extract(r'(\d{4})')
                cpi_df['month'] = cpi_df['月份'].str.extract(r'-(\d{1,2})$')
                # 确保月份是两位数
                cpi_df['month'] = cpi_df['month'].apply(lambda x: x.zfill(2) if pd.notnull(x) and len(x) == 1 else x)
                cpi_df['date'] = pd.to_datetime(
                    cpi_df['year'] + '-' + cpi_df['month'],
                    format='%Y-%m',
                    errors='coerce'
                )

            # 检查列是否存在
            cpi_col_map = {}
            if '全国-同比增长' in cpi_df.columns:
                cpi_col_map['全国-同比增长'] = 'cpi_yoy'
            elif '全国-同比' in cpi_df.columns:  # 备用列名
                cpi_col_map['全国-同比'] = 'cpi_yoy'

            if '全国-环比增长' in cpi_df.columns:
                cpi_col_map['全国-环比增长'] = 'cpi_mom'
            elif '全国-环比' in cpi_df.columns:  # 备用列名
                cpi_col_map['全国-环比'] = 'cpi_mom'

            cpi_df = cpi_df.rename(columns=cpi_col_map)

            # 只保留存在的列
            keep_cols = ['date']
            if 'cpi_yoy' in cpi_df.columns:
                keep_cols.append('cpi_yoy')
            if 'cpi_mom' in cpi_df.columns:
                keep_cols.append('cpi_mom')

            cpi_df = cpi_df[keep_cols].dropna(subset=['date'])

            # 获取M2数据
            m2_df = ak.macro_china_money_supply()
            m2_df['月份'] = m2_df['月份'].str.replace(r'(\d{4})年(\d{1,2})月份?', r'\1-\2', regex=True)
            m2_df['月份'] = m2_df['月份'].str.replace('份', '')  # 移除可能残留的"份"

            # 尝试解析日期
            m2_df['date'] = pd.to_datetime(m2_df['月份'], format='%Y-%m', errors='coerce')
            m2_df = m2_df.rename(columns={'货币和准货币(M2)-同比增长': 'm2_yoy'})
            m2_df = m2_df[['date', 'm2_yoy']]

            # 获取LPR数据
            lpr_df = ak.macro_china_lpr()
            lpr_df = lpr_df.rename(columns={'TRADE_DATE': 'date', 'LPR1Y': 'lpr_1y', 'LPR5Y': 'lpr_5y'})
            lpr_df['date'] = pd.to_datetime(lpr_df['date'])
            lpr_df = lpr_df[['date', 'lpr_1y', 'lpr_5y']]

            # 合并所有宏观经济数据
            macro_df = pd.merge(cpi_df, m2_df, on='date', how='outer')
            macro_df = pd.merge(macro_df, lpr_df, on='date', how='outer')
            macro_df = macro_df.sort_values('date')

            # 过滤日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            macro_df = macro_df[(macro_df['date'] >= start_dt) & (macro_df['date'] <= end_dt)]
            macro_df = macro_df.rename(columns={'date': '日期'})

            return macro_df
        except Exception as e:
            print(f"Error fetching macroeconomic indicators: {e}")
            return pd.DataFrame()

    def get_stock_industry_info(self, stock_code):
        """Gets the industry name and code with robust fallback mechanism."""
        print(f"Fetching industry info for {stock_code}...")
        try:
            # 方法1：使用基本股票信息
            info_df = ak.stock_individual_info_em(symbol=stock_code)

            # 尝试不同的行业字段名称
            industry_fields = ['所属行业', '行业', '行业类别', 'industry']
            industry_name = None

            for field in industry_fields:
                if field in info_df['item'].values:
                    industry_name = info_df[info_df['item'] == field]['value'].iloc[0]
                    break

            # 方法2：使用行业板块数据
            if not industry_name:
                try:
                    sector_df = ak.stock_board_industry_name_em()
                    # 在行业板块中搜索股票
                    for _, row in sector_df.iterrows():
                        constituents = ak.stock_board_industry_name_em(symbol=row['板块代码'])
                        if stock_code in constituents['代码'].values:
                            industry_name = row['板块名称']
                            break
                except:
                    pass

            # 方法3：使用默认行业（电力）
            if not industry_name:
                print(f"Using default industry for {stock_code}")
                industry_name = "电力"

            # 获取行业代码
            board_df = ak.stock_board_industry_name_em()
            industry_match = board_df[board_df['板块名称'] == industry_name]

            if not industry_match.empty:
                industry_code = industry_match['板块代码'].iloc[0]
            else:
                # 常见行业的默认代码映射
                default_codes = {
                    "电力": "BK0427",
                    "电力行业": "BK0427",
                    "银行": "BK0475",
                    "半导体": "BK1031",
                    "医药": "BK1040",
                    "新能源": "BK0495",
                    "默认": "sh000001"  # 上证指数
                }
                industry_code = default_codes.get(industry_name, "sh000001")

            return {'name': industry_name, 'code': industry_code}
        except Exception as e:
            print(f"Could not determine industry for {stock_code}: {e}")
            return {'name': '电力', 'code': 'BK0427'}  # 默认返回电力行业

    # --- Section 2: Feature Engineering ---

    def _calculate_technical_indicators(self, df):
        """Calculates a rich set of technical indicators."""
        # Moving Averages
        for ma in [5, 10, 20, 60]:
            df[f'ma{ma}'] = df['收盘'].rolling(window=ma).mean()
            df[f'ma{ma}_change'] = df[f'ma{ma}'].pct_change()

        # MACD
        df['ema12'] = df['收盘'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['收盘'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_mid'] = df['ma20']
        df['bb_std'] = df['收盘'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Average True Range (ATR)
        df['tr1'] = abs(df['最高'] - df['最低'])
        df['tr2'] = abs(df['最高'] - df['收盘'].shift())
        df['tr3'] = abs(df['最低'] - df['收盘'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].ewm(alpha=1 / 14, adjust=False).mean()

        # Lagged Returns
        for lag in [1, 3, 5]:
            df[f'return_{lag}d'] = df['收盘'].pct_change(periods=lag)

        # Drop intermediate calculation columns
        df = df.drop(columns=['ema12', 'ema26', 'bb_std', 'tr1', 'tr2', 'tr3', 'true_range'])
        return df

    def _calculate_valuation_ratios(self, stock_df, financial_df):
        """Calculates P/E and P/B ratios with robust error handling and fallbacks."""
        if financial_df is None or financial_df.empty:
            stock_df['pe_ratio'] = np.nan
            stock_df['pb_ratio'] = np.nan
            return stock_df

        try:
            # 确保数据排序正确
            stock_df_sorted = stock_df.sort_values('日期')
            financial_df_sorted = financial_df.sort_values('report_date')

            # 使用最近可用的财务报告
            merged_df = pd.merge_asof(
                stock_df_sorted,
                financial_df_sorted,
                left_on='日期',
                right_on='report_date',
                direction='backward'
            )

            # 计算PE比率
            if 'eps' in merged_df.columns:
                merged_df['pe_ratio'] = merged_df['收盘'] / merged_df['eps']
            else:
                merged_df['pe_ratio'] = np.nan

            # 计算PB比率 - 使用多种方法
            if 'roe' in merged_df.columns and 'eps' in merged_df.columns:
                # 方法1：使用ROE和EPS计算每股净资产
                merged_df['book_value_per_share'] = merged_df['eps'] / (merged_df['roe'] / 100)
                merged_df['pb_ratio'] = merged_df['收盘'] / merged_df['book_value_per_share']
            elif 'total_shares' in merged_df.columns and 'net_profit' in merged_df.columns:
                # 方法2：使用总股本和净利润
                merged_df['book_value_per_share'] = (
                        merged_df['net_profit'] / (merged_df['total_shares'] * 1e8))
                merged_df['pb_ratio'] = merged_df['收盘'] / merged_df['book_value_per_share']
            elif 'debt_ratio' in merged_df.columns and 'pe_ratio' in merged_df.columns:
                # 方法3：使用资产负债率近似计算
                merged_df['pb_ratio'] = merged_df['pe_ratio'] * (1 - merged_df['debt_ratio'] / 100)
            else:
                merged_df['pb_ratio'] = np.nan

            # 清理异常值
            merged_df['pe_ratio'] = merged_df['pe_ratio'].replace([np.inf, -np.inf], np.nan)
            merged_df['pb_ratio'] = merged_df['pb_ratio'].replace([np.inf, -np.inf], np.nan)

            # 合并回原始数据框
            stock_df['pe_ratio'] = merged_df['pe_ratio']
            stock_df['pb_ratio'] = merged_df['pb_ratio']

            # 添加其他有用的财务指标
            if 'roe' in merged_df.columns:
                stock_df['roe'] = merged_df['roe']
            if 'debt_ratio' in merged_df.columns:
                stock_df['debt_ratio'] = merged_df['debt_ratio']

            return stock_df
        except Exception as e:
            print(f"Valuation calculation error: {e}")
            stock_df['pe_ratio'] = np.nan
            stock_df['pb_ratio'] = np.nan
            return stock_df

    def _define_target_variable(self, df, forecast_horizon=5):
        """Creates the target variable for the prediction model."""
        # Shift future price back to the current row
        df[f'future_{forecast_horizon}d_close'] = df['收盘'].shift(-forecast_horizon)

        # Calculate the percentage change
        df[f'future_{forecast_horizon}d_return'] = (df[f'future_{forecast_horizon}d_close'] - df['收盘']) / df['收盘']

        # Create classification target based on return thresholds
        # This is a critical step and should be tailored to your strategy
        # Bins: [-inf, -5%), [-5%, -2%), [-2%, 2%], (2%, 5%], (5%, inf]
        bins = [-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf]
        labels = [0, 1, 2, 3, 4]  # 0:Strong Sell, 1:Sell, 2:Hold, 3:Buy, 4:Strong Buy
        df['target_class'] = pd.cut(df[f'future_{forecast_horizon}d_return'], bins=bins, labels=labels)

        return df

    # --- Section 3: Main Dataset Preparation ---

    def robust_data_fetch(self, fetch_func, *args, max_retries=3, **kwargs):
        """Robust data fetching with retry mechanism and proper result validation."""
        for attempt in range(max_retries):
            try:
                result = fetch_func(*args, **kwargs)
                # 正确检查DataFrame是否非空
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        if not result.empty:
                            return result
                    else:
                        return result  # 返回非DataFrame结果
                time.sleep(2 ** attempt)  # 指数退避
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # 指数退避

        print(f"Failed to fetch data after {max_retries} attempts")
        return None


    def prepare_ml_dataset(self, stock_code, start_date, end_date, forecast_horizon=5):
        """
        Orchestrates the entire data fetching, feature engineering, and alignment
        process to generate a final machine learning-ready dataset.
        """
        print(f"\n--- Starting ML Dataset Preparation for {stock_code} ---")

        # 1. Fetch Core Data (Stock Price)
        stock_df = self.get_stock_data(stock_code, start_date, end_date)
        if stock_df is None:
            print("Halting process due to failure in fetching stock data.")
            return None

        # 2. Fetch Industry and Index Data
        industry_info = self.get_stock_industry_info(stock_code)
        if industry_info:
            print(f"Stock {stock_code} belongs to industry: {industry_info['name']} ({industry_info['code']})")
            index_df = self.get_market_index_data(industry_info['code'], start_date, end_date)
            if index_df is not None:
                stock_df = pd.merge(stock_df, index_df, on='日期', how='left')
                # Calculate relative strength vs industry index
                stock_df['relative_return_1d'] = stock_df['涨跌幅'] - stock_df[f'index_{industry_info["code"]}_涨跌幅']

        # 3. Fetch and Align Financial & Macro Data
        # financial_df = self.get_financial_reports(stock_code)
        # macro_df = self.get_macro_economic_indicators(start_date, end_date)
        financial_df = self.robust_data_fetch(
            self.get_financial_reports,
            stock_code
        )

        macro_df = self.robust_data_fetch(
            self.get_macro_economic_indicators,
            start_date,
            end_date
        )

        # Use merge_asof for point-in-time correct alignment of low-frequency data
        # All dataframes must be sorted by date
        stock_df = stock_df.sort_values('日期')
        if macro_df is not None and not macro_df.empty:
            macro_df = macro_df.sort_values('日期')
            stock_df = pd.merge_asof(stock_df, macro_df, on='日期', direction='backward')

        # 4. Engineer Features
        print("Calculating technical indicators...")
        stock_df = self._calculate_technical_indicators(stock_df)

        print("Calculating valuation ratios (P/E, P/B)...")
        stock_df = self._calculate_valuation_ratios(stock_df, financial_df)

        # 5. Define Target Variable
        print(f"Defining target variable for {forecast_horizon}-day forecast...")
        stock_df = self._define_target_variable(stock_df, forecast_horizon)

        # 6. Final Cleanup
        # Drop rows with NaN in target variable (the last 'forecast_horizon' rows)
        stock_df = stock_df.dropna(subset=['target_class'])
        stock_df['target_class'] = stock_df['target_class'].astype(int)

        # Reorder columns for clarity
        core_cols = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', 'volume', 'amount']
        target_cols = [f'future_{forecast_horizon}d_close', f'future_{forecast_horizon}d_return', 'target_class']
        feature_cols = [col for col in stock_df.columns if col not in core_cols + target_cols]
        final_cols = core_cols + feature_cols + target_cols
        stock_df = stock_df[final_cols]

        # Add stock code column
        stock_df['股票代码'] = stock_code

        print("\n--- ML Dataset Preparation Complete ---")
        print(f"Final dataset shape: {stock_df.shape}")
        print("Sample of the final dataset:")
        print(stock_df.head())
        print("\nDataset Info:")
        stock_df.info()

        return stock_df


if __name__ == '__main__':
    # --- Main Execution ---
    scraper = FinancialDataEngine()

    # --- Parameters ---
    STOCK_TO_ANALYZE = '600900'  # Example: Three Gorges Renewables
    START_DATE = '2021-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    FORECAST_DAYS = 5

    # --- Generate the Dataset ---
    ml_dataset = scraper.prepare_ml_dataset(
        stock_code=STOCK_TO_ANALYZE,
        start_date=START_DATE,
        end_date=END_DATE,
        forecast_horizon=FORECAST_DAYS
    )

    # --- Save the Final Dataset ---
    if ml_dataset is not None:
        print("\nSaving the final ML-ready dataset...")
        scraper.save_data(
            data=ml_dataset,
            filename_prefix=f'ml_dataset_{STOCK_TO_ANALYZE}',
            file_format='csv'  # CSV is often more convenient for ML frameworks
        )

        # Optional: Save a JSON version as well
        scraper.save_data(
            data=ml_dataset,
            filename_prefix=f'ml_dataset_{STOCK_TO_ANALYZE}',
            file_format='json'
        )

    print("\nProcess finished successfully.")