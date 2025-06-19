
import functools
import pymysql
from flask import Flask, render_template, request, jsonify, url_for, redirect, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import jieba
from pyecharts.charts import Funnel, WordCloud as PyeWordCloud, Radar
from pyecharts.charts import Kline, Line

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from crawler_news import update_news
from flask_caching import Cache
import logging
import json
import os
import sqlite3
from sqlalchemy import create_engine, text
import numpy as np
from math import ceil
from pyecharts.charts import Timeline, Pie, Bar
from pyecharts import options as opts
import pandas as pd
import akshare as ak
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.config["SECRET_KEY"] = os.urandom(24)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# 配置缓存
# cache = Cache(app, config={'CACHE_TYPE': 'simple'}) # 开发环境使用 simple，生产环境用 Redis
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache'
cache = Cache(app)
cache.init_app(app)
DB_URL = "mysql+pymysql://root:root@localhost:3306/text?charset=utf8mb4"
engine = create_engine(DB_URL)

db = pymysql.connect(host="localhost", port=3306, user="root", passwd="root", db="text", charset='utf8mb4')
cur = db.cursor()

@app.route('/')
@cache.cached(timeout=3600)
def root():
    try:
        update_news()
    except Exception as e:
        app.logger.warning(f"更新新闻失败: {e}")
    user_name = session.get('user_name')

    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute(
        """SELECT image_url, title, pub_time, summary, article_link
                FROM news
                ORDER BY pub_time DESC
                LIMIT 10;
        """)
    news_data = cursor.fetchall()
    cursor.close()

    return render_template('index.html', data=news_data, user_name=user_name)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not all([name, email, password]):
            return render_template("register.html", message="请填写完整的注册信息！")

        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            return render_template("register.html", message="该邮箱已被注册，请使用其他邮箱！")

        # 【安全核心】对密码进行哈希处理
        hashed_password = generate_password_hash(password)

        # 将哈希后的密码存入数据库
        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        db.commit()

        # 【体验优化】注册成功后，自动为用户登录
        # 获取刚插入用户的 ID
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        session['user_id'] = user[0]
        session['user_name'] = name

        flash('注册成功并已自动登录！', 'success')
        return redirect(url_for('root'))

    return render_template("register.html")


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not email or not password:
            return render_template("login.html", message="请填写完整的登录信息！")

        # 1. 先只根据 email 查询用户是否存在
        cur.execute(
            "SELECT id, name FROM users WHERE email=%s AND password=%s",
            (email, password)
        )
        user = cur.fetchone() # user 会是 (id, name, hashed_password)

        # 2. 如果用户存在，再校验密码哈希
        if user and check_password_hash(user[2], password):  # user[2] 是数据库中存储的哈希密码
            # 登录成功：在 session 中记录用户信息
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            return redirect(url_for('root'))
        else:
            # 用户不存在或密码错误
            return render_template("login.html", message="邮箱或密码错误，请重新登录！")

    # GET 请求时，直接渲染登录页
    return render_template("login.html")

PAGE_SIZE = 30  # Number of items per page
# Suppress the warning about SQLAlchemy track modifications

def _handle_stock_history(request_args):
    """
    Handles all logic for fetching and displaying stock history data.
    """
    page = request_args.get('page', 1, type=int)
    stock_code = request_args.get('symbol', '').strip()
    start_date = request_args.get('start_date', '').strip()
    end_date = request_args.get('end_date', '').strip()

    context = {
        "query_params": {'symbol': stock_code, 'start_date': start_date, 'end_date': end_date},
        "data": [], "page": page, "total_pages": 0, "error": None
    }

    if not all([stock_code, start_date, end_date]):
        return context

    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        context['error'] = "日期格式不正确，请使用 YYYY-MM-DD 格式。"
        return context

    if not engine:
        context['error'] = "数据库未连接。"
        return context

    with engine.connect() as connection:
        try:
            count_query = text("""
                SELECT COUNT(*) FROM stock_history
                WHERE symbol=:symbol AND trade_date BETWEEN :start_date AND :end_date
            """)
            count_params = {"symbol": stock_code, "start_date": start_date, "end_date": end_date}
            total_count = connection.execute(count_query, count_params).scalar_one_or_none() or 0

            if total_count == 0:
                app.logger.info(f"No data for {stock_code} in DB. Fetching from Akshare.")
                df = ak.stock_zh_a_hist(
                    symbol=stock_code, period="daily",
                    start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''),
                    adjust="qfq"
                )
                if df is None or df.empty:
                    context['error'] = "无法获取数据，请检查股票代码或日期范围是否正确。"
                    return context

                df.rename(columns={'日期': 'trade_date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close',
                                   '成交量': 'volume', '成交额': 'turnover'}, inplace=True)
                df['symbol'] = stock_code
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date

                df_to_save = df[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover']].copy()
                df_to_save.replace({np.nan: None, pd.NaT: None}, inplace=True)

                df_to_save.to_sql('stock_history', con=connection, if_exists='append', index=False)
                total_count = connection.execute(count_query, count_params).scalar_one_or_none() or 0

            context['total_pages'] = ceil(total_count / PAGE_SIZE)
            offset = (page - 1) * PAGE_SIZE

            data_query = text("""
                SELECT symbol, trade_date, open, high, low, close, volume, turnover
                FROM stock_history WHERE symbol=:symbol AND trade_date BETWEEN :start_date AND :end_date
                ORDER BY trade_date DESC LIMIT :limit OFFSET :offset
            """)
            data_params = {**count_params, "limit": PAGE_SIZE, "offset": offset}
            result = connection.execute(data_query, data_params)
            context['data'] = result.fetchall()
        except Exception as e:
            app.logger.error(f"Error handling stock history for {stock_code}: {e}")
            context['error'] = f"处理股票数据时发生错误: {e}"
    return context


def _handle_interest_rate(request_args):
    """
    Handles all logic for fetching and displaying interest rate data.
    """
    page = request_args.get('page', 1, type=int)
    country = request_args.get('country', 'china').lower().strip()

    context = {"query_params": {'country': country}, "data": [], "page": page, "total_pages": 0, "error": None}

    supported = {'china': 'macro_bank_china_interest_rate', 'usa': 'macro_bank_usa_interest_rate',
                 'euro': 'macro_bank_euro_interest_rate',
                 'japan': 'macro_bank_japan_interest_rate', 'russia': 'macro_bank_russia_interest_rate',
                 'india': 'macro_bank_india_interest_rate'}
    if country not in supported:
        context['error'] = f"不支持的国家: {country}"
        return context

    if not engine:
        context['error'] = "数据库未连接。"
        return context

    with engine.connect() as connection:
        try:
            df = getattr(ak, supported[country])()
            if df is None or df.empty:
                context['error'] = "Akshare 未返回任何数据。"
                return context

            df.rename(columns={'日期': 'rate_date', '今值': 'value', '预测值': 'predict', '前值': 'prev'}, inplace=True)
            df['country'] = country
            df['rate_date'] = pd.to_datetime(df['rate_date'], errors='coerce').dt.date
            for col in ['value', 'predict', 'prev']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df_to_save = df[['country', 'rate_date', 'value', 'predict', 'prev']].copy()
            df_to_save.replace({np.nan: None, pd.NaT: None}, inplace=True)

            with connection.begin():
                connection.execute(text("DELETE FROM interest_rate WHERE country=:country"), {"country": country})
                df_to_save.to_sql('interest_rate', con=connection, if_exists='append', index=False)

            count_query = text("SELECT COUNT(*) FROM interest_rate WHERE country=:country")
            total_count = connection.execute(count_query, {"country": country}).scalar_one()

            context['total_pages'] = ceil(total_count / PAGE_SIZE)
            offset = (page - 1) * PAGE_SIZE

            data_query = text("""
                SELECT country, rate_date, value, predict, prev
                FROM interest_rate WHERE country=:country ORDER BY rate_date DESC LIMIT :limit OFFSET :offset
            """)
            data_params = {"country": country, "limit": PAGE_SIZE, "offset": offset}
            result = connection.execute(data_query, data_params)
            context['data'] = result.fetchall()
        except Exception as e:
            app.logger.error(f"Error handling interest rate for {country}: {e}")
            context['error'] = f"处理利率数据时发生错误: {e}"
    return context


@app.route('/about')
def about_page():
    """
    Main data display route. Dispatches to the correct handler based on 'table' parameter.
    """
    selected_table = request.args.get('table', 'stock_history')
    context = {}
    if selected_table == 'stock_history':
        context = _handle_stock_history(request.args)
    elif selected_table == 'interest_rate':
        context = _handle_interest_rate(request.args)
    else:
        return redirect(url_for('about_page', table='stock_history'))

    return render_template("about.html", selected_table=selected_table, **context)



def save_json(data, filename):
    # 确保 JSON 保存目录存在
    os.makedirs('datas', exist_ok=True)
    os.makedirs('datas', exist_ok=True)
    path = os.path.join('datas', filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def to_db(df: pd.DataFrame, table_name: str):
    df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

# 获取近14个交易日的规则数据函数
def get_recent_trade_days(n=14):
    dfs = []
    today = datetime.now()
    # 逆向遍历天数，直到收集到n个交易日
    for i in range(1, 50):  # 设置一个较大的范围，避免死循环
        dt = today - timedelta(days=i)
        date_str = dt.strftime('%Y%m%d')
        try:
            df_rule = ak.futures_rule(date=date_str)
            if not df_rule.empty:
                df_rule['date'] = date_str
                dfs.append(df_rule)
                if len(dfs) >= n:
                    break
        except Exception:
            continue
    if not dfs:
        raise ValueError("未获取到近14天的交易日历数据，请检查日期或接口返回。")
    return pd.concat(dfs, ignore_index=True)

def cache_by_query_param(param_name, default_value):
    """自定义缓存装饰器，基于查询参数"""

    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # 获取查询参数值
            param_value = request.args.get(param_name, default_value)

            # 生成缓存键
            cache_key = f"{f.__name__}_{param_value}"

            # 检查缓存
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response

            # 执行视图函数
            response = f(*args, **kwargs)

            # 缓存响应
            cache.set(cache_key, response, timeout=300)

            return response

        return decorated_function

    return decorator
@app.route('/services')
@cache_by_query_param('chart_type','wordcloud')
def services_page():
    chart_type = request.args.get('chart_type', 'wordcloud')

    # 漏斗图：基于 futures_rule + 综合评分公式 + 风险等级
    if chart_type == 'funnel':
        try:
            # 1. 拉取最近一个交易日的规则数据
            # df_rule = ak.futures_rule(date=datetime.now().strftime('%Y%m%d'))
            df_rule = get_recent_trade_days(14)
            if df_rule.empty:
                raise ValueError("无法获取当日期货规则，请手动指定近14天内有效交易日。")

            # 2. 数据预处理
            # 填充特殊参数
            df = df_rule.copy()
            df['特殊合约参数调整'] = df['特殊合约参数调整'].fillna('无')
            # 将百分号数字转为 float
            for col in ['交易保证金比例', '涨跌停板幅度']:
                df[col] = df[col].astype(str).str.rstrip('%').astype(float)
            # 其余数值列转为数值
            for col in ['合约乘数', '最小变动价位']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 缺失数值填中位数
            num_cols = ['交易保证金比例', '涨跌停板幅度', '合约乘数', '最小变动价位']
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

            # 3. 计算综合风险评分
            weights = {
                '交易保证金比例': 0.4,
                '涨跌停板幅度': 0.3,
                '合约乘数': 0.2,
                '最小变动价位': 0.1
            }
            scaler = MinMaxScaler()
            # 先将指标缩放到 [0,1]
            scaled = scaler.fit_transform(df[list(weights.keys())])
            df_scores = pd.DataFrame(scaled, columns=list(weights.keys()))
            # 计算得分
            scores = []
            for i, row in df_scores.iterrows():
                s = 0
                for j, (k, w) in enumerate(weights.items()):
                    val = row[k]
                    # 保证金与涨跌停幅度，值越大风险越高 => (1 - val)
                    if k in ['交易保证金比例', '涨跌停板幅度']:
                        s += (1 - val) * w
                    else:
                        s += val * w
                scores.append(s * 100)  # 百分制
            df['综合风险评分'] = scores

            # 4. 风险等级分类
            def classify(score):
                if score >= 80: return '低风险'
                if score >= 60: return '中低风险'
                if score >= 40: return '中等风险'
                if score >= 20: return '中高风险'
                return '高风险'

            df['风险等级'] = df['综合风险评分'].apply(classify)

            # 5. 按风险等级统计数量，并排序
            order = ['低风险', '中低风险', '中等风险', '中高风险', '高风险']
            cnt = df['风险等级'].value_counts().reindex(order, fill_value=0).reset_index()
            cnt.columns = ['风险等级', '数量']
            data = list(zip(cnt['风险等级'], cnt['数量']))

            # 6. 保存并写库
            save_json(data, 'funnel.json')
            to_db(pd.DataFrame(data, columns=['风险等级', '数量']), 'funnel')

            # 7. 绘制漏斗图 (白底)
            funnel = Funnel(init_opts=opts.InitOpts(bg_color="white"))
            funnel.add(
                "品种数量",
                data,
                label_opts=opts.LabelOpts(position="inside")  # 标签显示在内部
            )
            funnel.set_global_opts(
                title_opts=opts.TitleOpts(
                    title="国泰君安期货品种数量风险等级分布",
                    pos_left="center",
                    pos_top="20px"  # 调整标题的垂直位置
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical",  # 垂直排列
                    pos_top="middle",  # 垂直居中
                    pos_left="5%"  # 靠左显示，避免与图重叠
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter="{a} <br/>{b}: {c} ({d}%)"
                )
            )
            myechart = funnel.render_embed()
        except Exception as e:
            myechart = f"<p>生成漏斗图出错：{e}</p>"

    # 词云图：从 news 表提取关键词，并设背景白
    elif chart_type == 'wordcloud':
        try:
            df_news = pd.read_sql("SELECT title, summary FROM news", con=engine)
            texts = df_news['title'].fillna('') + ' ' + df_news['summary'].fillna('')
            combined = " ".join(texts.tolist())
            words = jieba.lcut(combined)
            filtered = [w for w in words if len(w) >= 2]
            from collections import Counter
            counter = Counter(filtered)
            most = counter.most_common(50)
            data = [(w, int(freq)) for w, freq in most]
            save_json(data, 'wordcloud.json')
            if data:
                to_db(pd.DataFrame(data, columns=['word', 'freq']), 'wordcloud')
            # 使用 Pyecharts WordCloud，背景白色
            wc = PyeWordCloud(init_opts=opts.InitOpts(bg_color="white", width="800px", height="400px"))
            wc.add("", data, word_size_range=[20, 100])
            wc.set_global_opts(title_opts=opts.TitleOpts(title=f"新闻关键字词云",pos_left="center"))
            myechart = wc.render_embed()
        except Exception as e:
            myechart = f"<p>生成词云图出错：{e}</p>"

    # 3. 雷达图：中美重要六个经济指标对比
    elif chart_type == 'radar':
        try:
            # 准备近两年年份列表
            current_year = pd.Timestamp.now().year
            years = [current_year - 1, current_year]

            # 通用年度平均函数，兼容多种日期格式
            def annual_avg(df, date_col, value_col):
                df2 = df.copy()
                # 统一将日期列转字符串
                df2['__dt_str'] = df2[date_col].astype(str)

                # 解析为 datetime，尝试多种格式
                def parse_date(s):
                    # 常见格式：YYYY-MM-DD 或 YYYY/MM/DD
                    try:
                        return pd.to_datetime(s, errors='raise')
                    except:
                        # 可能为 YYYYMM 或 YYYYMMDD
                        if len(s) == 6 and s.isdigit():
                            # YYYYMM，取当月第一天
                            try:
                                return pd.to_datetime(s, format='%Y%m', errors='raise')
                            except:
                                return pd.NaT
                        elif len(s) == 8 and s.isdigit():
                            # YYYYMMDD
                            try:
                                return pd.to_datetime(s, format='%Y%m%d', errors='raise')
                            except:
                                return pd.NaT
                        else:
                            # 其它格式，让 pandas 自行尝试
                            try:
                                return pd.to_datetime(s, errors='coerce')
                            except:
                                return pd.NaT

                # 应用解析
                df2['_parsed_date'] = df2['__dt_str'].apply(parse_date)
                # 丢弃无法解析的
                df2 = df2.dropna(subset=['_parsed_date'])
                if df2.empty:
                    return pd.Series({y: None for y in years})
                df2['year'] = df2['_parsed_date'].dt.year
                # 按年平均 value_col
                ser = df2.groupby('year')[value_col].mean()
                # 选出目标年份，若缺失则填 None
                return ser.reindex(years, fill_value=None)

            # 1. GDP指标
            # 中国年度GDP报告（季度报告，取当年最后一期“今值”）
            china_gdp_df = ak.macro_china_gdp_yearly()
            # 解析“日期”列
            china_gdp_df['parsed'] = pd.to_datetime(china_gdp_df['日期'].astype(str), errors='coerce')
            # 取每年最后一期
            china_gdp = (
                china_gdp_df.dropna(subset=['parsed'])
                .assign(year=lambda d: d['parsed'].dt.year)
                .sort_values('parsed')
                .groupby('year')['今值']
                .last()
                .reindex(years, fill_value=None)
            )

            # 美国GDP月度（或季度）同比，用年度平均
            usa_gdp_df = ak.macro_usa_gdp_monthly()
            usa_gdp = annual_avg(usa_gdp_df, '日期', '今值')

            # 2. CPI指标
            # 中国年度 CPI 报告，取当年最后一期“今值”
            china_cpi_df = ak.macro_china_cpi_yearly()
            china_cpi_df['parsed'] = pd.to_datetime(china_cpi_df['日期'].astype(str), errors='coerce')
            china_cpi = (
                china_cpi_df.dropna(subset=['parsed'])
                .assign(year=lambda d: d['parsed'].dt.year)
                .sort_values('parsed')
                .groupby('year')['今值']
                .last()
                .reindex(years, fill_value=None)
            )
            # 美国CPI年率（月度“现值”），年度平均
            usa_cpi_df = ak.macro_usa_cpi_yoy()
            # 注意：该接口列名可能为 '时间' 和 '现值'
            usa_cpi = annual_avg(usa_cpi_df, '时间', '现值')

            # 3. 制造业PMI指标
            # 中国官方制造业PMI年度报告，取当年最后一期
            china_pmi_df = ak.macro_china_pmi_yearly()
            china_pmi_df['parsed'] = pd.to_datetime(china_pmi_df['日期'].astype(str), errors='coerce')
            china_pmi = (
                china_pmi_df.dropna(subset=['parsed'])
                .assign(year=lambda d: d['parsed'].dt.year)
                .sort_values('parsed')
                .groupby('year')['今值']
                .last()
                .reindex(years, fill_value=None)
            )
            # 美国ISM制造业PMI（月度“今值”），年度平均
            usa_pmi_df = ak.macro_usa_ism_pmi()
            usa_pmi = annual_avg(usa_pmi_df, '日期', '今值')

            # 4. 对外贸易指标：出口年率
            # 中国出口年率（月度），年度平均
            china_exp_df = ak.macro_china_exports_yoy()
            china_exp = annual_avg(china_exp_df, '日期', '今值')
            # 美国出口价格指数（月度“今值”），作为近似对比
            usa_exp_df = ak.macro_usa_export_price()
            usa_exp = annual_avg(usa_exp_df, '日期', '今值')

            # 5. 失业率指标
            # 中国城镇调查失业率（月度 'date' 格式如 'YYYYMM'），年度平均
            china_unemp_df = ak.macro_china_urban_unemployment()
            china_unemp = annual_avg(china_unemp_df, 'date', 'value')
            # 美国失业率（月度“今值”），年度平均
            usa_unemp_df = ak.macro_usa_unemployment_rate()
            usa_unemp = annual_avg(usa_unemp_df, '日期', '今值')

            # 6. PPI指标
            # 中国PPI年度报告，取当年最后一期
            china_ppi_df = ak.macro_china_ppi_yearly()
            china_ppi_df['parsed'] = pd.to_datetime(china_ppi_df['日期'].astype(str), errors='coerce')
            china_ppi = (
                china_ppi_df.dropna(subset=['parsed'])
                .assign(year=lambda d: d['parsed'].dt.year)
                .sort_values('parsed')
                .groupby('year')['今值']
                .last()
                .reindex(years, fill_value=None)
            )
            # 美国PPI（月度“今值”），年度平均
            usa_ppi_df = ak.macro_usa_ppi()
            usa_ppi = annual_avg(usa_ppi_df, '日期', '今值')

            # 构造最新一年数据对比
            # 构造最新一年数据对比
            latest = years[-1]
            china_vals = [
                china_gdp.get(latest) or 0,
                china_cpi.get(latest) or 0,
                china_pmi.get(latest) or 0,
                china_exp.get(latest) or 0,
                china_unemp.get(latest) or 0,
                china_ppi.get(latest) or 0,
            ]
            usa_vals = [
                usa_gdp.get(latest) or 0,
                usa_cpi.get(latest) or 0,
                usa_pmi.get(latest) or 0,
                usa_exp.get(latest) or 0,
                usa_unemp.get(latest) or 0,
                usa_ppi.get(latest) or 0,
            ]
            inds = ['GDP同比(%)', 'CPI同比(%)', 'PMI', '出口同比(%)', '失业率(%)', 'PPI同比(%)']

            # 保存 JSON 与写库
            df_radar = pd.DataFrame([china_vals, usa_vals], columns=inds, index=['China', 'US']).reset_index().rename(
                columns={'index': 'country'})
            save_json(df_radar.to_dict('records'), 'radar.json')
            to_db(df_radar, 'radar')

            # 解决图表显示异常问题 - 为每个指标单独设置合适的坐标范围
            # 根据指标类型分别设置最大值
            max_vals = []
            for i, indicator in enumerate(inds):
                # 获取中美数据中的最大值
                max_val = max(abs(china_vals[i]), abs(usa_vals[i]))

                # 根据不同指标类型设置不同的缩放比例
                if indicator == 'PMI':
                    # PMI通常在50左右波动，设置固定范围40-60
                    max_vals.append(60)
                elif indicator == '失业率(%)':
                    # 失业率通常0-15%，设置稍大范围
                    max_vals.append(max_val * 1.5)
                elif indicator in ['GDP同比(%)', 'CPI同比(%)', '出口同比(%)', 'PPI同比(%)']:
                    # 百分比指标，设置适当范围
                    scale_factor = 1.8 if max_val < 5 else 1.3
                    max_vals.append(max_val * scale_factor)
                else:
                    # 默认缩放
                    max_vals.append(max_val * 1.3)

            # 确保最小值合理（特别是对于负值）
            min_vals = []
            for i, indicator in enumerate(inds):
                min_val = min(china_vals[i], usa_vals[i])
                # 对于PPI等可能出现负值的指标，设置负轴范围
                if indicator in ['PPI同比(%)'] and min_val < 0:
                    min_vals.append(min_val * 5)
                else:
                    min_vals.append(0)

            # 创建雷达图schema - 为每个指标设置独立范围
            schema = []
            for i in range(len(inds)):
                # 特殊处理失业率（反向指标，值越小越好）
                if inds[i] == '失业率(%)':
                    # 反转数值（值越小在雷达图上越靠外）
                    china_vals[i] = max_vals[i] - china_vals[i]
                    usa_vals[i] = max_vals[i] - usa_vals[i]
                    # 设置指标名称显示原始值
                    schema.append(
                        opts.RadarIndicatorItem(
                            name=f"{inds[i]}\n(中:{china_unemp.get(latest) or 0:.1f} 美:{usa_unemp.get(latest) or 0:.1f})",
                            max_=max_vals[i],
                            min_=min_vals[i]
                        )
                    )
                else:
                    # 其他指标正常显示
                    schema.append(
                        opts.RadarIndicatorItem(
                            name=inds[i],
                            max_=max_vals[i],
                            min_=min_vals[i]
                        )
                    )

            # 创建雷达图
            radar = Radar(init_opts=opts.InitOpts(
                bg_color="white",
                width="800px",
                height="600px"
            ))

            radar.add_schema(
                schema=schema,
                splitarea_opt=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.1)
                ),
                textstyle_opts=opts.TextStyleOpts(font_size=10)
            )

            # 添加数据系列
            radar.add(
                "中国",
                [china_vals],
                linestyle_opts=opts.LineStyleOpts(width=2),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
                color="#c23531"
            )
            radar.add(
                "美国",
                [usa_vals],
                linestyle_opts=opts.LineStyleOpts(width=2),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
                color="#2f4554"
            )

            # 设置全局选项
            radar.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"中美经济指标对比 ({latest-1}~{latest}年)",
                    pos_left="center"
                ),
                legend_opts=opts.LegendOpts(
                    pos_right="10%",
                    pos_top="10%",
                    orient="vertical"
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter="{a}: {c}"
                )
            )

            # 添加数据标签
            radar.set_series_opts(
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter="{c}"
                )
            )

            myechart = radar.render_embed()

        except Exception as e:
            import traceback
            error_msg = f"<p>生成雷达图出错：{str(e)}</p><pre>{traceback.format_exc()}</pre>"
            myechart = error_msg

    elif chart_type == 'fund_line':
        try:
            # 获取所有会员信息
            df_mem = ak.amac_member_info()
            df_mem = df_mem.copy()
            df_mem['入会时间_parsed'] = pd.to_datetime(df_mem['入会时间'].astype(str), errors='coerce')
            df_mem = df_mem.dropna(subset=['入会时间_parsed'])
            df_mem['year'] = df_mem['入会时间_parsed'].dt.year

            # 确定近5年年份列表，例如当前 2025，则取 2021-2025
            current_year = datetime.now().year
            years = [current_year - 4, current_year - 3, current_year - 2, current_year - 1, current_year]
            df_recent = df_mem[df_mem['year'].isin(years)].copy()

            # 分类统计：私募/公募机构数
            counts = {'year': [], '私募': [], '公募': []}
            for y in years:
                df_y = df_recent[df_recent['year'] == y]
                cnt_pm = df_y[df_y['机构类型'].str.contains('私募', na=False)].shape[0]
                cnt_gm = df_y[df_y['机构类型'].str.contains('公募', na=False)].shape[0]
                counts['year'].append(str(y))
                counts['私募'].append(int(cnt_pm))
                counts['公募'].append(int(cnt_gm))
            df_line = pd.DataFrame(counts)

            # 保存 JSON 与数据库（修正 to_dict 参数）
            save_json(df_line.to_dict('records'), 'fund_line.json')
            to_db(df_line, 'fund_member_5years')

            # 构建折线图（白背景），优化标题和图例
            line = (
                Line(init_opts=opts.InitOpts(bg_color="white", width="800px", height="400px"))
                .add_xaxis(df_line['year'].tolist())
                .add_yaxis(
                    "私募新会员数",
                    df_line['私募'].tolist(),
                    label_opts=opts.LabelOpts(is_show=True, position="top"),
                    is_smooth=True
                )
                .add_yaxis(
                    "公募新会员数",
                    df_line['公募'].tolist(),
                    label_opts=opts.LabelOpts(is_show=True, position="top"),
                    is_smooth=True
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="近5年基金会员（新入会机构）数变化",
                        pos_left="center",
                        pos_top="20px"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    xaxis_opts=opts.AxisOpts(name="年份", type_="category"),
                    yaxis_opts=opts.AxisOpts(name="数量"),
                    legend_opts=opts.LegendOpts(
                        orient="horizontal",
                        pos_top="60px",
                        pos_left="center"
                    )
                )
            )
            myechart = line.render_embed()
        except Exception as e:
            myechart = f"<p>生成基金会员折线图出错：{e}</p>"
    else:
        myechart = "<p>请选择有效的图表类型。</p>"

    return render_template("services.html", myechart=myechart, chart_type=chart_type)

def calculate_ma(day_count, data):
    result = []
    for i in range(len(data)):
        if i < day_count - 1:
            result.append(float('nan'))
        else:
            window = data[i - day_count + 1 : i + 1]
            # 若 window 中含 None 或 NaN，可先过滤或决定返回 NaN
            try:
                vals = [float(v) for v in window]
                result.append(sum(vals) / day_count)
            except:
                result.append(float('nan'))
    return result

@app.route('/portfolio')
@cache.cached(timeout=3600)
def portfolio_page():
    """
    Generates and displays a professional Kline chart for a given stock.
    """
    # 1. Fetch real-time stock data using akshare
    stock_code = "600905"
    try:
        # Fetch daily historical data for the last year (qfq: forward-adjusted prices)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception as e:
        return f"Error fetching stock data: {e}"

    # 2. Prepare data for the chart
    # Ensure the dataframe is not empty
    if df.empty:
        return "Could not fetch data for the stock. The symbol might be incorrect or delisted."

    # Rename columns to be more accessible
    df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)

    # Convert date strings to datetime objects for sorting
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # Prepare data for Kline chart [open, close, low, high]
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    kline_data = df[['open', 'close', 'low', 'high']].values.tolist()
    volumes = df['volume'].tolist()

    # 3. Create the Kline (Candlestick) Chart
    kline_chart = (
        Kline()
        .add_xaxis(xaxis_data=dates)
        .add_yaxis(
            series_name="日K",
            y_axis=kline_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",       # Color for bullish candle
                color0="#14b143",      # Color for bearish candle
                border_color="#ef232a",
                border_color0="#14b143",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"三峡能源({stock_code}) - 专业K线图",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=20)
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1], range_start=85, range_end=100),
                opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1], pos_top="90%", range_start=85, range_end=100),
            ],
            legend_opts=opts.LegendOpts(pos_right="20px", orient="vertical"),
        )
    )

    # 4. Create Moving Average Lines
    ma_line = (
        Line()
        .add_xaxis(xaxis_data=dates)
        .add_yaxis(
            series_name="MA5",
            y_axis=calculate_ma(5, df['close'].tolist()),
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.8),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="MA10",
            y_axis=calculate_ma(10, df['close'].tolist()),
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.8),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="MA20",
            y_axis=calculate_ma(20, df['close'].tolist()),
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.8),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="MA30",
            y_axis=calculate_ma(30, df['close'].tolist()),
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.8),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                split_number=3,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
            ),
        )
    )
    # 5. Overlap the Kline and MA Line charts
    # The Kline chart now includes the MA lines
    final_chart = kline_chart.overlap(ma_line)

    return render_template("portfolio.html", myechart=final_chart.render_embed())



# --- Helper Functions for Data Persistence ---
def init_db_and_save(df: pd.DataFrame, table_name: str):
    """Saves a DataFrame to a SQLite database."""
    # The 'instance' folder is automatically created by Flask for the database
    if not os.path.exists('instance'):
        os.makedirs('instance')
    conn = sqlite3.connect('instance/app.db')
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Data saved to SQLite table '{table_name}'")
    finally:
        conn.close()


# --- Chart Generation Functions ---
def init_db_and_save(df, table_name):
    """Saves a DataFrame to an SQLite database."""
    try:
        conn = sqlite3.connect('finance_data.db')
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        app.logger.info(f"Successfully saved data to table {table_name} in finance_data.db")
    except Exception as e:
        app.logger.error(f"Error saving data to SQLite: {e}")


# --- Helper Function for Robust Column Detection ---

def find_column(df: pd.DataFrame, candidates: list) -> str:
    """
    在 df.columns 中查找匹配候选名的列：
    优先精确匹配，其次部分匹配。返回第一个匹配列名，若未找到返回 None。
    """
    cols = df.columns.tolist()
    # 精确匹配
    for name in candidates:
        if name in cols:
            return name
    # 部分匹配
    for name in candidates:
        for col in cols:
            if name in col:
                return col
    return None

def draw_timeline_pie_chart():
    app.logger.info("Generating Timeline Pie Chart data...")
    end_year = datetime.now().year
    years = list(range(end_year - 4, end_year + 1))

    try:
        df_sh = ak.stock_register_sh()
        df_sz = ak.stock_register_sz()
        df_cyb = ak.stock_register_cyb()
        df_kcb = ak.stock_register_kcb()
    except Exception as e:
        app.logger.error(f"Error fetching IPO data: {e}")
        return f"<p class='text-red-500'>Error fetching IPO data: {e}</p>"

    board_data = {
        '主板(沪)': df_sh,
        '主板(深)': df_sz,
        '创业板': df_cyb,
        '科创板': df_kcb
    }

    counts = {'year': years}
    for board_name in board_data.keys():
        counts[board_name] = []

    for year in years:
        for board_name, df in board_data.items():
            if df is None or df.empty:
                counts[board_name].append(0)
                continue
            df_tmp = df.copy()

            # 找日期列
            date_col = find_column(df_tmp, ['更新日期', '受理日期', '最新状态', '注册生效日期', '日期'])
            if not date_col or date_col not in df_tmp.columns:
                app.logger.warning(f"No date column for {board_name}, year filter skipped.")
                counts[board_name].append(0)
                continue
            df_tmp['_date_parsed'] = pd.to_datetime(df_tmp[date_col], errors='coerce')
            df_year = df_tmp.dropna(subset=['_date_parsed'])
            df_year = df_year[df_year['_date_parsed'].dt.year == year]

            # 找状态列：包括“审核状态”“最新状态”“状态”“审核结果”等
            status_col = find_column(df_year, ['审核状态', '最新状态', '审核结果', '状态'])
            approved_count = 0
            if status_col and status_col in df_year.columns:
                # 过滤包含“注册生效”或“已通过发审会”或“核准”等关键字的记录
                mask = df_year[status_col].astype(str).str.contains('注册生效|已通过发审会|核准', na=False)
                approved_count = df_year[mask].shape[0]
            else:
                # 若无状态列，则退为当年条目总数
                approved_count = df_year.shape[0]
            counts[board_name].append(int(approved_count))

    # Persist data
    save_json(counts, 'ipo_by_year.json')
    df_db = pd.DataFrame(counts)
    init_db_and_save(df_db, 'ipo_by_year')

    # 构建 Timeline，背景白
    tl = Timeline(init_opts=opts.InitOpts(width="100%", height="600px", bg_color="#FFFFFF"))
    tl.add_schema(is_auto_play=True, play_interval=1500)

    for i, year in enumerate(years):
        data_pair = [(board, counts[board][i]) for board in board_data.keys() if counts[board][i] > 0]
        if not data_pair:
            continue
        pie = (
            Pie(init_opts=opts.InitOpts(bg_color="#FFFFFF"))
            .add(
                "上市获批数",
                data_pair,
                rosetype="radius",
                radius=["30%", "70%"],
                label_opts=opts.LabelOpts(formatter="{b}: {c}", position="inside")
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{year}年A股各板块上市获批数量", pos_left="center", pos_top="20px"),
                legend_opts=opts.LegendOpts(pos_left="left", orient="vertical", pos_top="15%")
            )
        )
        tl.add(pie, f"{year}年")
    return tl.render_embed()


def draw_finance_bar_chart():
    """
    Timeline 图：近5年月度制造业PMI、非制造业PMI、CPI同比，以及年度社融结构饼图。
    重点：针对接口不同的“月份”/“日期”格式做专门解析，确保非空数据列表，避免只显示时间轴。
    """
    app.logger.info("Generating Finance Indices Timeline Chart...")
    end_year = datetime.now().year
    current_month = datetime.now().month
    years = list(range(end_year - 4, end_year + 1))

    # --- 1. Data Fetching ---
    try:
        df_pmi = ak.macro_china_pmi()  # 列 '月份', '制造业-指数','制造业-同比增长','非制造业-指数','非制造业-同比增长'
        df_non_pmi = ak.macro_china_non_man_pmi()  # 列 '日期','今值' 即非制造业PMI
        df_sfi = ak.macro_china_shrzgm()  # 列 '月份' 如 '201501' 及各分项
        df_cpi = ak.macro_china_cpi_monthly()  # 列 '日期' 如 'YYYY-MM-DD','今值'
    except Exception as e:
        app.logger.error(f"Error fetching macro data from akshare: {e}")
        return f"<p class='text-red-500'>Error fetching macro data: {e}</p>"

    # --- 2. 解析 DataFrame，添加年份、月份列 ---
    # 2.1 PMI（月度）："月份" 格式 "2022年10月份"
    def parse_china_pmi(df):
        df2 = df.copy()
        if '月份' in df2.columns:
            # 提取“YYYY年MM月份”中的年份和月份
            # 可能值如 "2022年10月份"
            extracted = df2['月份'].astype(str).str.extract(r'(\d{4})年\s*(\d{1,2})')
            # 创建日期列，设为当月第一天
            df2['_year'] = pd.to_numeric(extracted[0], errors='coerce')
            df2['_month'] = pd.to_numeric(extracted[1], errors='coerce')
            df2['_dt'] = pd.to_datetime(dict(year=df2['_year'], month=df2['_month'], day=1), errors='coerce')
        else:
            df2['_dt'] = pd.NaT
        df2 = df2.dropna(subset=['_dt'])
        df2['年份'] = df2['_dt'].dt.year
        df2['月份_int'] = df2['_dt'].dt.month
        return df2

    df_pmi = parse_china_pmi(df_pmi)
    # PMI 列名可能为 '制造业-指数' 或 '制造业-同比增长'，根据需要选择；这里用“制造业-指数”作示例
    pmi_col = next((col for col in df_pmi.columns if '制造业-指数' in col), None)
    if not pmi_col:
        app.logger.error("PMI DataFrame 中未找到“制造业-指数”列")
        df_pmi['制造业-指数'] = pd.NA
        pmi_col = '制造业-指数'
    # 同时非制造业指数，可用 '非制造业-指数'
    non_index_col = next((col for col in df_pmi.columns if '非制造业-指数' in col), None)

    # 2.2 非制造业 PMI（月度）: df_non_pmi 有“日期”列
    def parse_non_man_pmi(df):
        df2 = df.copy()
        date_col = next((c for c in df2.columns if '日期' in c or '时间' in c), None)
        if date_col:
            df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
        else:
            df2['_dt'] = pd.NaT
        df2 = df2.dropna(subset=['_dt'])
        df2['年份'] = df2['_dt'].dt.year
        df2['月份_int'] = df2['_dt'].dt.month
        return df2

    df_non_pmi = parse_non_man_pmi(df_non_pmi)
    # 非制造业 PMI 值列通常为 '今值'
    non_pmi_col = next((col for col in df_non_pmi.columns if '今值' in col), None)
    if not non_pmi_col:
        app.logger.error("非制造业 PMI DataFrame 中未找到“今值”列")
        df_non_pmi['今值'] = pd.NA
        non_pmi_col = '今值'

    # 2.3 CPI 同比（月度）
    def parse_cpi(df):
        df2 = df.copy()
        date_col = next((c for c in df2.columns if '日期' in c or '时间' in c or '统计月份' in c), None)
        if date_col:
            df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
        else:
            df2['_dt'] = pd.NaT
        df2 = df2.dropna(subset=['_dt'])
        df2['年份'] = df2['_dt'].dt.year
        df2['月份_int'] = df2['_dt'].dt.month
        return df2

    df_cpi = parse_cpi(df_cpi)
    # CPI 同比列通常为 '今值' 或含“同比”
    value_col_cpi = next((col for col in df_cpi.columns if '同比' in col or col == '今值'), None)
    if not value_col_cpi:
        app.logger.error("CPI DataFrame 中未找到“同比”或“今值”列")
        df_cpi['今值'] = pd.NA
        value_col_cpi = '今值'

    # 2.4 社融（月度）
    def parse_sfi(df):
        df2 = df.copy()
        date_col = next((c for c in df2.columns if '月份' == c or '月份' in c or '日期' in c), None)
        if date_col:
            # 如果格式 '201501'，则：
            df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), format='%Y%m', errors='coerce')
            # 对于 'YYYY-MM' 或 'YYYY-MM-DD' 同样适用 errors='coerce'
            if df2['_dt'].isna().all():
                df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
        else:
            df2['_dt'] = pd.NaT
        df2 = df2.dropna(subset=['_dt'])
        df2['年份'] = df2['_dt'].dt.year
        df2['月份_int'] = df2['_dt'].dt.month
        return df2

    df_sfi = parse_sfi(df_sfi)
    sfi_items = ['社会融资规模增量', '其中-人民币贷款', '其中-委托贷款', '其中-外币贷款',
                 '其中-信托贷款', '其中-未贴现银行承兑汇票', '其中-企业债券', '其中-非金融企业境内股票融资']
    existing_items = [col for col in sfi_items if col in df_sfi.columns]
    if not existing_items:
        app.logger.error("社融 DataFrame 中未找到预期分项列，实际列: %s", df_sfi.columns.tolist())

    # --- 3. 构造 total_data，确保每年月度列表长度固定 12，padding None ---
    total_data = {}
    for year in years:
        # 当年应取到 current_month，否则满12
        month_count = current_month if year == end_year else 12

        # 制造业 PMI
        pmi_year_df = df_pmi[df_pmi['年份'] == year]
        pmi_vals = []
        for m in range(1, month_count + 1):
            sub = pmi_year_df[pmi_year_df['月份_int'] == m]
            if not sub.empty:
                try:
                    v = float(sub.iloc[-1][pmi_col])
                except:
                    v = None
            else:
                v = None
            pmi_vals.append(v)
        if len(pmi_vals) < 12:
            pmi_vals += [None] * (12 - len(pmi_vals))
        total_data[f"{year}_pmi"] = pmi_vals
        app.logger.info(f"Year {year} PMI list len: {len(pmi_vals)} values: {pmi_vals}")

        # 非制造业 PMI
        non_year_df = df_non_pmi[df_non_pmi['年份'] == year]
        non_vals = []
        for m in range(1, month_count + 1):
            sub = non_year_df[non_year_df['月份_int'] == m]
            if not sub.empty:
                try:
                    v = float(sub.iloc[-1][non_pmi_col])
                except:
                    v = None
            else:
                v = None
            non_vals.append(v)
        if len(non_vals) < 12:
            non_vals += [None] * (12 - len(non_vals))
        total_data[f"{year}_pmi_non_mfg"] = non_vals
        app.logger.info(f"Year {year} NonPMI list len: {len(non_vals)} values: {non_vals}")

        # CPI 同比
        cpi_year_df = df_cpi[df_cpi['年份'] == year]
        cpi_vals = []
        for m in range(1, month_count + 1):
            sub = cpi_year_df[cpi_year_df['月份_int'] == m]
            if not sub.empty:
                try:
                    v = float(sub.iloc[-1][value_col_cpi])
                except:
                    v = None
            else:
                v = None
            cpi_vals.append(v)
        if len(cpi_vals) < 12:
            cpi_vals += [None] * (12 - len(cpi_vals))
        total_data[f"{year}_cpi"] = cpi_vals
        app.logger.info(f"Year {year} CPI list len: {len(cpi_vals)} values: {cpi_vals}")

        # 社融年度饼图：累加全年月份
        if existing_items:
            sfi_year_df = df_sfi[df_sfi['年份'] == year]
            if not sfi_year_df.empty:
                sfi_sum = sfi_year_df[existing_items].sum()
            else:
                sfi_sum = pd.Series({name: 0 for name in existing_items})

            legend_short_map = {
                "社会融资规模增量": "社融增量",
                "其中-人民币贷款": "人民币贷",
                "其中-委托贷款": "委托贷",
                "其中-外币贷款": "外币贷",
                "其中-信托贷款": "信托贷",
                "其中-未贴现银行承兑汇票": "未贴现汇票",
                "其中-企业债券": "企业债",
                "其中-非金融企业境内股票融资": "股融资"
            }

            pie_data = []
            for name in existing_items:
                try:
                    v = float(sfi_sum.get(name, 0))
                    if pd.notna(v) and v != 0:
                        label = legend_short_map.get(name, name)
                        pie_data.append([label, round(v / 1e4, 2)])  # 单位：万亿元
                except Exception as e:
                    app.logger.warning(f"处理{name}时出错: {e}")
                    continue

            total_data[f"{year}_sfi_pie"] = pie_data
        else:
            total_data[f"{year}_sfi_pie"] = []

        app.logger.info(f"Year {year} SFI pie: {total_data[f'{year}_sfi_pie']}")

    # --- 4. 生成 Timeline 可视化 ---
    months_cn = [f"{m}月" for m in range(1, 13)]

    def get_year_overlap_chart(year: int):
        raw_pmi = total_data.get(f"{year}_pmi", [])
        raw_non = total_data.get(f"{year}_pmi_non_mfg", [])
        raw_cpi = total_data.get(f"{year}_cpi", [])

        # 将 None 替换为 0
        pmi = [(v if v is not None else 0) for v in raw_pmi]
        non = [(v if v is not None else 0) for v in raw_non]
        cpi = [(v if v is not None else 0) for v in raw_cpi]
        # 再次保证长度 12
        if len(pmi) != 12 or len(non) != 12 or len(cpi) != 12:
            app.logger.warning(f"{year} lists not length 12: PMI {len(pmi)}, Non {len(non)}, CPI {len(cpi)}")
            pmi = (pmi + [0]*12)[:12]
            non = (non + [0]*12)[:12]
            cpi = (cpi + [0]*12)[:12]

        bar = (
            Bar(init_opts=opts.InitOpts(width="100%", height="600px", bg_color="#FFFFFF"))
            .add_xaxis(months_cn)
            .add_yaxis("PMI", pmi, label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("non_PMI", non, label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("CPI同比", cpi, label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{year}年 全国宏观经济指标", pos_left="center", pos_top="20px"),
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="shadow"),
                legend_opts=opts.LegendOpts(pos_left="15%", orient="horizontal", pos_top="50px"),
            )
        )

        pie_data = total_data.get(f"{year}_sfi_pie", [])
        pie = (
            Pie(init_opts=opts.InitOpts(bg_color="#FFFFFF"))
            .add(
                series_name="社融结构(万亿)",
                data_pair=pie_data,
                center=["80%", "35%"],
                radius="28%",
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{year}年社会融资结构",
                    pos_left="65%",
                    pos_top="5%",
                    title_textstyle_opts=opts.TextStyleOpts(color="#000", font_size=14)
                ))
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    is_show=True,
                    trigger="item",
                    formatter="{b}: {c} 万亿 ({d}%)"
                ),
                label_opts=opts.LabelOpts(formatter="{b}: {d}%")
            ))
        return bar.overlap(pie)

    timeline = Timeline(init_opts=opts.InitOpts(width="100%", height="600px", bg_color="#FFFFFF"))
    timeline.add_schema(is_auto_play=True, play_interval=2000)
    for y in years:
        try:
            chart = get_year_overlap_chart(year=y)
            timeline.add(chart, time_point=str(y))
        except Exception as e:
            app.logger.error(f"Year {y} chart generation error: {e}")
    # 可选：持久化 total_data
    save_json(total_data, 'macro_indices_timeline.json')
    return timeline.render_embed()


# --- Main Flask Route ---

@app.route('/contact')
@cache_by_query_param('chart_type', 'rose')
def contact_page():
    """
    Main route to display charts based on the 'chart_type' query parameter.
    Defaults to the Rose Chart (Timeline Pie).
    """
    chart_type = request.args.get('chart_type', 'rose')
    myechart = ""
    try:
        if chart_type == 'bar':
            app.logger.info("Request received for Bar Chart.")
            myechart = draw_finance_bar_chart()
        else:  # Default to rose chart
            app.logger.info("Request received for Rose Chart.")
            myechart = draw_timeline_pie_chart()
    except Exception as e:
        app.logger.critical(f"An unhandled error occurred during chart generation for '{chart_type}': {e}",
                            exc_info=True)
        myechart = f"<div class='text-center text-red-600 bg-red-100 p-4 rounded-lg'><h3>图表生成失败</h3><p>非常抱歉，在生成 {chart_type} 图表时发生了意外错误: {e}</p></div>"

    # Pass the chart HTML and the current chart type to the template
    return render_template('contact.html', myechart=myechart, chart_type=chart_type)


if __name__ == '__main__':
    app.run()
