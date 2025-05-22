import csv
import os
from collections import defaultdict
from flask_paginate import Pagination
import pymysql
from pyecharts.charts import Pie, Radar

import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for, redirect
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Line, Funnel

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
db = pymysql.connect(host="localhost", port=3306, user="root", passwd="root", db="text")
cur = db.cursor()


@app.route('/')
def root():
    cur.execute("SELECT * FROM news")
    data = cur.fetchall()
    return render_template("index.html", data=data)


@app.route("/register", methods=["POST"])  # ,methods=["GET"]
def register():
    # 获取表单数据
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    # # 检查name字段是否为空
    # if not name:
    #     return render_template("register.html", message="请输入姓名！")
    # 查询数据库，判断该用户是否已经注册
    cur.execute("SELECT * FROM users WHERE email='%s'" % email)
    result = cur.fetchall()
    if result:
        # 用户已注册，给出提示信息
        return render_template("register.html", message="该邮箱已被注册，请使用其他邮箱注册！")

    else:
        # 将用户数据插入数据库
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        db.commit()
        # 注册成功，给出提示信息
        return render_template("register.html", message="注册成功！")


@app.route('/register')
def register_page():
    return render_template("register.html")


@app.route('/login', methods=["POST"])
def login_page():
    # 获取表单数据
    email = request.form.get("email")
    password = request.form.get("password")

    # 查询数据库，判断该用户是否存在
    cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    result = cur.fetchone()
    if result:
        # 登录成功，给出提示信息
        return render_template("login.html", message="登录成功！")
    else:
        # 登录失败，给出提示信息
        return render_template("login.html", message="邮箱或密码错误，请重新登录！")



@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/index')
def index_page():
    cur.execute("SELECT * FROM news")
    data = cur.fetchall()
    return render_template("index.html", data=data)


@app.route('/about')
def datas_page():
    selected_table = request.args.get('table')  # 获取用户选择的表格名

    if selected_table == 'air_quality':
        cur.execute("SELECT * FROM region_aqi")  # 查询空气质量数据表
    elif selected_table == 'weather_hourly':
        cur.execute("SELECT * FROM weather1")  # 查询小时天气数据表
    elif selected_table == 'weather_daily':
        cur.execute("SELECT * FROM weather14")  # 查询每日天气数据表
    else:
        # 如果用户没有选择或选择了无效的表格名，默认查询空气质量数据表
        cur.execute("SELECT * FROM region_aqi")

    datas = cur.fetchall()

    pageSize = 30
    # 对获取到的数据进行切片
    page = request.args.get('page', 1, type=int)
    start = (page - 1) * pageSize  # 开始，每一页开始位置
    end = start + pageSize  # 结束，每一页结束位置
    slicontent = datas[start:end]  # 切片

    # 下面就是得到的某一页的分页对象
    current_page = Pagination(datas, page=page, per_page=pageSize, total=len(datas), items=slicontent)
    total_page = current_page.total

    context = {
        'content': slicontent,  # 获取到的数据库的数据切片
        'total_page': total_page,  # 共有几条数据
    }

    return render_template("about.html", data=slicontent, current_page=current_page, selected_table=selected_table,
                           **context)


@app.route('/services')
def services_page():
    chart_type = request.args.get('chart_type')

    if chart_type == 'next_24_hours':
        # Read data from CSV for the next 24 hours
        df = pd.read_csv("weather1.csv", encoding="utf-8")

        # Ensure hour data is sorted and converted to integers
        df['小时'] = pd.to_numeric(df['小时'])
        df = df.sort_values(by='小时')

        # Extracting data for plotting
        hour = df["小时"]
        temperature = df["温度"]
        humidity = df["相对湿度"]

        # Configure the line chart for next 24 hours
        c = (
            Line(init_opts=opts.InitOpts(bg_color="white"))
            .add_xaxis(hour.tolist())  # Convert hour to list
            .add_yaxis("Temperature", temperature.tolist(), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("Humidity", humidity.tolist(), label_opts=opts.LabelOpts(is_show=False))
            .set_series_opts(
                linestyle_opts=opts.LineStyleOpts(width=2),  # Customize the line style
                label_opts=opts.LabelOpts(is_show=True),  # Show labels for data points
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Shanghai Weather Forecast for Next 24 Hours", pos_left="center",
                                          pos_top="20px",
                                          title_textstyle_opts={"color": "black"}),
                xaxis_opts=opts.AxisOpts(name="Hours"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_right="20px", pos_top="20px")
            )
        )
        return render_template("services.html", myechart=c.render_embed(), chart_type='next_24_hours')

    elif chart_type == 'next_14_days':
        # Read data from CSV for the next 14 days
        df = pd.read_csv("weather14.csv", encoding="utf-8")

        # Extracting data for plotting
        dates = df["日期"].astype(str)  # Convert dates to strings
        min_temps = df["最低气温"]
        max_temps = df["最高气温"]

        # Configure the line chart for next 14 days
        c = (
            Line(init_opts=opts.InitOpts(bg_color="white"))
            .add_xaxis(dates.tolist())  # Convert dates to list
            .add_yaxis("Min Temperature", min_temps.tolist(), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("Max Temperature", max_temps.tolist(), label_opts=opts.LabelOpts(is_show=False))
            .set_series_opts(
                linestyle_opts=opts.LineStyleOpts(width=2),
                label_opts=opts.LabelOpts(is_show=True),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Shanghai Temperature Forecast for Next 14 Days", pos_left="center",
                                          pos_top="20px", title_textstyle_opts={"color": "black"}),
                xaxis_opts=opts.AxisOpts(name="date"),
                yaxis_opts=opts.AxisOpts(name="℃"),
                legend_opts=opts.LegendOpts( orient="vertical",pos_right="20px", pos_top="20px")
            )
        )
        return render_template("services.html", myechart=c.render_embed(), chart_type='next_14_days')

    # 如果未选择图表类型，则默认显示 next_24_hours 图表
    else:
        # Read data from CSV for the next 24 hours as default
        df = pd.read_csv("weather1.csv", encoding="utf-8")

        # Ensure hour data is sorted and converted to integers
        df['小时'] = pd.to_numeric(df['小时'])
        df = df.sort_values(by='小时')

        # Extracting data for plotting
        hour = df["小时"]
        temperature = df["温度"]
        humidity = df["相对湿度"]

        # Configure the line chart for next 24 hours
        c = (
            Line(init_opts=opts.InitOpts(bg_color="white"))
            .add_xaxis(hour.tolist())  # Convert hour to list
            .add_yaxis("Temperature", temperature.tolist(), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("Humidity", humidity.tolist(), label_opts=opts.LabelOpts(is_show=False))
            .set_series_opts(
                linestyle_opts=opts.LineStyleOpts(width=2),  # Customize the line style
                label_opts=opts.LabelOpts(is_show=True),  # Show labels for data points
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Shanghai Weather Forecast for Next 24 Hours", pos_left="center",
                                          pos_top="20px",
                                          title_textstyle_opts={"color": "black"}),
                xaxis_opts=opts.AxisOpts(name="Hours"),
                yaxis_opts=opts.AxisOpts(name=""),
                legend_opts=opts.LegendOpts(orient="vertical",pos_right="20px", pos_top="20px")
            )
        )
        return render_template("services.html", myechart=c.render_embed(), chart_type='next_24_hours')


@app.route('/portfolio')
def portfolio_page():
    # 从CSV文件中读取数据并计算平均值
    df_bj = pd.read_csv("beijing-2022.csv")
    df_sh = pd.read_csv("shanghai-2022.csv")

    # 计算北京和上海的平均值
    value_bj = df_bj.iloc[:, 2:9].mean().round(2).tolist()
    value_sh = df_sh.iloc[:, 2:9].mean().round(2).tolist()

    # 定义雷达图的指标和数据
    c_schema = [
        {"name": "AQI", "max": 80},
        {"name": "PM2.5", "max": 50},
        {"name": "PM10", "max": 90},
        {"name": "SO2", "max": 10},
        {"name": "NO2", "max": 40},
        {"name": "CO", "max": 0.9},
        {"name": "O3", "max": 90},
    ]

    # 生成雷达图
    radar = (
        Radar(init_opts=opts.InitOpts(bg_color="white"))
        .add_schema(schema=c_schema, shape="circle")
        .add("北京", [value_bj], color="#f9713c")
        .add("上海", [value_sh], color="#b3e4a1")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="2022年北京和上海平均空气质量对比",pos_left="center",
                                                    pos_top="10px", title_textstyle_opts={"color": "black"}),
                         legend_opts=opts.LegendOpts(orient="vertical",pos_right="20px", pos_top="20px"))
    )
    # 返回 HTML 模板
    return render_template("portfolio.html", myechart=radar.render_embed())


@app.route('/contact')
def contact_page():
    df = pd.read_csv("shanghai-2022.csv", encoding="utf-8",
                     names=["date", "zldj", "aqi", "pm25", "pm10", "so2", "no2", "co", "o3"])

    m = df.groupby("zldj").count()
    c = (
        Pie(init_opts=opts.InitOpts(bg_color="#E6E6E6"))
        .add("", [list(z) for z in zip(m.index, m["date"])], radius=["40%", "75%"])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="2022年上海全年空气质量总和分析", pos_left="center", pos_top="20px",
                                      title_textstyle_opts={"color": "black"}),
            legend_opts=opts.LegendOpts(orient="vertical", pos_right="100px", pos_top="20px "))
    )
    return render_template("contact.html", myechart=c.render_embed())


if __name__ == '__main__':
    app.run()
