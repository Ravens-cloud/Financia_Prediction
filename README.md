# Flask 数据可视化与资讯平台

一个基于 Flask + Akshare + Pyecharts + MySQL/SQLite 的数据可视化和新闻展示平台，涵盖新闻爬取、用户认证、股票历史查询、宏观/期货/基金/IPO 等多种数据获取与可视化功能。

## 数据选取与生成

本项目通过封装的 `FinancialDataEngine`（或类似模块）自动获取股票日线、行业指数、宏观经济和财务报告等多源数据。首先使用 Akshare 接口拉取指定股票的历史价格数据，并对齐行业指数、宏观指标和财务报告，通过 `pd.merge_asof` 做时点正确对齐。数据预处理包括日期解析、异常值检测与插值填充，按需剔除高缺失列。随后在 DataFrame 中计算丰富的特征：技术指标（如 MA、MACD、RSI、布林带、ATR 等）、滞后收益与滚动统计、周期性编码（月份、工作日等的正余弦变换）、成交量特征、估值比率（P/E、P/B）等，并生成未来 N 天收盘价或收益率目标。最终将清洗后的数据保存为 CSV/JSON，供模型训练使用。

## 模型训练

对生成的机器学习数据集，先进行缺失值和异常值处理（IQR 上下限检测、时序插值、中位数填充），再添加未来收益率目标列并执行特征工程，剔除因滚动与滞后带来的 NaN。使用时序划分（如 70% 训练、15% 验证、15% 测试），训练 XGBoost 回归模型预测 1/2/3 日收益率：通过 TimeSeriesSplit 进行超参随机搜索（n\_estimators、max\_depth、learning\_rate 等），在验证集上评估 MAE 和 RMSE，并在测试集上验证模型稳定性。此外可选训练基于 StandardScaler 归一化后的 MLP 回归网络，采用 EarlyStopping 与 ReduceLROnPlateau，观察训练/验证损失曲线。结果通常可达到较低的 MAE 水平，体现特征工程与时序超参调优的有效性。


## 目录结构

```
project_root/
│
├── app.py                  # Flask 主入口：初始化、路由注册
├── crawler_news.py         # 新闻爬取模块
├── charts.py               # 可选：封装 Pyecharts 图表生成逻辑
├── data_fetch.py           # 可选：封装 Akshare 数据调用与预处理
├── db_utils.py             # 可选：数据库操作、分页、缓存辅助
├── auth.py                 # 可选：用户注册/登录逻辑
├── templates/
│   ├── layout.html         # 公共布局
│   ├── index.html          # 首页新闻列表
│   ├── register.html       # 注册页面
│   ├── login.html          # 登录页面
│   ├── about.html          # “数据展示”——股票历史、利率决议
│   ├── services.html       # 多种图表展示页面（词云/漏斗/雷达/基金折线）
│   ├── portfolio.html      # K 线图页面
│   ├── contact.html        # 宏观或 IPO 图表页面
│   └── …                   # 其它模板
│
├── static/                 # 静态资源 (CSS/JS/图标等)
├── datas/                  # JSON 导出目录，如 stock_history_XXX.json 等
├── cache/                  # Flask-Caching 文件缓存目录
├── instance/               # SQLite 数据库 (Flask instance/app.db) 或 finance_data.db
│   └── app.db
├── requirements.txt        # Python 依赖
└── README.md               # 本文件
```

## 技术栈

* **语言 & 框架**
  * Python 3.7+
  * Flask: Web 框架，路由、模板渲染、session 管理
  * Jinja2: 模板引擎
  * Flask-Caching: 页面或图表缓存（filesystem/Redis）

* **数据获取**
  * Akshare: 股票、宏观、期货、基金、IPO、行业、指数等接口
  * requests + BeautifulSoup4: 自定义新闻爬虫（如财新网）

* **可视化**
  * Pyecharts: 各类图表（Kline、Line、Bar、Pie、Timeline、Radar、Funnel、WordCloud 等）
  * 前端：Bootstrap/CSS + Pyecharts embed HTML/JS

* **数据处理**
  * pandas / numpy: 表格处理、缺失值处理、日期解析
  * sklearn.preprocessing.MinMaxScaler: 归一化（风险评分示例）
  * jieba: 中文分词（词云示例）

* **数据库 & 持久化**
  * MySQL (pymysql + SQLAlchemy)：用户、新闻、股票历史、利率决议等主业务数据存储
  * SQLite (`sqlite3`, pandas.to\_sql)：轻量缓存（图表数据、本地开发调试）
  * JSON 文件：导出查询或图表数据，存于 `datas/` 目录

* **身份认证 & 安全**
  * werkzeug.security: generate\_password\_hash / check\_password\_hash
  * session 管理、输入校验（日期格式、必填字段、参数范围等）
  * SQLAlchemy text 绑定参数或 cursor.execute 参数化，防注入

* **缓存 & 定时**
  * Flask-Caching: 基于查询参数缓存图表结果，减少重复 Akshare 调用
  * 可扩展：Celery/APScheduler 定时任务预拉新闻或宏观数据

* **日志 & 错误处理**
  * Python logging / app.logger: 记录 info/warning/error/critical
  * 异常捕获：友好提示前端，详细堆栈写日志
  * Akshare 接口调用、数据库写入、文件 I/O 均有 try/except

* **前端交互**
  * HTML 表单 (`<input type="date">`, `<select>`) 提交查询参数
  * JavaScript: 监听下拉 change 事件自动提交
  * Bootstrap 分页组件：分页导航，传递 page 参数保持状态
  * Pyecharts embed 直接插入页面

* **环境 & 部署**
  * 配置通过环境变量或 config 文件，例如 `SECRET_KEY`、`DATABASE_URL`
  * WSGI 容器 (gunicorn/uWSGI) + 反向代理 (NGINX) 部署
  * 生产环境缓存可用 Redis，日志输出到文件
  * 虚拟环境 (venv/conda) 管理依赖



### SQLite

* `instance/app.db` 或 `finance_data.db`：存储图表缓存数据表，如 `funnel`, `wordcloud`, `radar`, `ipo_by_year` 等，用于快速本地查询和开发调试。

---

## 主要路由与功能

### `/` 首页

* 调用 `update_news()` 爬取/更新新闻表（可定时或首次访问时），从 MySQL `news` 读取最新若干条，渲染 `index.html`。
* session 存储 `user_id`、`user_name`，用于显示登录状态。

### `/register` & `/login`

* **注册**：POST 接收姓名、邮箱、密码，校验、哈希、写入 `users`，自动登录并 redirect。
* **登录**：POST 接收邮箱、密码，查询哈希并校验。GET 显示登录表单。

### `/about` (“数据展示”)

* Query 参数 `table`：
  * `stock_history`：输入股票代码 (`symbol`) 及 `start_date`/`end_date` (`YYYY-MM-DD`)，调用 Akshare 获取日线，转换 DataFrame，保存 JSON、写 MySQL（删除旧记录再插入），分页查询并显示。
  * `interest_rate`：选择 `country`，调用 Akshare 对应接口获取利率决议历史，解析列重命名，保存 JSON、写 MySQL（先删除该国旧记录再插入），分页查询并显示。
* 默认若无 `table`，设为 `stock_history`，渲染相应表单。
* 前端模板 `about.html` 根据 `selected_table` 控制显示哪个表单及结果表格；分页导航保留查询参数。

### `/services` (“多图表页面”)

* Query `chart_type`：
  * `wordcloud`: 从 MySQL `news` 取标题+摘要，jieba 分词，计数前 50 词，生成 Pyecharts WordCloud，保存 JSON/SQLite/MySQL。
  * `funnel`: 调用 Akshare `futures_rule` 获取近 N 交易日规则，计算风险评分、分类分级，生成 Funnel 图，保存 JSON/SQLite。
  * `radar`: 中美六大年度经济指标对比雷达图，调用 Akshare 多种宏观接口（GDP/CPI/PMI/出口/失业/PPI），年度平均或最后一期值，生成 Radar，保存 JSON/SQLite。
  * `fund_line`: Akshare `amac_member_info` 会员信息，近 5 年私募/公募新增会员折线图，保存 JSON/SQLite。
* 使用自定义缓存 `@cache_by_query_param('chart_type', ...)` 避免重复计算。
* 渲染 `services.html`，插入 `myechart` embed HTML。

### `/portfolio` (“专业 K 线图”)

* 可通过 URL 参数指定 `symbol`，或示例中硬编码某 `stock_code`。
* 调用 Akshare `stock_zh_a_hist` 获取近周期（日/前复权）日线，DataFrame 排序重命名列，生成 Pyecharts Kline 图，叠加 MA5/10/20/30 折线，渲染 `portfolio.html`。

### `/contact` (“宏观/IPO 图表”)

* Query `chart_type`：
  * `bar`: 调用 `draw_finance_bar_chart()`：近 5 年月度制造业 PMI、非制造业 PMI、CPI 同比，年度社融结构饼图；解析不同列、pad 缺失、短标签映射、Pyecharts Timeline overlap Bar+Pie。
  * 默认或 `rose`: 调用 `draw_timeline_pie_chart()`：近 5 年各板块（主板沪/深、创业板、科创板）IPO 注册生效数量 Timeline Pie 图；解析 Akshare `stock_register_*` 接口日期列与状态列。
* 异常捕获后在前端友好显示错误信息。
* 渲染 `contact.html`，插入 `myechart` embed HTML。

---

## 关键辅助函数

* **`find_column(df, candidates)`**: 在 `df.columns` 中按精确或部分匹配查找列名。
* **`calculate_ma(day_count, data)`**: 简单移动平均，用于 MA 线。
* **`save_json(data, filename)`**: 将数据保存到 `datas/filename`。
* **`to_db(df, table_name)` / `init_db_and_save`**: 将 DataFrame 保存到 SQLite。
* **`get_recent_trade_days(n)`**: 逆向遍历日期调用 `ak.futures_rule` 收集近 n 个有效交易日数据。
* **`cache_by_query_param(param_name, default_value)`**: 自定义缓存装饰器，基于 URL 参数缓存视图输出。

---

## 环境与部署

1. **依赖安装**

   ```bash
   pip install -r requirements.txt
   ```

2. **环境变量**
   * `SECRET_KEY`: Flask session 加密
   * `DATABASE_URL`: MySQL 连接字符串，例如 `mysql+pymysql://root:root@localhost:3306/text?charset=utf8mb4`

3. **MySQL 建库建表**
   执行前面 “数据库设计” 中的 SQL 建表脚本。

4. **启动**
   * 本地开发：
     ```bash
     export FLASK_APP=app.py
     export FLASK_ENV=development
     flask run
     ```
   * 生产：使用 Gunicorn/uWSGI + Nginx，设置环境变量，开启合适的并发和缓存策略。

5. **定时任务**（可选）
   * 可用 APScheduler/Celery 定时调用 `update_news()`、预拉宏观数据缓存等，减小请求延迟。

6. **日志**
   * Python `logging` 配置：生产可调整日志级别并输出到文件。
   * 在 Flask 中通过 `app.logger` 记录关键事件。

---

## 运行流程简述

1. **用户访问首页 `/`**：
   * 调用 `update_news()` 爬取财新网新闻，存 MySQL 表 `news`；
   * 读取最新记录，渲染 `index.html`；
   * 若已登录显示用户名。

2. **数据展示 `/about?table=...`**：
   * `table=stock_history`: 输入股票代码和日期范围，后台检查 MySQL 中是否已有对应数据；若无则调用 Akshare 获取、保存 JSON、插入 MySQL，再分页查询并显示；若已有则直接分页查询显示。
   * `table=interest_rate&country=...`: 选择国家，调用 Akshare 获取利率决议历史，解析、保存 JSON、插入 MySQL，再分页查询并显示。
   * 模板 `<select>` 与 `<input type="date">` 实现交互；分页导航保留参数。

3. **图表页面 `/services?chart_type=...`**：
   * 根据类型调用相应生成函数，进行 Akshare 或爬虫调用、pandas 处理、保存 JSON/SQLite/MySQL、Pyecharts 渲染。
   * 缓存装饰器避免重复调用；前端展示 embed 图表。

4. **K 线图 `/portfolio`**：
   * 获取指定或示例 `symbol` 的日线数据，绘制 K 线 + MA 折线图，渲染页面。

5. **宏观/IPO 图表 `/contact?chart_type=bar|rose`**：
   * 生成 Timeline Pie 或 Bar+Pie 混合图：近 5 年宏观指标或 IPO 板块数量，渲染页面。

---

## 注意事项

* **Akshare 接口更新**：列名或格式可能变化，应及时调整数据解析逻辑（如 `find_column`、正则处理等）。
* **异常与提示**：捕获所有外部接口异常并记录日志，前端显示简明错误；避免抛栈直接影响用户体验。
* **性能**：
  * 对大型 DataFrame 写库、频繁 Akshare 调用应缓存或后台预拉；
  * Flask-Caching、本地 SQLite 缓存图表数据；
  * 并发场景下使用 MySQL 替代 SQLite。
* **安全**：
  * 密码哈希存储；
  * SQL 参数化；
  * 输入校验（symbol 格式、日期格式、country 列表）；
  * 固定并保护 `SECRET_KEY`；
  * 日志不泄露敏感信息。
* **前端体验**：
  * 使用 `<input type="date">` 实现日期选择；
  * 下拉菜单自动提交或手动提交；
  * 分页导航显示当前页、总页数；
  * 图表加载时可显示“加载中”提示（可选 JS 优化）；
  * 响应式布局适配不同屏幕。
* **测试**：
  * 单元测试数据解析、分页逻辑、辅助函数；
  * 对爬虫解析逻辑使用示例 HTML 进行测试；
  * Mock Akshare 返回小样本 DataFrame 测试图表函数。
* **部署**：
  * WSGI + Nginx，环境变量管理；
  * 缓存与数据库配置；
  * HTTPS、日志管理、监控。

---

## 快速上手

1. 克隆项目：
   ```bash
   git clone <repo_url>
   cd project_root
   ```
2. 创建并激活虚拟环境，安装依赖：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```
3. 配置环境变量或直接修改 `app.config`：
   * `SECRET_KEY`
   * `DATABASE_URL`（MySQL 连接）
4. 在 MySQL 中执行建库建表 SQL。
5. 启动 Flask：
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
   ```
6. 浏览器访问：
   * `/`：首页新闻列表
   * `/register`, `/login`：用户认证
   * `/about?table=stock_history`：股票历史查询
   * `/about?table=interest_rate&country=china`：利率决议
   * `/services?chart_type=wordcloud|funnel|radar|fund_line`：图表展示
   * `/portfolio`：K 线图
   * `/contact?chart_type=bar|rose`：宏观/IPO 图表

---

## 项目改进建议

* **日志管理**：将 `print` 替换为 `logging`，区分日志级别并输出到文件或监控系统。
* **接口缓存**：对不常变动的 Akshare 接口结果（如行业板块、宏观全量历史）做本地缓存，减少网络请求。
* **配置化**：将各种常量（路径、重试次数、默认值、列名映射、阈值等）集中到配置文件或环境变量，方便维护。
* **模块化**：将数据获取、特征工程、图表生成等拆分为独立模块，便于单元测试和复用。
* **异步/后台任务**：将耗时较长的 Akshare 调用或 ML 数据准备放到后台任务（Celery/RQ），保证 Web 请求响应及时，并可通过任务状态查询。
* **测试覆盖**：为关键逻辑编写单元测试／集成测试，模拟 Akshare 返回数据验证解析和特征计算。
* **前端优化**：显示加载动画；使用 AJAX 请求图表嵌入，提高用户体验；适配移动端。
* **安全加固**：严格校验所有用户输入、保护敏感配置、HTTPS 部署，并定期更新依赖。
* **监控与报警**：记录重要操作耗时、失败率，结合 Prometheus/Grafana 或日志告警，及时发现异常。
* **扩展更多数据源**：接入第三方 API、社交舆情数据、更多财务/行业指标等，丰富平台功能。
* **批量处理**：支持批量股票数据获取与分析，结合并行/分布式，满足更大规模需求。

---

## 总结

本项目示例展示如何结合 Flask、Akshare、Pyecharts、pandas、SQLAlchemy 等技术，实现一个集新闻爬取、用户管理、数据库存储、动态数据查询与丰富交互式图表展示于一体的平台。提供多种示例路由和图表类型，便于快速原型开发与二次扩展。按照上述架构和建议，可进一步拆分模块、增强缓存和异步能力、完善测试与监控，打造高可用、高性能的数据可视化服务。

祝开发顺利！