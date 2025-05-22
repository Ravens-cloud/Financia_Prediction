import requests
from bs4 import BeautifulSoup
import pymysql
import pandas as pd
from sqlalchemy import create_engine

# req = requests.get("http://data.cma.cn/article/getServiceCase/cateId/9.html")
# req.encoding = "utf-8"
# html = BeautifulSoup(req.text, "lxml")
# lis = html.find_all("li", attrs={"class": "service li"})
# for v in lis:
#     wrap = v.find("div", attrs={"class": "img par"})
#     url = wrap.find("img")["src"]
#     title = v.find("div", attrs={"class": "case name"}).text
#     div = v.find_all("div", attrs={"class": "item"})
#     type = div[0].find_all("div")[1].text
#     time = div[1].find_all("div")[1].text
#     dec = v.find("div", attrs={"class": "case achievement"}).text
#     print("{}, {}, {}, {}, {}".format(url, title, type, time, dec))
#     db = pymysql.connect(host="localhost", port=3306, user="root", passwd="root", db="text")
#     cur = db.cursor()
#     cur.execute("insert into news values(null,'{}','{}','{}','{}','{}')"
#                 .format(str(url).strip(), str(title).strip(),str(type).strip(), str(time).strip(),str(dec).strip()))
#     db.commit()
import time
import logging
import requests
from bs4 import BeautifulSoup
import pymysql
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 建立带重试的 Session
session = requests.Session()
retry_strategy = Retry(
    total=5,                # 总共重试次数
    backoff_factor=1,       # 重试时间间隔依据：{backoff_factor} * (2 ** (retry_num - 1))
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 模拟常见浏览器请求头
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                  " AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/113.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "http://data.cma.cn/",
})

def fetch_service_cases(url):
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        logging.warning(f"请求失败: {e}")
        return None

def parse_and_store(html):
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")
    lis = soup.find_all("li", class_="service li")
    db = pymysql.connect(host="localhost", port=3306,
                         user="root", passwd="root", db="text")
    cur = db.cursor()
    for v in lis:
        try:
            url = v.find("div", class_="img par").img["src"]
            title = v.find("div", class_="case name").get_text(strip=True)
            divs = v.find_all("div", class_="item")
            type_ = divs[0].find_all("div")[1].get_text(strip=True)
            time_ = divs[1].find_all("div")[1].get_text(strip=True)
            desc = v.find("div", class_="case achievement").get_text(strip=True)
            sql = """
                INSERT INTO news (url, title, type, time, description)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(sql, (url, title, type_, time_, desc))
        except Exception as e:
            logging.error(f"解析或插入单条记录失败: {e}")
    db.commit()
    cur.close()
    db.close()

def main():
    target_url = "http://data.cma.cn/article/getServiceCase/cateId/9.html"
    html = fetch_service_cases(target_url)
    parse_and_store(html)

if __name__ == "__main__":
    main()

# 读取CSV文件
df = pd.read_csv('news.csv')

# 连接到MySQL数据库
engine = create_engine('mysql+mysqlconnector://root:root@localhost:3306/text')

# 将数据存储到MySQL数据库的service_cases表中
df.to_sql('news', con=engine, if_exists='replace', index=False)

# 关闭数据库连接
engine.dispose()