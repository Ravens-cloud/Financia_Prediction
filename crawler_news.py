#D:\flaskProject1\crawler_news.py
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
import logging
import re
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 建立带重试的 Session
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 模拟浏览器请求头
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://finance.caixin.com/",
})

# 目标地址
BASE_URL = "https://finance.caixin.com/"


def fetch_page(url):
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logging.error(f"请求失败: {e}")
        return None


def parse_news_list(html):
    """
    解析主页面的新闻列表
    返回字段: 图片URL、标题、时间、简介
    """
    data = []
    soup = BeautifulSoup(html, 'lxml')

    # 使用财新网实际的新闻条目类名
    items = soup.select('div.boxa')

    for item in items:
        try:
            # 提取图片URL (div.pic > a > img)
            img_tag = item.select_one('div.pic a img')
            img_url = img_tag['src'] if img_tag and img_tag.has_attr('src') else None

            # 剔除image_url为null的数据
            if not img_url:
                continue

            # 补全图片链接
            if img_url.startswith('//'):
                img_url = 'https:' + img_url

            # 提取标题 (h4 > a)
            title_tag = item.select_one('h4 a')
            title = title_tag.get_text(strip=True) if title_tag else None

            # 提取时间文本 (span)
            time_tag = item.select_one('span')
            time_text = time_tag.get_text(strip=True) if time_tag else None

            # 从时间文本中提取具体时间
            pub_time = None
            if time_text:
                # 使用正则表达式匹配时间格式: YYYY年MM月DD日 HH:MM
                match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{1,2})', time_text)
                if match:
                    pub_time = match.group(1)

            # 提取简介 (p)
            desc_tag = item.select_one('p')
            summary = desc_tag.get_text(strip=True) if desc_tag else None

            # 提取文章链接 (h4 > a)
            link_tag = item.select_one('h4 a')
            link = link_tag['href'] if link_tag and link_tag.has_attr('href') else None

            data.append({
                'image_url': img_url,
                'title': title,
                'pub_time': pub_time,
                'summary': summary,
                'article_link': link
            })
        except Exception as e:
            logging.warning(f"解析新闻项失败: {e}")
    return data

def ensure_dir(path='datas'):
    os.makedirs(path, exist_ok=True)
    return path

def save_to_json(data, filename='news.json'):
    """
    将新闻数据保存为 JSON 文件，自动过滤无 pub_time 的记录。
    参数:
        data: List[Dict] 原始新闻数据
        filename: str 保存的文件名，默认为 'news.json'
    返回:
        pd.DataFrame：处理后的 DataFrame（仅包含有效记录）
    """
    df = pd.DataFrame(data).dropna(subset=['pub_time'])

    if df.empty:
        logging.warning("没有有效数据，跳过 JSON 保存")
        return df

    path = os.path.join(ensure_dir(), filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
        logging.info(f"已保存 {len(df)} 条新闻到 JSON: {path}")
    except Exception as e:
        logging.error(f"保存 JSON 失败: {e}")

    return df


def save_to_db(df):
    if df.empty:
        logging.warning("没有有效数据保存到数据库")
        return

    try:
        # 使用 pymysql 替代 mysql-connector-python
        engine = create_engine('mysql+pymysql://root:root@localhost:3306/text?charset=utf8mb4')
        df.to_sql('news', con=engine, if_exists='replace', index=False)
        logging.info(f"已保存 {len(df)} 条新闻到数据库表 news")
    except Exception as e:
        logging.error(f"写入数据库失败: {e}")
        # 打印更详细的错误信息
        import traceback
        logging.error(traceback.format_exc())
    finally:
        if 'engine' in locals():
            engine.dispose()



def update_news():
    """
    爬取并更新 news 表。Flask 路由中可直接调用此函数。
    """
    logging.info(f"开始爬取财新网: {BASE_URL}")
    html = fetch_page(BASE_URL)
    if not html:
        logging.error("获取网页内容失败，跳过更新")
        return
    news_data = parse_news_list(html)
    if not news_data:
        logging.warning("未解析到任何新闻数据，跳过更新")
        return
    df = pd.DataFrame(news_data)
    df = df.dropna(subset=['pub_time'])
    if df.empty:
        logging.warning("无有效新闻数据，跳过更新")
        return
    save_to_json(df)  # 可选
    save_to_db(df)

# 如果直接运行，可执行更新
if __name__ == '__main__':
    update_news()