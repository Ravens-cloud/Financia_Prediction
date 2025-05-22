import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 带重试 Session
session = requests.Session()
retry = Retry(total=5, backoff_factor=1,
              status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET"])
session.mount("http://", HTTPAdapter(max_retries=retry))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                  " AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/113.0.0.0 Safari/537.36",
})

def fetch_page(url, timeout=10):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.warning(f"第一次请求超时或错误：{e}，重试宽松模式")
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e2:
            logging.error(f"宽松模式仍失败：{e2}")
            return None

def scrape_aqi(city, year):
    records = []
    for month in range(1, 13):
        url = f"http://www.tianqihoubao.com/aqi/{city}-{year}{month:02d}.html"
        html = fetch_page(url)
        if not html:
            logging.error(f"跳过{year}-{month:02d}：无法获取页面")
            continue

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            logging.warning(f"{year}-{month:02d} 无 AQI 表格，跳过")
            continue

        rows = table.find_all("tr")[1:]
        if not rows:
            logging.warning(f"{year}-{month:02d} 仅表头无数据，跳过")
            continue

        for tr in rows:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            # 10 列时：0=date,1=zldj,2=aqi,3=（冗余）,4=pm25,5=pm10,6=so2,7=no2,8=co,9=o3
            if len(cols) == 10:
                date     = cols[0]
                zldj     = cols[1]
                aqi      = int(cols[2])
                pm25     = int(cols[4])
                pm10     = int(cols[5])
                so2      = int(cols[6])
                no2      = int(cols[7])
                co       = float(cols[8])
                o3       = int(cols[9])
                records.append({
                    "date": date,
                    "zldj": zldj,
                    "aqi": aqi,
                    "pm25": pm25,
                    "pm10": pm10,
                    "so2": so2,
                    "no2": no2,
                    "co": co,
                    "o3": o3,
                })
            else:
                logging.warning(f"{year}-{month:02d}：跳过一行（{len(cols)} 列）-> {cols}")

        logging.info(f"{year}-{month:02d} 累计 {len(records)} 条记录")
        time.sleep(1)

    return records


if __name__ == "__main__":
    data = scrape_aqi("beijing", 2022)
    if data:
        df = pd.DataFrame(data)
        df.to_csv("beijing-2022.csv", index=False, encoding="utf-8")
        engine = create_engine('mysql+pymysql://root:root@localhost:3306/text')
        df.to_sql("region_aqi", con=engine, if_exists="replace", index=False)
        engine.dispose()
        logging.info("全部写入完毕")
    else:
        logging.error("未抓到任何 AQI 数据，未写入")
