# Weather Data Visualization and Analytical Platform

## Overview

This project is a comprehensive Weather Data Visualization and Analytical Platform designed to collect, analyze, and present weather and air quality data. It includes:

* **Data Crawlers**:

  * `weather.py`: Fetches real-time (hourly) and 14-day weather forecasts from China Meteorological Administration.
  * `crawler-news.py`: Scrapes service case news from CMA website and stores into the `news` database table.
  * `crawler-region.py`: Collects historical AQI (Air Quality Index) data for specified cities and years.

* **Database**:

  * MySQL database named `text` with tables:

    * `users` (user registration and login)
    * `news` (service case news)
    * `region_aqi` (historical AQI data)
    * `weather1` (next 24 hours weather)
    * `weather14` (next 14 days weather)

* **Web Application**:

  * Flask-based server in `app.py` providing:

    * User registration and login
    * Data display and pagination for news, AQI, hourly/daily weather
    * Interactive charts using pyecharts for:

      * 24-hour and 14-day temperature, humidity
      * Air quality radar and pie charts

## Getting Started

### Prerequisites

* Python 3.8+
* MySQL Server (create database `text`)
* pip packages:

  ```bash
  pip install requests beautifulsoup4 pymysql mysql-connector-python sqlalchemy pandas flask flask_paginate pyecharts urllib3
  ```

### Database Setup

1. Execute `create_tables.sql` to create the `text` database and necessary tables:

   ```sql
   CREATE DATABASE IF NOT EXISTS `text` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   USE `text`;
   -- run table creation statements for users, news, region_aqi, weather1, weather14
   ```
2. Adjust MySQL credentials in scripts if necessary.

### Project Structure

```
flaskProject1/
├── app.py               # Flask application
├── weather.py           # Weather crawler and DB writer
├── crawler-news.py      # CMA service case news crawler
├── crawler-region.py    # AQI data crawler
├── create_tables.sql    # SQL script for DB setup
├── templates/           # HTML templates (index, register, login, about, services, portfolio, contact)
└── static/              # Static assets (CSS, JS)
```

## Usage

### 1. Data Crawling

* **Hourly & 14-Day Weather**:

  ```bash
  python weather.py
  ```
* **Service Case News**:

  ```bash
  python crawler-news.py
  ```
* **Historical AQI Data**:

  ```bash
  python crawler-region.py
  ```

### 2. Launch Web Application

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`.

## Configuration

* **MySQL Connection**: Edit database URI in scripts (default: `root:root@localhost:3306/text`).
* **Flask Secret Key**: Automatically generated via `os.urandom(24)`; for production, set a fixed secret in environment.

## Features

* User registration/login system
* Data pagination for large datasets
* Dynamic charts (Line, Pie, Radar) with pyecharts
* Robust crawling with retry logic and error handling

## Future Improvements

* Support for proxy pools and rotating user agents
* Caching and asynchronous crawling for performance
* Deployment via Docker and CI/CD pipeline

## License

MIT License
