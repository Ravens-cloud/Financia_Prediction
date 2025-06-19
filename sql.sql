-- 创建数据库
CREATE DATABASE IF NOT EXISTS `text`
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci;
USE `text`;

-- 用户表
CREATE TABLE IF NOT EXISTS`users` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) NOT NULL,
  `email` VARCHAR(100) NOT NULL UNIQUE,
  `password` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 新闻表
CREATE TABLE IF NOT EXISTS`news` (
id INT AUTO_INCREMENT PRIMARY KEY,
    image_url VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    pub_time DATETIME NOT NULL,
    summary TEXT,
    article_link VARCHAR(255) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS`stock_history` (
  `symbol` VARCHAR(20) NOT NULL,
  `trade_date` DATE NOT NULL,
  `open` DOUBLE,
  `high` DOUBLE,
  `low` DOUBLE,
  `close` DOUBLE,
  `volume` BIGINT,
  `turnover` DOUBLE,
  `amplitude` DOUBLE,
  `pct_chg` DOUBLE,
  `change_amount` DOUBLE,
  `turnover_rate` DOUBLE,
  PRIMARY KEY (`symbol`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS`interest_rate` (
  `country` VARCHAR(20) NOT NULL,
  `rate_date` DATE NOT NULL,
  `value` DECIMAL(10,4) DEFAULT NULL,
  `predict` DECIMAL(10,4) DEFAULT NULL,
  `prev` DECIMAL(10,4) DEFAULT NULL,
  PRIMARY KEY (`country`, `rate_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



