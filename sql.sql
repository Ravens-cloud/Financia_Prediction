-- 创建数据库
CREATE DATABASE IF NOT EXISTS `text`
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci;
USE `text`;

-- 用户表
CREATE TABLE `users` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) NOT NULL,
  `email` VARCHAR(100) NOT NULL UNIQUE,
  `password` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 新闻表
CREATE TABLE `news` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `url` VARCHAR(512) NOT NULL,
  `title` VARCHAR(255) NOT NULL,
  `type` VARCHAR(100) NOT NULL,
  `time` VARCHAR(100) NOT NULL,
  `description` TEXT,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 区域空气质量表
CREATE TABLE `region_aqi` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `date` DATE NOT NULL,
  `zldj` VARCHAR(50) NOT NULL COMMENT '空气质量等级',
  `aqi` INT NOT NULL,
  `pm25` INT NOT NULL,
  `pm10` INT NOT NULL,
  `so2` INT NOT NULL,
  `no2` INT NOT NULL,
  `co` DECIMAL(4,2) NOT NULL,
  `o3` INT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 小时天气表（Next 24 Hours）
CREATE TABLE `weather1` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `hour` TINYINT UNSIGNED NOT NULL COMMENT '小时（0-23）',
  `temperature` SMALLINT NOT NULL COMMENT '温度 (℃)',
  `wind_direction` VARCHAR(50) NOT NULL,
  `wind_level` TINYINT UNSIGNED NOT NULL,
  `precipitation` DECIMAL(5,2) NOT NULL COMMENT '降水量 (mm)',
  `humidity` TINYINT UNSIGNED NOT NULL COMMENT '相对湿度 (%)',
  `air_quality` TINYINT UNSIGNED COMMENT '空气质量指数',
  PRIMARY KEY (`id`),
  INDEX (`hour`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每日天气表（Next 14 Days）
CREATE TABLE `weather14` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `date` DATE NOT NULL,
  `weather` VARCHAR(100) NOT NULL COMMENT '天气描述',
  `min_temp` SMALLINT NOT NULL COMMENT '最低气温 (℃)',
  `max_temp` SMALLINT NOT NULL COMMENT '最高气温 (℃)',
  `wind_dir1` VARCHAR(50) NOT NULL,
  `wind_dir2` VARCHAR(50) NOT NULL,
  `wind_level` TINYINT UNSIGNED NOT NULL,
  PRIMARY KEY (`id`),
  INDEX (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
