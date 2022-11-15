/*
 Navicat Premium Data Transfer

 Source Server         : 10.96.130.69
 Source Server Type    : MySQL
 Source Server Version : 80029
 Source Host           : 10.96.130.69:3306
 Source Schema         : tt_ywdata

 Target Server Type    : MySQL
 Target Server Version : 80029
 File Encoding         : 65001

 Date: 15/11/2022 09:00:31
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for author_analysed
-- ----------------------------
DROP TABLE IF EXISTS `author_analysed`;
CREATE TABLE `author_analysed`  (
  `author` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '用户unique id elizangiaraujo',
  `author_id` bigint(0) NOT NULL COMMENT '用户数字ID ',
  `tt_number` int(0) NULL DEFAULT 0 COMMENT '分析文本条数',
  `tt_zh_number` int(0) NULL DEFAULT 0 COMMENT '清洗后文本数',
  `cluster_bad_number` int(0) NULL DEFAULT 0 COMMENT 'cluster有害数量',
  `bert_bad_number` int(0) NULL DEFAULT 0 COMMENT 'bert有害数量',
  `keyword_bad_number` int(0) NULL DEFAULT 0 COMMENT '关键词有害数量',
  `desc` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT '描述信息',
  `save_time` int(0) NULL DEFAULT NULL COMMENT '保存时间 分析完插入时间',
  `from_src` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT '来源，根据任务id可以推导',
  `score` int(0) NULL DEFAULT 0 COMMENT '综合得分 人工审核打分',
  PRIMARY KEY (`author`, `author_id`) USING BTREE,
  UNIQUE INDEX `author`(`author`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci COMMENT = '账号的分析信息' ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
