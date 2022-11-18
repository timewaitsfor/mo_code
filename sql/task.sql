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

 Date: 18/11/2022 10:41:10
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for task
-- ----------------------------
DROP TABLE IF EXISTS `task`;
CREATE TABLE `task`  (
  `id` bigint(0) NULL DEFAULT 0,
  `value` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL DEFAULT '' COMMENT '关键词，或账号',
  `type` int(0) NOT NULL DEFAULT 1 COMMENT '0：关键词，1账号时间线，2粉丝,3关注，4评论',
  `website` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT 'tiktok',
  `d_flag` int(0) NULL DEFAULT 0 COMMENT '0未删除，1已删除',
  `get_flag` int(0) NULL DEFAULT 0 COMMENT '是否已下任务，1已下；0未下 ',
  `exec_flag` int(0) NULL DEFAULT 0 COMMENT '是否执行完毕，0未下发；1正在执行；2执行完毕；3终止',
  `create_time` bigint(0) NULL DEFAULT NULL COMMENT '创建时间',
  `start_time` bigint(0) NULL DEFAULT NULL COMMENT '开始时间',
  `end_time` bigint(0) NULL DEFAULT NULL COMMENT '结束时间',
  `update_time` bigint(0) NULL DEFAULT NULL COMMENT '更新时间',
  `period_type` int(0) NULL DEFAULT 0 COMMENT '0:once,1:period',
  `period` int(0) NULL DEFAULT 0 COMMENT '周期，秒',
  `host` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT '执行的服务器地址',
  `image_state` int(0) NULL DEFAULT 0 COMMENT '是否下载视频封面 0否 1是 ',
  `get_comment` int(0) NULL DEFAULT 0 COMMENT '是否下评论任务，1已下，0未下',
  PRIMARY KEY (`value`, `type`) USING BTREE,
  UNIQUE INDEX `value_type`(`value`, `type`) USING BTREE,
  INDEX `d_type`(`type`, `d_flag`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci COMMENT = '任务表' ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
