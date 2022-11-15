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

 Date: 15/11/2022 09:00:16
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for author
-- ----------------------------
DROP TABLE IF EXISTS `author`;
CREATE TABLE `author`  (
  `author` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT 'unique id elizangiaraujo',
  `author_id` bigint(0) NOT NULL COMMENT '用户数字ID 6672849546949706757',
  `nickname` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `avatar_larger` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '头像',
  `avatar_src_url` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '头像链接',
  `signature` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '个人说明',
  `verified` tinyint(0) NULL DEFAULT NULL COMMENT '是否验证',
  `sec_uid` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `follower_count` bigint(0) NULL DEFAULT NULL COMMENT '粉丝',
  `following_count` bigint(0) NULL DEFAULT NULL COMMENT '关注',
  `heart_count` bigint(0) NULL DEFAULT NULL COMMENT '累计被点赞',
  `video_count` bigint(0) NULL DEFAULT NULL COMMENT '发布视频数',
  `digg_count` bigint(0) NULL DEFAULT NULL COMMENT '累计点赞数',
  `open_favorite` tinyint(0) NULL DEFAULT NULL COMMENT '是否开放收藏',
  `private_account` tinyint(0) NULL DEFAULT NULL COMMENT '是否私有账号',
  `created_at` bigint(0) NULL DEFAULT NULL,
  `updated_at` bigint(0) NULL DEFAULT NULL,
  `flag` int(0) NULL DEFAULT 0 COMMENT '标识',
  `task_id` bigint(0) NULL DEFAULT NULL COMMENT '任务id',
  `analysed` int(0) NULL DEFAULT 0 COMMENT '1:已分析，0未分析，分析结果再author_analyse表中',
  `from_src` int(0) NULL DEFAULT NULL COMMENT '0：原库；1:关注列表拓展; 2：关键词拓展；3：评论拓展',
  PRIMARY KEY (`author_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
