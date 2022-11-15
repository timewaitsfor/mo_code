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

 Date: 15/11/2022 09:00:07
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for content
-- ----------------------------
DROP TABLE IF EXISTS `content`;
CREATE TABLE `content`  (
  `id` bigint(0) NOT NULL COMMENT '视频描述信息/评论内容/OCR/回复评论内容ID',
  `author` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '用户unique id elizangiaraujo',
  `author_id` bigint(0) NULL DEFAULT NULL COMMENT '用户数字ID 6672849546949706757',
  `tt_text` varchar(4096) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '视频描述信息/评论内容/OCR/回复评论内容/语音转文本',
  `tt_type` int(0) NOT NULL COMMENT '0：视频描述，1：评论，2：回复，3：OCR；4：语音转文本',
  `clean_txt` varchar(4096) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT '清洗后推文',
  `publish_date` int(0) NULL DEFAULT NULL COMMENT '发布时间',
  `save_time` int(0) NULL DEFAULT NULL COMMENT '保存时间',
  `video_id` bigint(0) NOT NULL COMMENT '视频id',
  `video_author` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `video_author_id` bigint(0) NULL DEFAULT NULL,
  `video_src_url` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '视频访问链接',
  `music_playUrl` text CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT '视频配乐url',
  `music_title` varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL COMMENT '视频配乐名称',
  `stats_diggCount` bigint(0) NULL DEFAULT NULL COMMENT '点赞量',
  `stats_shareCount` bigint(0) NULL DEFAULT NULL COMMENT '分享量',
  `stats_commentCount` bigint(0) NULL DEFAULT NULL COMMENT '视频评论量',
  `stats_playCount` bigint(0) NULL DEFAULT NULL COMMENT '视频播放量',
  `created_at` bigint(0) NULL DEFAULT NULL COMMENT '入库时间',
  `updated_at` bigint(0) NULL DEFAULT NULL COMMENT '更新时间',
  `ocr_result` text CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'OCR识别结果',
  `face_result` text CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT '人脸识别结果',
  `ocr_flag` int(0) NULL DEFAULT NULL COMMENT '0：无害 1：有害',
  `face_flag` int(0) NULL DEFAULT NULL COMMENT '0：无害 1：有害',
  `task_id` bigint(0) NULL DEFAULT NULL COMMENT '任务id',
  `bert_score` float(6, 3) NULL DEFAULT 0.000 COMMENT 'bert打分',
  `cluster` json NULL COMMENT '聚类打分',
  PRIMARY KEY (`id`, `tt_type`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
