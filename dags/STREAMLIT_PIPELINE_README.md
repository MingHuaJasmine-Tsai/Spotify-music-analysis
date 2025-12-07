# Streamlit Data Pipeline DAG

## 概述

这个 Airflow DAG (`streamlit_data_pipeline_dag.py`) 自动化了为 Streamlit 仪表板准备数据的过程。

## 功能

1. **处理 YouTube 数据**
   - 从 GCS 下载原始 CSV 文件（如果需要）
   - 清洗和合并数据
   - 生成 `youtube_summary.parquet` 和 `youtube_comments.parquet`

2. **处理 Reddit 数据**
   - 从 GCS 下载原始 CSV 文件（如果需要）
   - 清洗和合并数据
   - 生成 `reddit_summary.parquet` 和 `reddit_comments.parquet`

3. **验证数据文件**
   - 检查所有必需的文件是否存在
   - 验证文件大小是否合理

4. **上传到 GCS**（可选）
   - 将处理后的 parquet 文件上传到 GCS 的 `streamlit-data/` 目录
   - 这样 Streamlit Cloud 可以从 GCS 读取最新数据

## 配置要求

### 1. GCP 连接
在 Airflow 中配置一个名为 `gcp_conn` 的连接：
- **连接类型**: Google Cloud
- **项目 ID**: `ba882-qstba-group7-fall2025`
- **密钥文件 JSON**: 包含服务账户密钥的 JSON

### 2. 项目结构
确保项目结构如下：
```
project-root/
├── dags/
│   └── streamlit_data_pipeline_dag.py
├── src/
│   └── processing/
│       ├── clean_youtube.py
│       └── clean_reddit.py
├── data/
│   └── processed/  (处理后的文件会保存在这里)
└── gcs_downloads/  (从 GCS 下载的原始文件)
    ├── summary/
    └── comments/
```

### 3. 数据源
- 原始数据应该位于 GCS bucket `apidatabase` 中
- 或者已经在本地 `gcs_downloads/` 目录中

## 调度

- **默认调度**: 每天 UTC 时间 2:00 AM（美国东部时间前一天晚上 9:00 PM）
- 可以在 DAG 定义中修改 `schedule` 参数

## 任务流程

```
[process_youtube_data] ──┐
                         ├──> [verify_streamlit_data]
[process_reddit_data] ───┘
```

两个数据处理任务并行运行，然后验证任务会检查所有文件是否已成功生成。

## 输出文件

处理完成后，以下文件将位于 `data/processed/` 目录：

- `youtube_summary.parquet`
- `youtube_comments.parquet`
- `reddit_summary.parquet`
- `reddit_comments.parquet`

如果启用了上传功能，这些文件也会被上传到 GCS 的 `streamlit-data/` 目录。

## 使用说明

### 在 Astronomer 上部署

1. 确保所有依赖已在 `requirements.txt` 中列出
2. 将代码推送到 GitHub
3. 在 Astronomer 上部署项目
4. 在 Airflow UI 中配置 GCP 连接
5. 启用 DAG

### 本地测试

```bash
# 启动 Airflow 环境
astro dev start

# 在 Airflow UI 中查看和运行 DAG
# 访问 http://localhost:8080
```

## 故障排除

### 问题：找不到清洗脚本
- **解决**: 确保 `src/processing/` 目录存在且包含 `clean_youtube.py` 和 `clean_reddit.py`

### 问题：GCP 认证失败
- **解决**: 检查 Airflow 连接 `gcp_conn` 是否正确配置

### 问题：找不到原始数据
- **解决**: 确保数据文件在 `gcs_downloads/` 目录中，或者实现从 GCS 下载数据的逻辑

### 问题：处理后的文件为空
- **解决**: 检查原始 CSV 文件是否存在且格式正确

## 下一步改进

1. 添加从 GCS 自动下载最新数据的功能
2. 添加数据质量检查（数据完整性、值域检查等）
3. 添加失败通知（邮件/Slack）
4. 添加数据备份功能
5. 可选：推送到 GitHub 以触发 Streamlit Cloud 自动重新部署




