#!/usr/bin/env python3
"""
检查 GCS Silver Layer 最新数据文件
"""
from google.cloud import storage
import re
from datetime import datetime

PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
GCS_CLEANED_PREFIX = "cleaned"

def extract_date_from_filename(filename):
    """从文件名中提取日期 YYYYMMDD"""
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    return None

def get_latest_files():
    """获取最新的 Silver Layer 文件"""
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        # 列出所有 cleaned/ 目录下的文件
        blobs = list(bucket.list_blobs(prefix=f"{GCS_CLEANED_PREFIX}/"))
        
        # 按类型分组
        summary_files = []
        comments_files = []
        topic_model_files = []
        
        for blob in blobs:
            if blob.name.endswith('.csv'):
                filename = blob.name.split('/')[-1]
                
                if 'daily_song_summary_' in filename:
                    date_str = extract_date_from_filename(filename)
                    if date_str:
                        summary_files.append((filename, date_str, blob.time_created))
                elif 'all_comments_' in filename and 'topic_model' not in filename:
                    date_str = extract_date_from_filename(filename)
                    if date_str:
                        comments_files.append((filename, date_str, blob.time_created))
                elif 'all_comments_topic_model_' in filename:
                    date_str = extract_date_from_filename(filename)
                    if date_str:
                        topic_model_files.append((filename, date_str, blob.time_created))
        
        print("=" * 80)
        print("GCS Silver Layer 最新文件检查")
        print("=" * 80)
        print(f"\nBucket: gs://{BUCKET_NAME}/{GCS_CLEANED_PREFIX}/")
        print(f"Project: {PROJECT_ID}\n")
        
        # Summary files
        if summary_files:
            summary_files.sort(key=lambda x: x[1], reverse=True)  # 按日期排序
            latest_summary = summary_files[0]
            print(f"📊 Summary Files (共 {len(summary_files)} 个):")
            print(f"   最新: {latest_summary[0]}")
            print(f"   日期: {latest_summary[1]} ({datetime.strptime(latest_summary[1], '%Y%m%d').strftime('%Y-%m-%d')})")
            print(f"   上传时间: {latest_summary[2]}")
            if len(summary_files) > 1:
                print(f"   其他文件: {len(summary_files) - 1} 个")
        else:
            print("📊 Summary Files: 未找到")
        
        # Comments files
        if comments_files:
            comments_files.sort(key=lambda x: x[1], reverse=True)
            latest_comments = comments_files[0]
            print(f"\n💬 Comments Files (共 {len(comments_files)} 个):")
            print(f"   最新: {latest_comments[0]}")
            print(f"   日期: {latest_comments[1]} ({datetime.strptime(latest_comments[1], '%Y%m%d').strftime('%Y-%m-%d')})")
            print(f"   上传时间: {latest_comments[2]}")
            if len(comments_files) > 1:
                print(f"   其他文件: {len(comments_files) - 1} 个")
        else:
            print("\n💬 Comments Files: 未找到")
        
        # Topic model files
        if topic_model_files:
            topic_model_files.sort(key=lambda x: x[1], reverse=True)
            latest_topic = topic_model_files[0]
            print(f"\n🤖 Topic Model Files (共 {len(topic_model_files)} 个):")
            print(f"   最新: {latest_topic[0]}")
            print(f"   日期: {latest_topic[1]} ({datetime.strptime(latest_topic[1], '%Y%m%d').strftime('%Y-%m-%d')})")
            print(f"   上传时间: {latest_topic[2]}")
            if len(topic_model_files) > 1:
                print(f"   其他文件: {len(topic_model_files) - 1} 个")
        else:
            print("\n🤖 Topic Model Files: 未找到")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_latest_files()

