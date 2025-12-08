#!/usr/bin/env python3
"""
检查 Reddit 数据情况
"""
from google.cloud import storage
import pandas as pd
import io

PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
GCS_CLEANED_PREFIX = "cleaned"

def check_reddit_data():
    """检查 Reddit 数据"""
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        print("=" * 80)
        print("Reddit 数据检查")
        print("=" * 80)
        
        # 1. 检查 Silver Layer summary 中的 reddit_comment_count
        print("\n1. 检查 Silver Layer Summary 中的 Reddit 数据:")
        print("-" * 80)
        
        # 找到最新的 summary 文件
        blobs = list(bucket.list_blobs(prefix=f"{GCS_CLEANED_PREFIX}/daily_song_summary_"))
        if blobs:
            latest_summary = max(blobs, key=lambda x: x.time_created)
            print(f"   文件: {latest_summary.name}")
            
            content = latest_summary.download_as_text()
            df_summary = pd.read_csv(io.StringIO(content))
            
            print(f"   总行数: {len(df_summary)}")
            print(f"   列名: {list(df_summary.columns)}")
            
            if 'reddit_comment_count' in df_summary.columns:
                total_reddit = df_summary['reddit_comment_count'].sum()
                non_zero = (df_summary['reddit_comment_count'] > 0).sum()
                print(f"   reddit_comment_count 总和: {total_reddit:,.0f}")
                print(f"   有 Reddit 数据的行数: {non_zero} / {len(df_summary)}")
                print(f"   占比: {non_zero/len(df_summary)*100:.1f}%")
                
                if non_zero > 0:
                    print(f"\n   有 Reddit 数据的艺术家:")
                    reddit_data = df_summary[df_summary['reddit_comment_count'] > 0]
                    for artist in reddit_data['artist'].unique()[:10]:
                        artist_total = reddit_data[reddit_data['artist'] == artist]['reddit_comment_count'].sum()
                        print(f"     - {artist}: {artist_total:,.0f} comments")
            else:
                print("   ⚠️ 没有 'reddit_comment_count' 列")
        else:
            print("   ❌ 未找到 summary 文件")
        
        # 2. 检查独立的 Reddit 数据文件
        print("\n2. 检查独立的 Reddit 数据文件:")
        print("-" * 80)
        
        reddit_blob = bucket.blob("reddit/summary/summary_all.csv")
        if reddit_blob.exists():
            print(f"   文件存在: reddit/summary/summary_all.csv")
            if reddit_blob.size:
                print(f"   大小: {reddit_blob.size / 1024 / 1024:.2f} MB")
            else:
                print(f"   大小: 未知")
            
            content = reddit_blob.download_as_text()
            df_reddit = pd.read_csv(io.StringIO(content))
            
            print(f"   总行数: {len(df_reddit)}")
            print(f"   列名: {list(df_reddit.columns)}")
            
            if 'artist' in df_reddit.columns:
                artists = df_reddit['artist'].dropna().unique()
                artists_str = [str(a) for a in artists if pd.notna(a)]
                print(f"   艺术家数量: {len(artists_str)}")
                print(f"   艺术家列表: {', '.join(artists_str[:10])}")
            
            if 'num_comments' in df_reddit.columns:
                total_comments = df_reddit['num_comments'].sum()
                print(f"   总评论数 (num_comments): {total_comments:,.0f}")
            elif 'comment_count' in df_reddit.columns:
                total_comments = df_reddit['comment_count'].sum()
                print(f"   总评论数 (comment_count): {total_comments:,.0f}")
            
            # 显示前几行
            print(f"\n   前5行数据:")
            print(df_reddit.head().to_string())
        else:
            print("   ❌ 文件不存在: reddit/summary/summary_all.csv")
        
        # 3. 检查 Silver Layer comments 文件
        print("\n3. 检查 Silver Layer Comments 文件:")
        print("-" * 80)
        
        blobs = list(bucket.list_blobs(prefix=f"{GCS_CLEANED_PREFIX}/all_comments_"))
        if blobs:
            latest_comments = max(blobs, key=lambda x: x.time_created)
            print(f"   文件: {latest_comments.name}")
            print(f"   大小: {latest_comments.size / 1024 / 1024:.2f} MB")
            
            # 只读取前几行来检查
            content = latest_comments.download_as_text()
            lines = content.split('\n')[:1000]  # 只读取前1000行
            sample_content = '\n'.join(lines)
            df_comments = pd.read_csv(io.StringIO(sample_content))
            
            print(f"   样本行数: {len(df_comments)}")
            print(f"   列名: {list(df_comments.columns)}")
            
            if 'comment' in df_comments.columns:
                print(f"   评论列存在，样本数量: {len(df_comments)}")
            
            # 检查是否有 Reddit 相关的列
            reddit_cols = [col for col in df_comments.columns if 'reddit' in col.lower()]
            if reddit_cols:
                print(f"   Reddit 相关列: {reddit_cols}")
            else:
                print(f"   ⚠️ 没有 Reddit 相关列")
        else:
            print("   ❌ 未找到 comments 文件")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_reddit_data()

