# Spotify-music-analysis
This repository is for spotify marketing analysis project use, created bt Team 7 at class BA882

Overview

This project — Music Trends Analytics — aims to analyze October Spotify new song releases through a unified, automated data pipeline integrating Spotify, YouTube, and Reddit data.
We focus on understanding how social buzz and streaming performance interact, and how this insight can enhance music marketing strategies.

🧩 A. Business Problems

Our project explores how social and streaming data can inform marketing strategy in the music industry:

Cross-platform Influence
→ How does social buzz (Reddit, YouTube) impact Spotify streaming performance?

Trend Prediction
→ Can Reddit discussions or YouTube engagement forecast Spotify popularity?
→ Through text mining, how can SEO/GEO insights help music promotion?

Campaign Optimization
→ When is the best time to launch or promote a song for maximum impact?

🎯 B. Objective

To build a unified, automated data pipeline and analytics dashboard that:

Integrates data from Spotify, YouTube, and Reddit

Refreshes daily to ensure data timeliness

Provides actionable insights for:

Music marketing strategy

Artist promotion

Trend forecasting

🔗 C. Data Feeds
Source	API	Key Data	Purpose
🎧 Spotify Web API	Spotify Developer API
	Popularity score, followers, audio features	Measure streaming performance
📺 YouTube Data API	YouTube Data API
	Views, likes, comments	Assess content engagement & virality
💬 Reddit API	Reddit API
	Posts & comments about artists/songs	Perform text mining, sentiment & topic analysis

⚙️ All APIs are refreshed daily via automated Airflow DAGs.

🧱 D. Data Pipeline Overview

API Extraction → Raw data fetched from Spotify, YouTube, Reddit

Airflow DAG → Orchestrates daily ETL job (scheduled in GCP Composer)

Data Warehouse (BigQuery) → Cleansed & transformed data

ML Layer (optional) → Sentiment scoring, trend modeling

Dashboard (Streamlit) → Visual insights for music marketing decisions

📁 Example directory structure:

src/
 ├── api_fetch/
 ├── airflow_dags/
 ├── ml_model/
 ├── dashboard/
 ├── utils/
data/
 ├── raw/
 ├── processed/

🧮 E. Data Warehouse (GCP – BigQuery)

The centralized data warehouse:

Performs SQL-based transformation and aggregation

Enriches tables with:

Sentiment scores (using VADER)

Keyword extraction

Engagement metrics (views, likes, sentiment ratios)

Daily insert jobs maintain consistent and up-to-date tables

🧠 F. Data Model (Simplified)
Table	Description
spotify_tracks	Track-level metrics from Spotify
youtube_engagement	Engagement stats per artist/video
reddit_posts	Post- and comment-level text data
artist_sentiment_daily	Aggregated sentiment by artist/day
music_trends_summary	Combined performance indicators

📊 G. Reporting & Visualization

Developed in Streamlit (Phase 1) with integration to Looker Studio (future).

Dashboard Features

Audience Sentiment Over Time (by Artist)
Track day-to-day sentiment changes to identify reaction shifts.

Engagement vs Sentiment (Artist Comparison)
Compare fan positivity vs engagement volume across artists.

Daily Discussion Volume (by Artist)
Identify peaks in attention that align with releases or viral events.

Upcoming Enhancements

🔍 Topic Analysis – Keyword and topic modeling for Reddit discussions

📈 Engagement Correlation – Visual correlation of Reddit activity vs YouTube views

☁️ H. Deployment (GCP Composer)

Pipeline deployed via Google Cloud Composer (Airflow)

Scheduled for daily runs

Scales up to track ~15 songs concurrently

Data outputs automatically refreshed to BigQuery → Streamlit Dashboard
