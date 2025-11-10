
import os
import json
import csv
import time
import math
import base64
import random
import pathlib
import datetime
import traceback
from typing import Any, Dict, List, Optional

import requests

# ========================
# ⚠️ 建议：把密钥放到环境变量更安全
# EN: For security, consider moving these to env vars: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
# ========================
CLIENT_ID     = "9cf5a71c9e414abb88fe1d0ff28d4123"
CLIENT_SECRET = "3a2823b9274e4706b7d0c7eb2a8e820e"

# Market influences search & availability
MARKET = "US"

# ========================
# 目标歌曲 / Target songs (10)
# ========================
TARGET_SONGS: List[Dict[str, str]] = [
    {"title": "The Fate of Ophelia", "artist": "Taylor Swift"},
    {"title": "Camera",              "artist": "Ed Sheeran"},
    {"title": "Gorgeous",            "artist": "Doja Cat"},
    {"title": "Kiss",                "artist": "Demi Lovato"},
    {"title": "DEPRESSED",           "artist": "Anne-Marie"},
    {"title": "bittersweet",         "artist": "Madison Beer"},
    {"title": "Safe",                "artist": "Cardi B"},  # feat. Kehlani（主艺人用 Cardi B 更稳）
    {"title": "Artificial Angels",   "artist": "Grimes"},
    {"title": "CHANEL",              "artist": "Tyla"},
    {"title": "Dracula",             "artist": "Tame Impala"},
]

# Output base directory
BASE_OUT_DIR = pathlib.Path("data")

# Max retries for transient failures
MAX_RETRIES = 3


#  Retry helpers ==========

def _sleep_with_jitter(seconds: float) -> None:
    """Sleep with jitter to avoid thundering herd."""
    time.sleep(seconds * (0.9 + 0.2 * random.random()))

def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 15.0) -> float:
    """Exponential backoff delay with cap (秒)."""
    return min(cap, base * (2 ** attempt))

def robust_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> requests.Response:
    """
    对 Spotify API 的稳健请求：支持 429/5xx 重试，保留 4xx 作为硬失败。
    Robust HTTP request with retries for 429/5xx. Raises on persistent failure.
    """
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.request(method, url, headers=headers, params=params, data=data, timeout=timeout)

        # 429 限速：按 Retry-After 或指数退避
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else _backoff_delay(attempt)
            print(f"[RATE LIMIT] 429 received. Waiting {wait_s:.1f}s ...")
            _sleep_with_jitter(wait_s)
            continue

        # 5xx：可重试
        if 500 <= resp.status_code < 600:
            if attempt < MAX_RETRIES:
                wait_s = _backoff_delay(attempt)
                print(f"[RETRY] {resp.status_code} Server error. Retry in {wait_s:.1f}s ...")
                _sleep_with_jitter(wait_s)
                continue

        # 其他：直接返回（让上层决定如何处理 4xx）
        return resp

    # 理论上到不了这里
    return resp


# ========== 2) 授权 / Authorization (Client Credentials) ==========

def get_access_token(client_id: str, client_secret: str) -> str:
    """
    获取 Bearer Token（有效期约 1 小时）
    Get a Bearer token (valid ~1 hour)
    """
    url = "https://accounts.spotify.com/api/token"
    auth_str = f"{client_id}:{client_secret}".encode("utf-8")
    headers = {"Authorization": "Basic " + base64.b64encode(auth_str).decode("utf-8")}
    data = {"grant_type": "client_credentials"}

    resp = robust_request("POST", url, headers=headers, data=data, params=None)
    if resp.status_code != 200:
        raise requests.HTTPError(f"Token error {resp.status_code}: {resp.text}", response=resp)

    js = resp.json()
    return js["access_token"]


# ========== 3) API 封装 / API wrappers ==========

def api_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def search_track(token: str, title: str, artist: Optional[str], market: str, limit: int = 10) -> List[Dict[str, Any]]:
    q = f'track:"{title}"'
    if artist:
        q += f' artist:"{artist}"'

    url = "https://api.spotify.com/v1/search"
    params = {"q": q, "type": "track", "limit": limit, "market": market}

    resp = robust_request("GET", url, headers=api_headers(token), params=params)
    if resp.status_code != 200:
        raise requests.HTTPError(f"search_track {resp.status_code}: {resp.text}", response=resp)
    return resp.json().get("tracks", {}).get("items", [])

def search_track_with_fallback(token: str, title: str, artist: Optional[str], market: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    先按标题+艺人搜索；若无结果且提供了艺人，则退化为仅标题搜索。
    EN: Try title+artist first; if empty and artist provided, retry with title only.
    """
    items = search_track(token, title, artist, market, limit)
    if not items and artist:
        print("[FALLBACK] retry with title only...")
        items = search_track(token, title, None, market, limit)
    return items

def get_track(token: str, track_id: str, market: str) -> Dict[str, Any]:
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    params = {"market": market}
    resp = robust_request("GET", url, headers=api_headers(token), params=params)
    if resp.status_code != 200:
        raise requests.HTTPError(f"get_track {resp.status_code}: {resp.text}", response=resp)
    return resp.json()

def get_audio_features(token: str, track_id: str) -> Dict[str, Any]:
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    resp = robust_request("GET", url, headers=api_headers(token))
    if resp.status_code != 200:
        raise requests.HTTPError(f"get_audio_features {resp.status_code}: {resp.text}", response=resp)
    return resp.json()

def get_artist(token: str, artist_id: str) -> Dict[str, Any]:
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    resp = robust_request("GET", url, headers=api_headers(token))
    if resp.status_code != 200:
        raise requests.HTTPError(f"get_artist {resp.status_code}: {resp.text}", response=resp)
    return resp.json()


# ========== 4) 匹配评分 / Heuristic scoring for best match ==========

def score_candidate(item: Dict[str, Any], want_title: str, want_artist: str) -> int:
    """
    粗略打分：标题匹配+艺人匹配-惩罚（remix/live 等）
    Heuristic scoring to pick the most likely track.
    """
    title = (item.get("name") or "").lower()
    artists = " ".join(a.get("name", "") for a in item.get("artists", [])).lower()

    score = 0
    # 完全标题命中额外加分 / exact match bonus
    if title == want_title.lower():
        score += 4
    if want_title.lower() in title:
        score += 3
    if want_artist and want_artist.lower() in artists:
        score += 3

    bad_tokens = ["live", "remix", "sped up", "slowed", "instrumental", "acoustic"]
    if any(tok in title for tok in bad_tokens):
        score -= 1

    # 额外加权：主艺人完全相等时再加 1
    if want_artist and any(a.get("name", "").lower() == want_artist.lower() for a in item.get("artists", [])):
        score += 1

    return score


# ========== 5) 文件与落盘 / File IO helpers ==========

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def append_daily_csv(row: Dict[str, Any], csv_path: pathlib.Path) -> None:
    header = list(row.keys())
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ========== 6) 安全调用包装（失败跳过）/ Safe call wrapper ==========

def safe_get(desc: str, func, *args, **kwargs):
    """
    调用 func(*args, **kwargs)。失败时打印+返回 None，不中断主循环。
    Call and return func(*args, **kwargs). On failure, log and return None.
    """
    try:
        return func(*args, **kwargs)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "NA")
        text = ""
        try:
            text = e.response.text[:240]
        except Exception:
            pass
        print(f"[SKIP] {desc} failed: HTTP {code} | {text}")
        return None
    except Exception as e:
        print(f"[SKIP] {desc} failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)
        return None


# ========== 7) 主流程 / Main pipeline ==========

def main():
    # 0) 参数检查 / sanity check
    assert CLIENT_ID and CLIENT_SECRET and CLIENT_ID != "YOUR_CLIENT_ID_HERE", \
        "请先填写 CLIENT_ID/CLIENT_SECRET，或设置环境变量 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET"

    # 1) 获取 Token（Client Credentials）
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    print("[OK] Access token acquired.")

    # 2) 输出目录（按天）
    today = datetime.date.today().isoformat()
    out_dir = BASE_OUT_DIR / today
    ensure_dir(out_dir)

    # 3) 循环每首目标歌
    for target in TARGET_SONGS:
        title  = target.get("title", "").strip()
        artist = target.get("artist", "").strip()
        print(f"\n==> Searching: {title} | {artist or '(unknown artist)'}")

        # 3.1 搜索（先标题+艺人，失败fallback仅标题）
        candidates = safe_get("search_track", search_track_with_fallback, token, title, artist or None, MARKET, 10)
        if not candidates:
            print("[SKIP] No candidates. Go next track.")
            continue

        # 3.2 选择最像的一条
        best = max(candidates, key=lambda it: score_candidate(it, title, artist))
        track_id = best.get("id")
        track_name = best.get("name", "")
        artist_names = ", ".join(a.get("name", "") for a in best.get("artists", []))
        print(f"Picked: {track_name} — {artist_names} | id={track_id}")

        if not track_id:
            print("[SKIP] Missing track id; go next.")
            continue

        # 3.3 拉取 track 详情（若失败，整首跳过）
        track = safe_get("get_track", get_track, token, track_id, MARKET)
        if not track:
            print("[SKIP] Track details unavailable; go next.")
            continue

        # 3.4 音频特征（若失败，仅跳过该步骤）
        features = safe_get("get_audio_features", get_audio_features, token, track_id) or {}

        # 3.5 艺人信息（若失败，仅跳过该步骤）
        artist_info = {}
        try:
            main_artist_id = track.get("artists", [{}])[0].get("id")
        except Exception:
            main_artist_id = None

        if main_artist_id:
            tmp = safe_get("get_artist", get_artist, token, main_artist_id)
            if tmp:
                artist_info = tmp

        # 3.6 写出 JSON（只要拿到 track 就会写）
        payload = {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "market": MARKET,
            "search_title": title,
            "search_artist": artist or None,
            "track": track,
            "audio_features": features or None,
            "artist": artist_info or None,
        }
        json_path = out_dir / f"{track_id}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] JSON -> {json_path}")

        # 3.7 关键字段汇总 → 追加到 CSV（缺失字段留空）
        row = {
            "date": today,
            "track_id": track_id,
            "track_name": track.get("name", ""),
            "artist_name": "; ".join(a.get("name", "") for a in track.get("artists", []) or []),
            "album_name": track.get("album", {}).get("name", ""),
            "release_date": track.get("album", {}).get("release_date", ""),
            "popularity": track.get("popularity", ""),
            "duration_ms": track.get("duration_ms", ""),
            "explicit": track.get("explicit", ""),
            "available_markets_count": len(track.get("available_markets", []) or []),
            # audio features (may be blank)
            "danceability": features.get("danceability", "") if features else "",
            "energy": features.get("energy", "") if features else "",
            "valence": features.get("valence", "") if features else "",
            "tempo": features.get("tempo", "") if features else "",
            "speechiness": features.get("speechiness", "") if features else "",
            "acousticness": features.get("acousticness", "") if features else "",
            "instrumentalness": features.get("instrumentalness", "") if features else "",
            "liveness": features.get("liveness", "") if features else "",
            # artist (may be blank)
            "artist_followers": artist_info.get("followers", {}).get("total", "") if artist_info else "",
            "artist_genres": "; ".join(artist_info.get("genres", []) or []) if artist_info else "",
        }
        append_daily_csv(row, BASE_OUT_DIR / "daily_tracks.csv")
        print("[APPEND] data/daily_tracks.csv updated.")

    print("\n[DONE] All targets processed.")


# ========== 8) 入口 / Entrypoint ==========

if __name__ == "__main__":
    main()
