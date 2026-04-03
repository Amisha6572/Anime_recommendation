#!/usr/bin/env python3
"""
Poster Downloader for Anime Recommendation System
Downloads real anime posters from Jikan API v4 and saves them locally.
"""

import pandas as pd
import requests
import time
import os
from pathlib import Path

# Configuration
POSTER_DIR = "anime_recommendation"
TOP_N = 100  # Download posters for top N anime by rating
DELAY = 0.5  # Delay between API calls (seconds) to respect rate limits

def create_poster_directory():
    """Create directory for storing posters."""
    Path(POSTER_DIR).mkdir(exist_ok=True)
    print(f"✓ Created/verified directory: {POSTER_DIR}")

def fetch_poster_url(anime_name):
    """Fetch poster URL from Jikan API v4."""
    try:
        url = f"https://api.jikan.moe/v4/anime?q={requests.utils.quote(anime_name)}&limit=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['data']:
            return data['data'][0]['images']['jpg']['large_image_url']
        return None
    except Exception as e:
        print(f"  ✗ Error fetching {anime_name}: {e}")
        return None

def download_poster(anime_name, poster_url, anime_id):
    """Download and save poster image."""
    try:
        response = requests.get(poster_url, timeout=10)
        response.raise_for_status()
        
        # Save with anime_id as filename to avoid special character issues
        filename = f"{POSTER_DIR}/{anime_id}.jpg"
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"  ✓ Downloaded: {anime_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {anime_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("🎌 ANIME POSTER DOWNLOADER")
    print("=" * 60)
    
    # Create directory
    create_poster_directory()
    
    # Load anime data
    print("\n📂 Loading anime.csv...")
    try:
        anime_df = pd.read_csv("anime.csv")
        print(f"✓ Loaded {len(anime_df)} anime records")
    except FileNotFoundError:
        print("✗ Error: anime.csv not found in current directory")
        return
    
    # Get top N anime by rating
    top_anime = anime_df.nlargest(TOP_N, 'rating')
    print(f"\n🎯 Downloading posters for top {TOP_N} rated anime...")
    print(f"⏱️  Rate limit: {DELAY}s delay between requests\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, (_, row) in enumerate(top_anime.iterrows(), 1):
        anime_name = row['name']
        anime_id = row['anime_id']
        
        print(f"[{idx}/{TOP_N}] {anime_name}")
        
        # Check if already downloaded
        filename = f"{POSTER_DIR}/{anime_id}.jpg"
        if os.path.exists(filename):
            print(f"  ⊙ Already exists, skipping")
            success_count += 1
            continue
        
        # Fetch poster URL
        poster_url = fetch_poster_url(anime_name)
        if not poster_url:
            fail_count += 1
            time.sleep(DELAY)
            continue
        
        # Download poster
        if download_poster(anime_name, poster_url, anime_id):
            success_count += 1
        else:
            fail_count += 1
        
        # Rate limiting
        time.sleep(DELAY)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {success_count}/{TOP_N}")
    print(f"✗ Failed: {fail_count}/{TOP_N}")
    print(f"📁 Posters saved in: {POSTER_DIR}/")
    print("\n🎉 Done! You can now use these posters in your app.")
    print("   Update fetch_poster() in app.py to use local files first.")

if __name__ == "__main__":
    main()
