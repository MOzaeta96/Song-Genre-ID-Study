from __future__ import annotations

import os
from typing import Any
import requests
import pandas as pd

LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"


def _call_lastfm(params: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise ValueError("LASTFM_API_KEY is not set. Add it to your environment or .env file.")

    response = requests.get(
        LASTFM_BASE_URL,
        params={**params, "api_key": api_key, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_track_tags(artist: str, track: str) -> list[str]:
    payload = _call_lastfm({"method": "track.getTopTags", "artist": artist, "track": track})
    tags = payload.get("toptags", {}).get("tag", [])
    return [tag.get("name", "").strip().lower() for tag in tags if tag.get("name")]


def get_artist_tags(artist: str) -> list[str]:
    payload = _call_lastfm({"method": "artist.getTopTags", "artist": artist})
    tags = payload.get("toptags", {}).get("tag", [])
    return [tag.get("name", "").strip().lower() for tag in tags if tag.get("name")]


def enrich_with_tags(df: pd.DataFrame, artist_col: str = "artist", track_col: str = "song") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        artist = row[artist_col]
        track = row[track_col]
        track_tags = get_track_tags(artist, track)
        artist_tags = [] if track_tags else get_artist_tags(artist)
        rows.append({
            **row.to_dict(),
            "track_tags": track_tags,
            "artist_tags": artist_tags,
            "tag_source": "track" if track_tags else ("artist" if artist_tags else "none"),
        })
    return pd.DataFrame(rows)
