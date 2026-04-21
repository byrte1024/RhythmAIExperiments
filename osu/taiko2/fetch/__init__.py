"""osu! API fetchers: star ratings and other per-beatmap metadata."""
from .client import OsuV2Client
from .stars import fetch_star_ratings, update_manifest_stars

__all__ = ["OsuV2Client", "fetch_star_ratings", "update_manifest_stars"]
