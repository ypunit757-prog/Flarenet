"""Small HTTP layer for NASA / NOAA endpoints: retries, atomic downloads, directory listings.

Honours the standard proxy environment variables (HTTPS_PROXY ...) via ``requests``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests

from flarenet.core.utils import get_logger
from flarenet.sources.base import SourceUnavailable

log = get_logger(__name__)
USER_AGENT = "flarenet/1.0 (solar flare research; https://github.com/)"


def http_get(
    url: str,
    params: dict | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 2.0,
    stream: bool = False,
    session: requests.Session | None = None,
) -> requests.Response:
    """GET with retries on connection errors and 5xx/429. 4xx (except 429) fail immediately."""
    getter = session.get if session is not None else requests.get
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = getter(
                url,
                params=params,
                timeout=timeout,
                stream=stream,
                headers={"User-Agent": USER_AGENT},
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            if resp.status_code >= 400:
                raise SourceUnavailable(f"HTTP {resp.status_code} for {resp.url}")
            return resp
        except SourceUnavailable:
            raise
        except requests.RequestException as e:
            last = e
            if attempt < retries:
                wait = backoff**attempt
                log.warning(
                    "GET %s failed (%s); retry %d/%d in %.0fs", url, e, attempt, retries, wait
                )
                time.sleep(wait)
    raise SourceUnavailable(
        f"Could not reach {url} after {retries} attempts ({last}). If this machine has restricted "
        "network access, download the files manually and use kind 'nasa_goes_file' or the "
        "'local_dir' option instead (see docs/NASA_DATA.md)."
    )


def get_json(url: str, params: dict | None = None, **kw):
    resp = http_get(url, params=params, **kw)
    try:
        return resp.json()
    except ValueError as e:
        raise SourceUnavailable(f"{url} did not return JSON: {e}") from e


def download_file(url: str, dest: Path, **kw) -> Path:
    """Download to ``dest`` atomically (``.part`` file renamed at the end)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    attempts, backoff = int(kw.get("retries", 3)), float(kw.get("backoff", 2.0))
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = http_get(url, stream=True, **kw)
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            part.replace(dest)
            return dest
        except SourceUnavailable:
            part.unlink(missing_ok=True)
            raise
        except (requests.RequestException, OSError) as e:  # connection dropped mid-file
            part.unlink(missing_ok=True)
            last = e
            if attempt < attempts:
                wait = backoff**attempt
                log.warning("download of %s broke (%s); retry %d/%d in %.0fs", url, e, attempt,
                            attempts, wait)  # fmt: skip
                time.sleep(wait)
    raise SourceUnavailable(f"Download of {url} failed after {attempts} attempts ({last})")


def list_directory(url: str, pattern: str, **kw) -> list[str]:
    """Return the link targets of an Apache-style directory index that match regex ``pattern``."""
    html = http_get(url, **kw).text
    hrefs = re.findall(r'href="([^"?#]+)"', html)
    rx = re.compile(pattern)
    return sorted({h.split("/")[-1] for h in hrefs if rx.search(h)})
