"""Download a small, fixed slice of GPR1200 and define the two retrieval tasks.

GPR1200 (Schall et al., 2021, https://arxiv.org/abs/2111.13122) has 1,200 groups of
10 images. Each block of 200 groups comes from a different source, so one small
benchmark mixes *instance* retrieval (the same landmark, object, or product) with
*category* retrieval (the same species or ImageNet class). We keep every 8th group
from five blocks (1,250 images) and skip the sixth block, which holds celebrity faces.

Only the selected JPEGs are read from the 1.25 GB archive, using HTTP range requests.
GPR1200 is provided for non-commercial research and educational use only; the images
remain the property of their original owners.
"""

import csv
import http.client
import io
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from .augment import AUGMENTATIONS
from .embed import ImageItem

ARCHIVE_URLS = (
    # Hugging Face mirror pinned to one commit, then the authors' own server.
    "https://huggingface.co/datasets/Visual-Computing-Group/GPR1200/resolve/"
    "a7c3ec7870a22b66c2eaa5ddc4243037886ef541/GPR1200.zip",
    "https://files.visual-computing.com/GPR1200/GPR1200.zip",
)
ARCHIVE_SIZE = 1_247_901_539
MAX_IMAGE_BYTES = 50_000_000  # the largest selected JPEG is about 7 MB
DOWNLOAD_ERRORS = (OSError, RuntimeError, zipfile.BadZipFile, http.client.HTTPException, zlib.error)


@dataclass(frozen=True)
class Domain:
    name: str
    kind: str  # "instance" or "category"
    source: str
    first_group: int


DOMAINS = (
    Domain("Landmarks", "instance", "Google Landmarks v2", 0),
    Domain("Nature", "category", "iNaturalist species", 200),
    Domain("Sketches", "category", "ImageNet-Sketch", 400),
    Domain("Objects", "instance", "INSTRE", 600),
    Domain("Products", "instance", "Stanford Online Products", 800),
)  # Groups 1000–1199 (IMDB celebrity faces) are intentionally not used.
GROUP_STEP = 8  # keep groups 0, 8, 16, ... → 25 groups per domain
IMAGES_PER_GROUP = 10


@dataclass(frozen=True)
class Photo:
    image_id: str  # file name without extension
    path: Path
    group: int     # images in the same group show the same thing
    domain: str


def selected_groups() -> dict[int, str]:
    """Group number → domain name for the groups used in this homework."""
    return {
        group: domain.name
        for domain in DOMAINS
        for group in range(domain.first_group, domain.first_group + 200, GROUP_STEP)
    }


class _HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched in byte ranges.

    zipfile asks for small pieces; the io.BufferedReader in `_open_archive` groups
    them into 256 KB requests, so reading one JPEG costs one or two HTTP requests.
    """

    def __init__(self, url: str):
        super().__init__()
        self.source, self.position = url, 0
        self._resolve()

    def _resolve(self) -> None:
        """Follow redirects (CDN links expire after a while) and check the archive size."""
        request = urllib.request.Request(self.source, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.url
            length = response.headers.get("Content-Length")
        if length is None or int(length) != ARCHIVE_SIZE:
            raise RuntimeError(f"Unexpected archive size ({length} bytes) at {self.source}.")
        self.size = ARCHIVE_SIZE

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        length = min(len(buffer), self.size - self.position)
        if length <= 0:
            return 0
        data = self._get_range(self.position, self.position + length - 1)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)

    def _get_range(self, start: int, end: int, attempts: int = 3) -> bytes:
        """Fetch bytes start..end, retrying brief network problems."""
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise RuntimeError("The server ignored the byte-range request.")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise http.client.IncompleteRead(data)
                return data
            except urllib.error.HTTPError as error:
                if attempt == attempts:
                    raise
                if error.code == 403:
                    self._resolve()  # the signed CDN link expired; ask for a new one
            except (OSError, http.client.HTTPException):
                if attempt == attempts:
                    raise
            time.sleep(2**attempt)
        raise AssertionError("unreachable")


def _open_archive(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_HTTPRangeFile(url), buffer_size=1 << 18))


def _download_members(url: str, members: list[str], image_dir: Path, workers: int) -> None:
    local = threading.local()

    def fetch(name: str) -> None:
        destination = image_dir / PurePosixPath(name).name
        if destination.is_file():
            return  # finished in an earlier run
        if not hasattr(local, "archive"):
            local.archive = _open_archive(url)  # one connection per thread
        info = local.archive.getinfo(name)
        if info.file_size > MAX_IMAGE_BYTES:
            raise RuntimeError(f"{name} is unexpectedly large ({info.file_size:,} bytes).")
        data = local.archive.read(info)  # zipfile checks each file's CRC-32
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, name) for name in members]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
                future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)  # stop queued work, try the next mirror
            raise


def _is_complete(root: Path) -> bool:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return False
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(selected_groups()) * IMAGES_PER_GROUP
    return len(rows) == expected and all((root / "images" / row["file"]).is_file() for row in rows)


def _list_members(url: str, groups: dict[int, str]) -> list[str]:
    with _open_archive(url) as archive:
        members = sorted(
            info.filename for info in archive.infolist()
            if not info.is_dir() and _group_of(info.filename) in groups
        )
    expected = len(groups) * IMAGES_PER_GROUP
    if len(members) != expected:
        raise RuntimeError(f"Expected {expected} images, found {len(members)}.")
    if len({PurePosixPath(name).name for name in members}) != len(members):
        raise RuntimeError("Two archive members share a file name.")
    return members


def _write_manifest(root: Path, members: list[str], groups: dict[int, str]) -> None:
    temporary = root / "manifest.csv.part"
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "group", "domain"])
        for name in members:
            group = _group_of(name)
            writer.writerow([PurePosixPath(name).name, group, groups[group]])
    temporary.replace(root / "manifest.csv")  # written last, so it marks a finished download


def prepare_gpr1200(data_dir: Path, *, workers: int = 8) -> Path:
    """Download the 1,250 selected images once; later calls reuse them."""
    root = data_dir / "gpr1200"
    if _is_complete(root):
        return root
    groups = selected_groups()
    (root / "images").mkdir(parents=True, exist_ok=True)
    errors = []
    for url in ARCHIVE_URLS:
        try:
            print(f"Reading the GPR1200 file list from {url.split('/')[2]} ...")
            members = _list_members(url, groups)
            print(f"Downloading {len(members):,} images (about 150 MB) ...")
            _download_members(url, members, root / "images", workers)
            _write_manifest(root, members, groups)
            return root
        except DOWNLOAD_ERRORS as error:
            errors.append(f"{url}: {error!r}")
    raise RuntimeError("Could not download GPR1200. Check your internet connection.\n" + "\n".join(errors))


def _group_of(member_name: str) -> int | None:
    prefix = PurePosixPath(member_name).name.split("_", 1)[0]
    return int(prefix) if prefix.isdecimal() else None


def read_photos(root: Path) -> list[Photo]:
    """Read the manifest and check that every group has all ten images."""
    with (root / "manifest.csv").open(newline="") as stream:
        photos = [
            Photo(
                image_id=PurePosixPath(row["file"]).stem,
                path=root / "images" / row["file"],
                group=int(row["group"]),
                domain=row["domain"],
            )
            for row in csv.DictReader(stream)
        ]
    counts: dict[int, int] = {}
    for photo in photos:
        counts[photo.group] = counts.get(photo.group, 0) + 1
    if set(counts) != set(selected_groups()) or set(counts.values()) != {IMAGES_PER_GROUP}:
        raise ValueError("The image folder does not match the expected 125 groups of 10.")
    if len({photo.image_id for photo in photos}) != len(photos):
        raise ValueError("Image ids must be unique.")
    return sorted(photos, key=lambda photo: (photo.group, photo.image_id))


@dataclass(frozen=True)
class CopyQuery:
    item: ImageItem
    source_id: str
    edit: str


def copy_queries(photos: list[Photo]) -> list[CopyQuery]:
    """Task 2: one edited copy per (group, edit), made from each group's first image."""
    first = {}
    for photo in photos:
        first.setdefault(photo.group, photo)
    return [
        CopyQuery(
            item=ImageItem(f"{photo.image_id}__{edit}", photo.path, edit),
            source_id=photo.image_id,
            edit=edit,
        )
        for photo in first.values()
        for edit in AUGMENTATIONS
    ]


def gallery_items(photos: list[Photo]) -> list[ImageItem]:
    return [ImageItem(photo.image_id, photo.path) for photo in photos]
