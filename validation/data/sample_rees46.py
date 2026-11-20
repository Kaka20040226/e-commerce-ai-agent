"""Stream a deterministic user sample from the REES46 marketplace release."""

import argparse
from contextlib import contextmanager
from datetime import date
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import struct
import zlib


SOURCE = "https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz"
KAGGLE_DATASET = "mkechinov/ecommerce-behavior-data-from-multi-category-store"
KAGGLE_FILE = "2019-Oct.csv"
COLUMNS = (b"event_time,event_type,product_id,category_id,category_code,brand,"
           b"price,user_id,user_session")
ROOT = Path(__file__).resolve().parent


@contextmanager
def open_gzip(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location",
               "--connect-timeout", "15", "--max-time", "420", url]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as stream:
        try:
            with gzip.GzipFile(fileobj=stream.stdout) as raw:
                yield iter(raw)
        finally:
            if stream.poll() is None:
                stream.terminate()


def zip_csv_lines(raw):
    header = raw.read(30)
    if len(header) != 30 or not header.startswith(b"PK\x03\x04"):
        raise ValueError("Kaggle did not serve the expected ZIP archive")
    flags, method = struct.unpack_from("<HH", header, 6)
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    name = raw.read(name_len).decode("utf-8")
    raw.read(extra_len)
    if Path(name).name != KAGGLE_FILE or method != 8 or flags & 1:
        raise ValueError("Unexpected ZIP member or compression method")
    inflater = zlib.decompressobj(-15)
    carry = b""
    while not inflater.eof:
        chunk = raw.read(262144)
        if not chunk:
            raise EOFError("Kaggle ZIP stream ended unexpectedly")
        lines = (carry + inflater.decompress(chunk)).split(b"\n")
        carry = lines.pop()
        for line in lines:
            yield line + b"\n"
    if carry:
        yield carry


@contextmanager
def open_kaggle():
    from kaggle.api.kaggle_api_extended import ApiDownloadDatasetRequest, KaggleApi

    api = KaggleApi()
    api.authenticate()
    request = ApiDownloadDatasetRequest()
    request.owner_slug, request.dataset_slug = KAGGLE_DATASET.split("/", 1)
    request.file_name = KAGGLE_FILE
    with api.build_kaggle_client() as client:
        with client.datasets.dataset_api_client.download_dataset(request) as response:
            response.raise_for_status()
            yield zip_csv_lines(response.raw)


def sample(url, end, output, kaggle=False):
    cutoff = end.encode("ascii")
    pending = output.with_suffix(output.suffix + ".part")
    manifest = output.with_name(output.name.removesuffix(".csv.gz") + ".json")
    digest = sha256()
    seen = kept = 0
    previous = None
    first = last = None
    source = open_kaggle() if kaggle else open_gzip(url)
    try:
        with source as raw:
            with pending.open("wb") as target:
                with gzip.GzipFile(filename="", fileobj=target, mode="wb", compresslevel=3, mtime=0) as saved:
                    header = next(raw, b"")
                    if header.strip() != COLUMNS:
                        raise ValueError("Unexpected CSV header")
                    saved.write(header)
                    digest.update(header)
                    for line in raw:
                        stamp = line[:19]
                        if previous is not None and stamp < previous:
                            raise ValueError("Dataset events are not ordered by timestamp")
                        if stamp[:10] >= cutoff:
                            break
                        previous = stamp
                        seen += 1
                        user_id = line.rsplit(b",", 2)[1]
                        if not user_id.isdigit():
                            raise ValueError(f"Invalid user identifier at row {seen}")
                        if zlib.crc32(user_id) % 10 < 2:
                            saved.write(line)
                            digest.update(line)
                            kept += 1
                            first = first or stamp
                            last = stamp
                        if seen % 1_000_000 == 0:
                            print(f"Scanned {seen:,}, kept {kept:,}, through {stamp.decode()} UTC", flush=True)
                    else:
                        raise RuntimeError("Unable to confirm temporal boundary")
        if not kept:
            raise RuntimeError("No events sampled")
        pending.replace(output)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    meta = {
        "source_url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}" if kaggle else url,
        "source_file": KAGGLE_FILE if kaggle else Path(url).name,
        "stop_before_utc_date": end,
        "first_event_utc": first.decode(),
        "last_event_utc": last.decode(),
        "user_sample": "CRC32 of the decimal user ID modulo 10 is less than 2",
        "source_events_scanned": seen,
        "sample_events": kept,
        "uncompressed_sample_sha256": digest.hexdigest(),
    }
    manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SOURCE)
    parser.add_argument("--kaggle", action="store_true", help="Use the authenticated Kaggle SDK download.")
    parser.add_argument("--end", default="2019-10-09")
    parser.add_argument("--output", type=Path, default=ROOT / "marketplace-oct-sample.csv.gz")
    args = parser.parse_args()
    sample(args.url, date.fromisoformat(args.end).isoformat(), args.output, kaggle=args.kaggle)
