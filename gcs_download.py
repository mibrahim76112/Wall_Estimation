from pathlib import Path
from google.cloud import storage

def download_gcs_object_if_missing(bucket: str, object_name: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    blob.download_to_filename(str(dest_path))
