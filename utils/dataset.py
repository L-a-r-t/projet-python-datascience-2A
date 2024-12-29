import os
import s3fs
from dotenv import load_dotenv


load_dotenv()

KEY = os.environ.get("MINIO_KEY")
SECRET = os.environ.get("MINIO_SECRET")

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}, key=KEY, secret=SECRET)

def download_dataset():
    paths = fs.ls("tlartigau/projet-python-2A/intermediate_data")
    for path in paths:
        local_path = "data/" + path.split("/").pop()
        if os.path.isfile(local_path) or ".keep" in local_path:
            continue
        with fs.open(path, "r", encoding="utf-8") as remote_f:
            with open(local_path, "w+", encoding="utf-8") as local_f:
                local_f.write(remote_f.read())