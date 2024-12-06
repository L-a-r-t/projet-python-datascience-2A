import s3fs
from dotenv import load_dotenv
import os

load_dotenv()
SECRET = os.environ.get("MINIO_SECRET_KEY")

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}, key="python-2A", secret=SECRET)

# Dossier du projet dans le bucket S3 sur sspcloud
ROOT = "tlartigau/projet-python-2A"

# https://prim.iledefrance-mobilites.fr/fr/apis/idfm-disruptions_bulk
ENDPOINT_URL = "https://prim.iledefrance-mobilites.fr/marketplace/disruptions_bulk/disruptions/v2"