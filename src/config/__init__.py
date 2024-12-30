"""
Module de Configuration pour l'Extraction et la Sauvegarde des Données API

Ce module charge les variables d'environnement nécessaires à l'application,
configure le système de fichiers S3 pour l'interaction avec SSPCloud,
et définit les constantes de configuration utilisées par le script principal.

Les variables d'environnement doivent être définies dans un fichier `.env` ou
via les secrets de GitHub Actions pour garantir la sécurité des informations sensibles.
"""

import s3fs
from dotenv import load_dotenv
import os

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Récupération des clés API et des configurations S3 depuis les variables d'environnement
IDFM_API_KEY = os.environ.get("IDFM_API_KEY")
MINIO_S3_ENDPOINT = os.environ.get("MINIO_S3_ENDPOINT")
MINIO_ROOT = os.environ.get("MINIO_ROOT")
MINIO_KEY = os.environ.get("MINIO_KEY")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")

# Configuration du système de fichiers S3
fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": MINIO_S3_ENDPOINT},
    key=MINIO_KEY,
    secret=MINIO_SECRET_KEY
)