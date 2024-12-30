"""
Script d'Extraction et de Sauvegarde des Données API vers S3

Ce script est exécuté périodiquement via une GitHub Action
(voir .github/workflows/scheduler.yml).

Ce script effectue une requête GET à l'API IDF Mobilités pour récupérer les perturbations.
Les données reçues sont vérifiées pour s'assurer qu'elles sont valides (statuts 200, 403, 404).
En cas de succès, les données JSON sont sauvegardées dans un bucket S3 sur SSPCloud,
avec un nom de fichier incluant la date et l'heure actuelles.
"""

import requests
import json
from datetime import datetime, timezone

import config

# URL de l'API IDF Mobilités pour les perturbations
# voir la documentation à https://prim.iledefrance-mobilites.fr/fr/apis/idfm-disruptions_bulk
ENDPOINT_URL = "https://prim.iledefrance-mobilites.fr/marketplace/disruptions_bulk/disruptions/v2"

# Configuration des en-têtes de la requête
headers = {
    "Accept": "application/json",
    "apiKey": config.IDFM_API_KEY
}

try:
    # Exécution de la requête GET
    response = requests.get(ENDPOINT_URL, headers=headers)

    # Vérification du statut de la réponse
    if response.status_code == 200:
        # Obtention de la date et heure actuelles en UTC
        current_datetime = datetime.now(timezone.utc)
        timestamp = current_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        path = f"{config.MINIO_ROOT}/{timestamp}.json"

        # Sauvegarde des données JSON dans un fichier S3
        with config.fs.open(path, "w") as f:
            json.dump(response.json(), f)

        print(f"Données sauvegardées dans {path}")

    elif response.status_code == 403:
        # Gestion de l'erreur d'accès refusé
        print("Erreur 403 : Accès refusé. Vérifiez la clé API et les permissions.")
        raise SystemExit("Fin du script en raison d'une erreur 403.")

    elif response.status_code == 404:
        # Gestion de l'erreur de ressource non trouvée
        print("Erreur 404 : Ressource non trouvée. Vérifiez l'URL de l'API.")
        raise SystemExit("Fin du script en raison d'une erreur 404.")

    else:
        # Gestion des autres erreurs HTTP
        # Ce cas n'est pas censé survenir si la requête est bien traitée par l'API IDF Mobilités
        print(f"Erreur {response.status_code} : {response.reason}")
        raise SystemExit(f"Fin du script en raison d'une erreur {response.status_code}.")

except requests.exceptions.RequestException as e:
    # Gestion des exceptions liées à la requête HTTP
    print(f"Erreur lors de la requête API : {e}")
    raise SystemExit("Fin du script en raison d'une exception lors de la requête API.")