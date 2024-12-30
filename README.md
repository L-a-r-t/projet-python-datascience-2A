# Prédire la durée des incidents sur le réseau francilien à partir des alertes de trafic ?

#### LARTIGAU Théo et SIAHAAN--GENSOLLEN Rémy

Tout le monde ou presque s'est déjà retrouvé confronté à des incidents  dans les transports en commun : un jour, alors
qu'on se rend à un rendez-vous important, notre train est soudainement bloqué, ou il est en retard, ou ne vient juste 
pas. On n'a parfois qu'une annonce simple, une alerte de trafic qui ressemble peu ou proue à ça :

![catpure d'un post sur X (anciennement Twitter) de @RER_B: ⚠️ Le trafic est perturbé en direction de 
Saint-Rémy-lès-Chevreuse / Robinson en raison d'un train en panne au Bourget. #RERB](images/x.png)
_Source : X (anciennement Twitter)._

Comme le montre cet exemple, ces messages ne sont parfois pas d'une grande aide. Souvent, avoir une idée de la durée
de l'incident peut s'avérer utile, voire très utile, ne serait-ce que pour prévenir du retard, ou mieux planifier
un trajet alternatif. Dans ce projet, on se propose d'entraîner un modèle sur les données fournies par l'API d'IDF
Mobilités sur les perturbations du réseau francilien, contenant notamment les messages et les durées, afin d'essayer
de répondre à ce problème. Nous analyserons également ces données de sorte à mieux comprendre la nature des
perturbations sur le réseaux de transports en commun d'Île de France.

Ce fichier détaille notre démarche et les résultats obtenus, ainsi que des considérations pratiques vis-à-vis du projet
(fonctionnement, reproductibilité…).

#### Table des matières
- [Structure du projet](#struct)  
  - [Code](#code)
  - [Dépendances](#dep)
  - [Variables d'environnement](#env)
  - [Accélération GPU](#gpu)
- [Démarche](#work)
  - [Scraping de l'API IDF Mobilités](#scraping)
  - [Pré-traitement des données](#preprocessing)


<a name="struct"></a>
## Structure du projet

<a name="code"></a>
#### Code
Le code est situé dans le dossier `src`, principalement dans un notebook `main.ipynb`. Des parties du code ne sont
cependant pas dedans :

- le scraping périodique de l'API IDF Mobilités est effectué par une GitHub Action paramétrée dans
`.github/workflows.scheduler.yml`, qui éxécute le fichier `scraping-job.py`. Les réponses brutes de l'API IDF Mobilités
sont directement stockées au format JSON dans un bucket du SSP Cloud. Voir [Scraping de l'API IDF Mobilités](#scraping)
pour plus de détails.

- Le pré-traitement des données, qui peut être un peu long, est effectué à part dans le notebook `preprocessing.ipynb`. 
Les données sont enregistrées dans le dossier `src/data` au format _feather_. Voir 
[Pré-traitement des données](#preprocessing) pour plus de détails.


<a name="dep"></a>
#### Dépendances
Le fichier `requirements.txt` liste selon les conventions les dépendances du projet. Il est possible d'installer 
ces dépendances via

```bash
pip install -r requirements.txt
```

Une cellule avec cette commande est sinon disponible dans chaque notebook.

<a name="env"></a>
#### Variables d'environnement
Le projet requiert plusieurs variables d'environnement, notamment une clé sur 
l'[API IDF Mobilités](https://prim.iledefrance-mobilites.fr/fr) et un compte de service pour l'accès MinIO du SSP Cloud
(voir la [documentation du SSP Cloud](https://docs.sspcloud.fr/content/storage.html#cr%C3%A9er-un-compte-de-service) 
à ce sujet). Ces variables peuvent être stockées dans un fichier `.env` sur la racine, suivant les conventions. 
Le fichier `.env.example` détaille la structure et le nom des variables :

```py
## Clé de l'API IDF Mobilités
# voir https://prim.iledefrance-mobilites.fr/fr
IDFM_API_KEY=""

## Configuration du système de stockage
MINIO_S3_ENDPOINT="https://minio.lab.sspcloud.fr"
MINIO_ROOT="username/projet-python-2A"
# On utilise un compte de service
# voir https://docs.sspcloud.fr/content/storage.html#cr%C3%A9er-un-compte-de-service
MINIO_KEY=""
MINIO_SECRET_KEY=""
```

<a name="gpu"></a>
#### Accélération GPU

Les opérations effectuées au cours de ce projet peuvent être assez longue. Pour gagner du temps (ou simplement parce
les CPU n'étaient parfois pas adaptés) nous avons utilisé des GPU, notamment ceux mis en place par le SSP Cloud.
Ceux du SSP Cloud fonctionnent avec CUDA 12, et on peut alors installer cuDF pour cette version avec le code suivant :

```bash
!pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.12.* dask-cudf-cu12==24.12.* cuml-cu12==24.12.* cugraph-cu12==24.12.*
```

puis le charger dans les notebooks avec ```%load_ext cudf.pandas```. Pour les versions adaptées à CUDA 11, il faut se 
référer à la [documentation RAPIDS](https://docs.rapids.ai/install/). Une cellule avec cette commande est sinon 
disponible dans chaque notebook. 

<a name="struct"></a>
## Démarche

<a name="scraping"></a>
#### Scraping de l'API IDF Mobilités

La première étape de ce projet a consisté à récupérer les données concernant les incidents sur le réseau francilien.
IDF Mobilités possède plusieurs comptes sur des réseaux sociaux — principalement X, anciennement Twitter — mais les API 
d'accès de ces réseaux sont plutôt onéreuses, et par ailleurs ces comptes à eux seuls ne représentent pas exhaustivement 
 toute l'information disponible.

Une autre approche est plutôt de se tourner vers la plateforme PRIM (Plateforme Régionale d'Information pour la 
mobilité) gérée par IDF Mobilités. Celle-ci dispose de plusieurs APIs, dont une qui a été mise en place pour les jeux 
olympiques de Paris 2024 : L’[API Messages Info 
Trafic](https://prim.iledefrance-mobilites.fr/fr/apis/idfm-disruptions_bulk), qui renvoie l'intégralité des informations
de perturbation en cours et à venir, ainsi que la liste des lignes et arrêts concernés.

Cependant celle-ci ne propose qu'un endpoint renvoyant les perturbations en cours au moment de la requête. Nous avons 
donc créé un script (`scraping-job.py`) qui requête cette API et stocke de manière horodatée la réponse dans le stockage
du SSP Cloud, éxécuté périodiquement par une GitHub action.

L'action est en théorie censée s'exécuter toutes les deux minutes ainsi que paramétrée par la tache CRON 
(`.github/workflows/scheduler.yml`) et selon ce que permettaient les quotas d'utilisations de l'API IDF Mobilités. En 
pratique, nous nous sommes rendus compte après analyse qu'une partie des perturbations dure moins de deux minutes, ce qui 
les laisse passer sous le radar. Par ailleurs, et c'est le principal défaut, les workers proposés par GitHub (même dans
la version pro) ne garantissent pas l'exécution à temps des tâches CRON (voir le 
[forum communautaire GitHub](https://github.com/orgs/community/discussions/27130)). Une solution aurait été de faire
tourner sois-même un worker (ou la tâche CRON directement), mais nous ne disposions pas de serveur pour faire cela.

**Nous avons fait tourner le script de scraping du 6 décembre 2024 à 13h50, jusqu'au 29 décembre à 20h56.**

<a name="preprocessing"></a>
#### Pré-traitement des données `preprocessing.ipynb`

Les données issues du scraping de l'API IDF Mobilités ne constituent pas un jeu de données en tant que tel. L'on a donc 
cherché à transformer ces données brutes, stockées sur le SSP Cloud, en un jeu de données exploitable. Les fichiers
JSON issus du scraping contiennent des informations sur les perturbations du réseau francilien, comme leurs causes, 
gravités et périodes d'application (il faut noter la quantité d'information donnée par l'API — les champs disponibles 
dans la réponse — varie d'une requête à une autre). Ces perturbations impactent certains objets, entités spécifiques 
du réseau : lignes de transport, ou arrêts sur ces lignes, permettant de localiser précisément les perturbations.

Pour cela, nous avons filtré les doublons, et structurés les données en trois tables principales. Une pour les 
perturbations, une pour les objets impactés (lignes, stations), et une faisant la jointure entre les deux : un lien 
objet-perturbation, correspondant à une perturbation unique sur une période unique et pour un objet impacté unique. 
Ainsi, une perturbation sur deux périodes et impactant trois objets (par exemple des travaux sur deux week-ends et 
affectant trois lignes) correspondra à six liens objet-perturbation. Cet aplatissement est plus pratique pour étudier 
_in fine_ les perturbations. 

Les données ainsi nettoyées ont été converties en DataFrames pandas. Créer des DataFrames plus rudimentaires puis les
raffiner s'avérait être un processus très complexe du fait d'objets imbriqués et de listes de longueurs variables dans 
nos données brutes. Au final, nous avons :

| Catégorie                               | Total     | Total (sans doublons) |
|-----------------------------------------|-----------|-----------------------|
| Résultats                               | 2 357     | 2 357                 |
| Perturbations traitées                  | 1 724 401 | 30 177                |
| Objets impactés traités                 | 5 476 570 | 7 570                 |
| Liens objet-perturbation traités        | -         | 102 807               |

Le Pré-traitement pouvant être long, en particulier sur CPU, nous avons enregistré ces jeux de données dans le dossier
`src/data`, au format _feather_. Ce format était idéal pour nous, car bien moins volumineux que le CSV (notamment du fait
de la précense de messsages en texte plein, qui alourdissaient considérablement les fichiers si non stockés en binaire),
et sur lesquels ils nous étaient plus rapide d'itérer qu'avec _parquet_.

<a name="data-analysis"></a>
#### Analyse des données `analysis.ipynb`

Nous cherchons à atteindre plusieurs objectifs via l'analyse des données, ils sont l'identification :
1. Des lignes et des tronçons les plus problématiques sur le réseau IDF Mobilités
2. Des types de perturbations les plus fréquentes
3. De potentiels problèmes dans la restitution des données par l'API IDF Mobilités

Parmi les objets impactés, nous avons pu récupérer l'ensemble des lignes de de métro, de RER et la quasi-totalité des 
lignes de Tramway francilien sur la période de scraping. Nous avons aussi récupéré la moitié des lignes de Transilien et 
les deux-tiers lignes de bus du réseau.

L'analyse des données sur les incidents nous a permis de constater qu'hormis les informations sur la temporalité 
d'une perturbation, la quasi-totalité des informations la concernant sont contenues dans le message d'information 
associé à celle-ci. Le traitement du langage naturel est donc apparu comme étant une solution pour augmenter le jeu 
de données, surtout si nous souhaitons identifier les _types_ de perturbations les plus fréquentes. L'API IDF Mobilités 
ne précise que si une perturbation est due à des travaux, et ce de façon non systématique.

Néanmoins, nous n'avons pas réussi à identifier les lignes les plus problématiques du réseau de transports. Nous avons
en effet fini par constater que l'API IDF Mobilités renvoyait parfois des perturbations à l'identique des dizaines de
fois et avec des identifiants uniques différents, ce qui rend toute déduplication impossible sans faire appel à des
grands modèles de langage (LLMs) très énergivores. 

Cela veut néanmoins dire que nous avons atteint notre troisième objectif. Nous avons donc contacté l'équipe en charge
de l'API IDF Mobilités pour leur signaler le problème.

Une solution naïve pour augmenter notre jeu de données est de déterminer le type de la perturbation en fonction de la 
présence de mots-clés soigneusement choisis à l'intérieur du message d'info-traffic lié, mais celle-ci n'a pas été
efficace.
Enfin, à l'aide d'une version de [CamemBERT](dangvantuan/sentence-camembert-base) (sélectionné car entraîné sur un corpus 
de textes francophones) finetunée pour produire des embeddings de phrases, et de la librairie 
[KeyBERT](https://maartengr.github.io/KeyBERT/), nous avons essayé d'augmenter les données sans grand succès.
