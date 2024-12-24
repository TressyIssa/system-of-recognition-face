# Face Recongnition
![Intro](https://github.com/user-attachments/assets/030722b6-f96d-4dd3-b463-6aaea1b9e368)

## Contexte
Le projet vise à comparer les visages détectés dans quatres images afin d'identifier les visages similaires.
Cette tâche est essentielle dans des applications telles que la reconnaissance faciale, la sécurité, et l'analyse de photographie.
Pour ce faire, nous avons utilisé des techniques de détection de visages et de comparaison de caractéristiques faciales..

## Motivation
La motivation derrière ce travail est de développer une méthode efficace pour comparer des visages entre deux images, ce qui peut être appliqué dans divers domaines comme :
- **Sécurité** : Pour identifier des individus dans des systèmes de surveillance.
- **Réseaux sociaux** : Pour taguer des amis automatiquement dans des photos.
- **Photographie** : Pour organiser et rechercher des photos contenant des visages similaires.
- **Recherche académique** : Pour étudier les caractéristiques faciales et les relations entre différents visages.

## Installation
Pour exécuter ce projet, vous devez installer les bibliothèques nécessaires. Suivez les étapes ci-dessous pour configurer votre environnement.

### Prérequis
- Python 3.x
- pip (gestionnaire de paquets pour Python)

### Étapes d'Installation
1. Clonez ce dépôt GitHub sur votre machine locale :
    ```sh
    git clone https://github.com/votre-nom-utilisateur/votre-repo.git
    cd votre-repo
    ```

2. Installez les bibliothèques nécessaires :
    ```sh
    pip install tensorflow keras keras-applications keras-preprocessing mtcnn scikit-learn opencv-python matplotlib
    ```

## Utilisation
Après avoir installé les dépendances, suivez les étapes ci-dessous pour comparer les visages entre deux images.

### 1. Détection et Extraction des Visages
Nous utilisons le modèle MTCNN pour détecter et extraire les visages des images.

```python
import cv2
from mtcnn import MTCNN

def detect_faces(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)
    extracted_faces = []

    for face in faces:
        x, y, width, height = face['box']
        face_img = image_rgb[y:y+height, x:x+width]
        extracted_faces.append(face_img)

    return extracted_faces
