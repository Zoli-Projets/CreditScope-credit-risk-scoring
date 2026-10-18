# CreditScope — déploiement dans credit-risk-scoring

1. Dans le dépôt GitHub existant, créer un dossier `CreditScope/`.
2. Y copier `app.py` et `requirements.txt` fournis ici.
3. Placer `model_data.joblib` **dans ce même dossier**, uniquement si sa publication est autorisée. Le modèle est volontairement absent de cette archive.
4. Le `.gitignore` du dépôt ignore `*.joblib`. Pour autoriser uniquement ce modèle, ajouter à la fin de `.gitignore` : `!CreditScope/model_data.joblib` (sous réserve de l'autorisation de publication).
5. Sur Streamlit Community Cloud, sélectionner le dépôt `credit-risk-scoring`, la branche `main` et `CreditScope/app.py`.
6. Vérifier les logs et fixer les versions des dépendances selon la version utilisée pour entraîner le modèle.

Attention : ne pas publier les fichiers CSV bancaires ni les données confidentielles sans autorisation. Le fichier de dépendances fourni est une base de départ et doit être validé avec l'environnement du modèle sauvegardé.
