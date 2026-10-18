import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='CreditScope | Scoring bancaire',page_icon='🏦',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:linear-gradient(125deg,#f4f8fc,#eef4fa);color:#14233b}
section[data-testid="stSidebar"]{background:#10243e}
section[data-testid="stSidebar"] *{color:#f5f7fb!important}
.block-container{padding-top:2rem;max-width:1450px}
div[data-testid="stMetric"]{background:white;padding:20px;border-radius:16px;border:1px solid #e5ebf3;box-shadow:0 3px 16px #10243e0b}
h1,h2,h3{color:#132d4c}
.hero{padding:30px 34px;background:linear-gradient(110deg,#10243e,#1d5c8c);border-radius:20px;color:white;margin:10px 0 28px}
.hero h2{color:white;margin:0}.hero p{color:#d8e9f8;margin:10px 0 0}
</style>''',unsafe_allow_html=True)
BASE=Path(__file__).parent
@st.cache_resource
def load_model(path):
    return joblib.load(path)
@st.cache_data
def read_csv(blob):
    return pd.read_csv(io.BytesIO(blob))

st.sidebar.title('🏦 CreditScope')
st.sidebar.caption('Plateforme d’analyse du risque de crédit')
page=st.sidebar.radio('Navigation',['Vue d’ensemble','Exploration des données','Scoring individuel','Scoring par lot','Performance & explicabilité','Méthodologie'])
st.sidebar.divider()
st.sidebar.caption("📊 CreditScope • Outil d'aide à la décision")
model_upload=st.sidebar.file_uploader('Modèle entraîné (.joblib)',type=['joblib'])
model_path=BASE/'model_data.joblib'
artifact=None
try:
    if model_upload is not None:
        artifact=joblib.load(io.BytesIO(model_upload.getvalue()))
    elif model_path.exists():
        artifact=load_model(str(model_path))
except Exception as e:
    st.sidebar.error(f'Chargement du modèle impossible : {e}')
if artifact is not None and not isinstance(artifact,dict):
    artifact={'model':artifact,'features':list(getattr(artifact,'feature_names_in_',[]))}
model=artifact.get('model') if artifact else None
features=list(artifact.get('features',[])) if artifact else []
if model is not None and not features:features=list(getattr(model,'feature_names_in_',[]))

def prepare_for_model(df):
    """Applique les paramètres MinMax par variable, sans exiger les 17 colonnes.

    Un MinMaxScaler traite chaque colonne indépendamment. Cette opération est
    algébriquement identique à sa transformation sur la colonne correspondante,
    sous réserve que le notebook ait réellement standardisé ces colonnes.
    """
    if model is None or not features:
        raise ValueError('Chargez un modèle et la liste de ses variables.')
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError('Colonnes manquantes : ' + ', '.join(missing))
    x = df.loc[:, features].copy()
    for c in features:
        x[c] = pd.to_numeric(x[c], errors='raise')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Valeurs manquantes ou non finies dans les données.')
    scaler = artifact.get('scaler') if artifact else None
    if scaler is not None:
        cols = list(artifact.get('cols_to_scale', []))
        trained_names = list(getattr(scaler, 'feature_names_in_', []))
        if not trained_names:
            raise ValueError('Le scaler ne contient pas feature_names_in_ : impossible de faire correspondre les colonnes en sécurité.')
        if len(trained_names) != len(scaler.scale_) or len(trained_names) != len(scaler.min_):
            raise ValueError('Paramètres du scaler incohérents.')
        idx = {name: i for i, name in enumerate(trained_names)}
        for col in cols:
            if col in x.columns:
                if col not in idx:
                    raise ValueError(f'Variable absente du scaler entraîné : {col}')
                j = idx[col]
                x[col] = x[col] * float(scaler.scale_[j]) + float(scaler.min_[j])
    return x.loc[:, features]


def score(df):
    x = prepare_for_model(df)
    return model.predict_proba(x)[:, 1]

st.title({'Vue d’ensemble':'Pilotage du risque de crédit','Exploration des données':'Exploration des portefeuilles','Scoring individuel':'Évaluation individuelle','Scoring par lot':'Scoring d’un portefeuille','Performance & explicabilité':'Performance & explicabilité','Méthodologie':'Méthodologie du projet'}[page])
st.caption("CreditScope • Outil d'aide à la décision en matière de risque de crédit • Résultats indicatifs")

if page=='Vue d’ensemble':
    st.markdown(
    '''
    <div class="hero" style="text-align: center;">
        <h2>Intelligence du risque de crédit</h2>
        <p>Analysez votre portefeuille, évaluez les dossiers et interprétez les résultats d’un modèle de scoring.</p>
    </div>
    ''',
    unsafe_allow_html=True
)
    st.info('Projet fondé sur 50 000 dossiers, issus des bases prêts, clients et bureau de crédit (rapport de recherche).')
    a,b,c,d=st.columns(4)
    a.metric('Dossiers étudiés','50 000');b.metric('Défauts observés','4 297');c.metric('Taux de défaut','8,6 %');d.metric('Modèle final','Régression logistique')
    st.subheader('Du dossier client à une estimation de risque')
    st.markdown('**Données client et prêt** → **Prétraitement / ingénierie** → **Modèle entraîné** → **Probabilité de défaut** → **Analyse des facteurs**')
    st.success('Modèle chargé' if model is not None else 'Ajoutez model_data.joblib (généré à la fin du notebook) pour activer le scoring.')
    left,right=st.columns(2)
    with left:
        st.subheader('Répartition du portefeuille')
        st.plotly_chart(px.pie(names=['Sans défaut','En défaut'],values=[45703,4297],hole=.62,color_discrete_sequence=['#246ba0','#e37b69']),use_container_width=True)
    with right:
        st.subheader('Statut technique')
        st.write('**Modèle** : '+('chargé et opérationnel' if model is not None else 'non chargé'))
        st.write(f'**Variables attendues** : {len(features)}')
        st.write('**Entrées brutes** : formulaire de saisie avec prétraitement automatique conforme aux règles du notebook.')
        st.info("Les statistiques présentées proviennent de l'étude sur des données bancaires historiques.")
elif page=='Exploration des données':
    files=st.file_uploader('Importer un CSV consolidé (avec éventuellement la cible default)',type='csv')
    if files:
        df=read_csv(files.getvalue());st.dataframe(df.head(100),use_container_width=True)
        a,b,c=st.columns(3);a.metric('Lignes',f'{len(df):,}');b.metric('Variables',len(df.columns));c.metric('Valeurs manquantes',int(df.isna().sum().sum()))
        if 'default' in df.columns:
            counts=df['default'].value_counts().rename_axis('Classe').reset_index(name='Nombre')
            st.plotly_chart(px.bar(counts,x='Classe',y='Nombre',title='Répartition de la cible'),use_container_width=True)
        numeric=df.select_dtypes(include='number').columns.tolist()
        if numeric:
            selected=st.selectbox('Variable à explorer',numeric)
            st.plotly_chart(px.histogram(df,x=selected,color='default' if 'default' in df else None,marginal='box'),use_container_width=True)
    else:
        st.info("Importez un fichier CSV contenant vos données de crédit pour explorer la composition du portefeuille, visualiser les principales caractéristiques des emprunteurs et analyser les indicateurs de risque disponibles.")
        st.caption("ℹ️Cette page est dédiée à l'analyse descriptive des données "
                   "et ne réalise pas de prédictions individuelles."
                   
        )
elif page=='Scoring individuel':
    st.markdown('### 🧾 Formulaire de demande de crédit')
    st.info("Renseignez les informations du demandeur et les caractéristiques du crédit. CreditScope vous donne l'estimation du risque de défaut.")
    if model is None or not features:
        st.warning('Ajoutez model_data.joblib à côté de app.py pour activer la prédiction.')
    else:
        expected = {'age','loan_tenure_months','number_of_open_accounts','credit_utilization_ratio',
                    'residence_type_Owned','residence_type_Rented','loan_purpose_Home',
                    'loan_purpose_Personal','loan_type_Unsecured','loan_to_income',
                    'delinquency_ratio','avg_dpd_per_delinquency'}
        if set(features) != expected:
            st.error('Le modèle chargé ne possède pas les 12 variables attendues par ce formulaire. Vérifiez le fichier joblib.')
        else:
            with st.form('credit_application'):
                st.subheader('👤 Profil de l’emprunteur')
                c1,c2,c3=st.columns(3)
                age=c1.number_input('Âge (années)',min_value=18,max_value=100,value=35,step=1)
                income=c2.number_input('Revenu annuel (€)',min_value=0.01,value=30000.0,step=500.0)
                residence=c3.selectbox('Situation résidentielle',['Propriétaire','Locataire','Crédit immobilier en cours'])
                st.subheader('💳 Crédit demandé')
                c1,c2,c3=st.columns(3)
                amount=c1.number_input('Montant emprunté (€)',min_value=0.01,value=15000.0,step=500.0)
                tenure=c2.number_input('Durée du prêt (mois)',min_value=1,max_value=600,value=36,step=1)
                purpose=c3.selectbox('Objet du prêt',['Personnel','Immobilier','Automobile','Études'])
                loan_type=st.radio('Type de prêt',['Garanti','Non garanti'],horizontal=True)
                st.subheader('📈 Historique de crédit')
                c1,c2,c3=st.columns(3)
                accounts=c1.number_input('Comptes de crédit ouverts',min_value=0,max_value=100,value=2,step=1)
                total_months=c2.number_input('Total des mois de crédit',min_value=1,max_value=1000,value=60,step=1)
                late_months=c3.number_input('Mois avec retard',min_value=0,max_value=1000,value=2,step=1)
                c1,c2=st.columns(2)
                total_dpd=c1.number_input('Nombre total de jours de retard',min_value=0,max_value=100000,value=15,step=1)
                utilization=c2.number_input('Utilisation du crédit (%)',min_value=0.0,max_value=100.0,value=35.0,step=1.0)
                submit=st.form_submit_button('🔎 Calculer la probabilité de défaut',type='primary',use_container_width=True)
            st.info(
                              "Les informations saisies sont automatiquement préparées "
                              "selon les règles du modèle."
                            )
            with st.expander('Méthodologie et unités utilisées'):
                
                 st.markdown("""
                            - **Ratio crédit/revenu** : montant du prêt / revenu, arrondi à 2 décimales.
                            - **Taux de retard** : 100 × mois en retard / total des mois, arrondi à 1 décimale.
                            - **Retard moyen** : jours de retard / mois en retard, arrondi à 1 décimale (0 sans retard).
                            - **Utilisation du crédit** : exprimée en pourcentage, de 0 à 100 %.
                             """) 
                  
            if submit:
                if late_months > total_months:
                    st.error('Les mois avec retard ne peuvent pas dépasser le total des mois de crédit.')
                elif late_months == 0 and total_dpd > 0:
                    st.error('Un total de jours de retard positif exige au moins un mois avec retard.')
                else:
                    values = {
                        'age':float(age), 'loan_tenure_months':float(tenure),
                        'number_of_open_accounts':float(accounts),
                        'credit_utilization_ratio':float(utilization),
                        'residence_type_Owned':float(residence=='Propriétaire'),
                        'residence_type_Rented':float(residence=='Locataire'),
                        'loan_purpose_Home':float(purpose=='Immobilier'),
                        'loan_purpose_Personal':float(purpose=='Personnel'),
                        'loan_type_Unsecured':float(loan_type=='Non garanti'),
                        'loan_to_income':round(float(amount)/float(income),2),
                        'delinquency_ratio':round(100.0*float(late_months)/float(total_months),1),
                        'avg_dpd_per_delinquency':round(float(total_dpd)/float(late_months),1) if late_months else 0.0,
                    }
                    try:
                        raw=pd.DataFrame([values])[features]
                        p=float(score(raw)[0])
                        st.success('Prédiction calculée à partir du modèle chargé.')
                        a,b,c=st.columns(3)
                        a.metric('Probabilité estimée de défaut',f'{p:.4%}')
                        b.metric('Ratio prêt / revenu',f"{values['loan_to_income']:.2f}")
                        c.metric('Taux de mois en retard',f"{values['delinquency_ratio']:.1f} %")
                        st.progress(min(max(p,0.0),1.0))
                        st.caption('Estimation statistique indicative, sans seuil d’acceptation ni décision automatique.')
                        with st.expander('Voir les 12 variables transmises au modèle'):
                            st.dataframe(raw.T.rename(columns={0:'Valeur avant standardisation'}),use_container_width=True)
                        if hasattr(model,'coef_'):
                            x=prepare_for_model(raw)
                            coefs=np.asarray(model.coef_).reshape(-1)
                            if len(coefs)==len(features):
                                contribution=pd.DataFrame({'Variable':features,'Contribution au log-odds':x.iloc[0].to_numpy(dtype=float)*coefs}).sort_values('Contribution au log-odds')
                                st.subheader('Facteurs contribuant à la prédiction')
                                st.plotly_chart(px.bar(contribution,x='Contribution au log-odds',y='Variable',orientation='h',color='Contribution au log-odds',color_continuous_scale='RdBu_r'),use_container_width=True)
                                st.caption('Contributions mathématiques sur l’échelle log-odds, sans interprétation causale.')
                    except Exception as e:
                        st.error(f'Prédiction impossible : {e}')
elif page=='Scoring par lot':
    st.subheader('📁 Mode avancé — Importation de données préparées')
    st.write('Importez un fichier CSV contenant les 12 variables attendues par le modèle pour calculer la probabilité de défaut de plusieurs emprunteurs simultanément.')
    st.info('**Fichier attendu :** un CSV contenant les **12 colonnes du modèle**, avec leurs noms exacts. Les valeurs doivent être **avant standardisation** : CreditScope applique automatiquement le MinMaxScaler enregistré.')
    st.warning('**Important :** un fichier bancaire quelconque (âge, salaire, montant du prêt…) ne peut pas être utilisé directement. Il faut d’abord calculer les ratios et encoder les catégories. Les probabilités sont indicatives, sans décision automatique.')
    if features:
        with st.expander('Voir les 12 colonnes obligatoires et leur signification'):
            descriptions = {
                'age':'Âge du client (années)',
                'loan_tenure_months':'Durée du prêt (mois)',
                'number_of_open_accounts':'Nombre de comptes de crédit ouverts',
                'credit_utilization_ratio':'Utilisation du crédit (0 à 100)',
                'residence_type_Owned':'Propriétaire : 1 si oui, sinon 0',
                'residence_type_Rented':'Locataire : 1 si oui, sinon 0',
                'loan_purpose_Home':'Prêt immobilier : 1 si oui, sinon 0',
                'loan_purpose_Personal':'Prêt personnel : 1 si oui, sinon 0',
                'loan_type_Unsecured':'Prêt non garanti : 1 si oui, sinon 0',
                'loan_to_income':'Montant du prêt / revenu annuel, arrondi à 2 décimales',
                'delinquency_ratio':'Mois en retard × 100 / mois de crédit, arrondi à 1 décimale',
                'avg_dpd_per_delinquency':'Jours de retard / mois en retard, arrondi à 1 décimale (0 sans retard)',
            }
            st.dataframe(pd.DataFrame({'Colonne obligatoire':features,'Signification':[descriptions.get(f,'Variable du modèle') for f in features]}),hide_index=True,use_container_width=True)
        st.download_button('📋 Télécharger le modèle CSV (12 colonnes)',pd.DataFrame(columns=features).to_csv(index=False).encode('utf-8-sig'),'modele_scoring_avance.csv','text/csv',help='Gabarit vide avec les noms exacts des colonnes. Remplissez une ligne par client.')
        st.caption('Les colonnes supplémentaires sont autorisées et seront conservées dans le résultat. La colonne default est facultative : elle n’est pas utilisée pour prédire.')
    else:
        st.warning('Chargez model_data.joblib pour connaître les colonnes attendues.')
    upload=st.file_uploader('Importer un CSV contenant les 12 variables préparées',type='csv',key='advanced_batch_upload')
    if upload is not None:
        if model is None or not features:
            st.error('Modèle indisponible. Placez model_data.joblib dans le même dossier que app.py, puis rechargez la page.')
        else:
            try:
                df=read_csv(upload.getvalue())
                if df.empty:
                    st.error('Le fichier ne contient aucun dossier. Ajoutez au moins une ligne de données.')
                else:
                    missing=[c for c in features if c not in df.columns]
                    if missing:
                        st.error('**Importation impossible : colonnes obligatoires manquantes :** '+', '.join(f'`{c}`' for c in missing))
                        st.info('Téléchargez le modèle CSV ci-dessus, puis renommez ou complétez vos colonnes. Les noms doivent correspondre exactement, y compris les majuscules et les underscores (_).')
                    else:
                        numeric=df[features].apply(pd.to_numeric,errors='coerce')
                        bad=(~np.isfinite(numeric.to_numpy(dtype=float)))
                        if bad.any():
                            errors=[]
                            for j,col in enumerate(features):
                                indices=np.flatnonzero(bad[:,j])[:5]
                                if len(indices):
                                    errors.append(f'{col} (lignes CSV : '+', '.join(str(i+2) for i in indices)+')')
                            st.error('**Valeurs manquantes ou non numériques :** '+ '; '.join(errors[:8]))
                            st.info('Corrigez les cellules signalées puis importez à nouveau le CSV. Les numéros de ligne incluent la ligne d’en-tête.')
                        else:
                            binary=['residence_type_Owned','residence_type_Rented','loan_purpose_Home','loan_purpose_Personal','loan_type_Unsecured']
                            invalid_binary=[c for c in binary if c in numeric and not numeric[c].isin([0,1]).all()]
                            if invalid_binary:
                                st.error('**Encodage invalide :** les colonnes suivantes doivent contenir uniquement 0 ou 1 : '+', '.join(invalid_binary))
                            elif ((numeric['credit_utilization_ratio']<0)|(numeric['credit_utilization_ratio']>100)).any() or ((numeric['delinquency_ratio']<0)|(numeric['delinquency_ratio']>100)).any():
                                st.error('**Échelle invalide :** credit_utilization_ratio et delinquency_ratio doivent être exprimés en pourcentage, entre 0 et 100.')
                            elif (numeric['age']<0).any() or (numeric['loan_tenure_months']<=0).any():
                                st.error('**Valeurs incohérentes :** âge positif et durée de prêt strictement positive requis.')
                            elif ((numeric['residence_type_Owned']+numeric['residence_type_Rented'])>1).any() or ((numeric['loan_purpose_Home']+numeric['loan_purpose_Personal'])>1).any():
                                st.error('**Catégories incompatibles :** un même dossier ne peut pas activer simultanément Owned et Rented, ni Home et Personal.')
                            else:
                                st.success(f'Fichier vérifié : {len(df):,} dossiers et les 12 variables requises sont présents.')
                                st.caption('Les contrôles vérifient la structure et certaines valeurs, mais ne peuvent pas certifier que les ratios ont été calculés selon le notebook ni que le CSV n’est pas déjà standardisé.')
                                st.dataframe(df.head(10),use_container_width=True)
                                if st.button('🔎 Calculer les probabilités de défaut',type='primary',use_container_width=True):
                                    p=score(df)
                                    out=df.copy()
                                    out['probabilite_defaut']=p
                                    st.metric('Dossiers évalués',f'{len(out):,}')
                                    st.dataframe(out.head(100),use_container_width=True)
                                    st.plotly_chart(px.histogram(out,x='probabilite_defaut',nbins=30,title='Distribution des probabilités estimées'),use_container_width=True)
                                    st.download_button('⬇️ Télécharger les résultats CSV',out.to_csv(index=False).encode('utf-8-sig'),'scoring_resultats.csv','text/csv')
            except Exception as e:
                st.error(f'Impossible de traiter ce fichier : {e}')
                st.info('Vérifiez que le fichier est un CSV lisible, avec une ligne d’en-tête et les 12 colonnes du gabarit.')
elif page=='Performance & explicabilité':
    st.info("Les indicateurs de performance sont calculés sur un jeu de test distinct des données d'entraînement.")
    if model is not None and hasattr(model,'coef_') and features:
        coefs=np.asarray(model.coef_).reshape(-1)
        if len(coefs)==len(features):
            imp=pd.DataFrame({'Variable':features,'Coefficient':coefs}).sort_values('Coefficient')
            st.plotly_chart(px.bar(imp,x='Coefficient',y='Variable',orientation='h',title='Coefficients du modèle logistique'),use_container_width=True)
            st.info("Interprétation des coefficients:")
            st.caption('Le signe indique la direction de l’association dans le modèle, toutes choses égales par ailleurs ; l’amplitude dépend de l’échelle des variables.')
            st.caption("Un coefficient positif indique "
                       "une augmentation du risque de défaut, tandis qu'un coefficient "
                       "négatif indique une diminution, à caractéristiques comparables. "
                       "Leur amplitude dépend de l'échelle des variables."
                      )
    else:st.warning('Chargez le modèle final pour consulter ses coefficients, si disponibles.')
    eval_file=st.file_uploader('Optionnel : jeu de test préparé avec colonne default',type='csv')
    if eval_file and model is not None:
        try:
            from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,classification_report
            df=read_csv(eval_file.getvalue());y=df['default'].astype(int);p=score(df)
            st.metric('ROC AUC sur le fichier fourni',f'{roc_auc_score(y,p):.3f}')
            fpr,tpr,_=roc_curve(y,p);st.plotly_chart(px.line(x=fpr,y=tpr,labels={'x':'Taux de faux positifs','y':'Sensibilité'},title='Courbe ROC'),use_container_width=True)
            threshold=st.slider('Seuil de classification exploratoire',0.05,0.95,0.50,0.05)
            st.dataframe(pd.DataFrame(confusion_matrix(y,p>=threshold),index=['Réel 0','Réel 1'],columns=['Prédit 0','Prédit 1']))
            st.text(classification_report(y,p>=threshold,zero_division=0))
            st.divider()
            st.subheader('Calibration des probabilités')
            st.caption('Chaque point compare la probabilité moyenne prédite au taux de défaut réellement observé dans un groupe de dossiers. Une bonne discrimination (AUC) ne garantit pas une bonne calibration.')
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score, f1_score
            bins = st.select_slider('Nombre de groupes pour la calibration', options=[5, 10, 15, 20], value=10)
            method = st.radio('Regroupement', ['Quantiles (effectifs comparables)', 'Intervalles de probabilité'], horizontal=True)
            strategy = 'quantile' if method.startswith('Quantiles') else 'uniform'
            frac_pos, mean_pred = calibration_curve(y.to_numpy(), p, n_bins=bins, strategy=strategy)
            cal_df = pd.DataFrame({'Probabilité moyenne prédite':mean_pred,'Fréquence de défaut observée':frac_pos})
            import plotly.graph_objects as go
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Calibration parfaite',line=dict(dash='dash',color='gray')))
            fig_cal.add_trace(go.Scatter(x=cal_df.iloc[:,0],y=cal_df.iloc[:,1],mode='lines+markers',name='Modèle CreditScope'))
            fig_cal.update_layout(xaxis_title='Probabilité moyenne prédite',yaxis_title='Fréquence de défaut observée',xaxis=dict(range=[0,1],tickformat='.0%'),yaxis=dict(range=[0,1],tickformat='.0%'),legend=dict(orientation='h'))
            st.plotly_chart(fig_cal,use_container_width=True)
            k1,k2=st.columns(2)
            k1.metric('Score de Brier (plus bas = meilleur)',f'{brier_score_loss(y,p):.4f}')
            k2.metric('Log loss (plus bas = meilleur)',f'{log_loss(y,p):.4f}')
            with st.expander('Données de la courbe de calibration'):
                st.dataframe(cal_df.style.format('{:.4%}'),use_container_width=True)
            st.caption('Les groupes sans observation peuvent ne pas apparaître avec le regroupement par intervalles. Une courbe proche de la diagonale est un indice de bonne calibration, sans preuve de validité hors échantillon.')

            st.divider()
            st.subheader('Comparaison des seuils de classification')
            st.caption('Le seuil de 50 % n’est pas une règle de décision recommandée. Les seuils suivants sont exploratoires ; leur choix nécessite une analyse des coûts et des contraintes métier.')
            threshold_values=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
            rows=[]
            for t in threshold_values:
                pred=(p>=t).astype(int)
                tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
                rows.append({'Seuil':t,'Défauts détectés (VP)':int(tp),'Défauts manqués (FN)':int(fn),'Faux positifs (FP)':int(fp),'Vrais négatifs (VN)':int(tn),'Précision':precision_score(y,pred,zero_division=0),'Rappel':recall_score(y,pred,zero_division=0),'F1-score':f1_score(y,pred,zero_division=0)})
            threshold_df=pd.DataFrame(rows)
            st.dataframe(threshold_df.style.format({'Seuil':'{:.0%}','Précision':'{:.1%}','Rappel':'{:.1%}','F1-score':'{:.1%}'}),use_container_width=True,hide_index=True)
            fig_threshold=go.Figure()
            for metric in ['Précision','Rappel','F1-score']:
                fig_threshold.add_trace(go.Scatter(x=threshold_df['Seuil'],y=threshold_df[metric],mode='lines+markers',name=metric))
            fig_threshold.update_layout(xaxis_title='Seuil de classification',yaxis_title='Valeur',xaxis=dict(tickformat='.0%'),yaxis=dict(range=[0,1],tickformat='.0%'))
            st.plotly_chart(fig_threshold,use_container_width=True)
            st.download_button('Télécharger la comparaison des seuils (CSV)',threshold_df.to_csv(index=False).encode('utf-8-sig'),'CreditScope_comparaison_seuils.csv','text/csv')

        except Exception as e:st.error(str(e))
else:
    st.markdown("""
### Pipeline de recherche

1. Fusion des trois bases : prêts, clients et historique de crédit.
2. Nettoyage, contrôles de cohérence, traitement des valeurs manquantes.
3. Création de ratios comportementaux et financiers (LTI, délinquance, gravité des retards).
4. Sélection de variables par Information Value et préparation des caractéristiques.
5. Comparaison de régression logistique, Random Forest et XGBoost ; essais de rééquilibrage et optimisation.
6. Évaluation ROC AUC, Gini, KS, déciles et importance des variables.
    """)

    st.info(
        "Préparation des données : CreditScope utilise les règles "
        "de prétraitement du notebook pour construire les variables "
        "nécessaires au modèle et calculer les probabilités de défaut.",
        icon="ℹ️"
    )
    
# Pied de page de la barre latérale
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <p style="text-align: center; color: #9CA3AF; font-size: 12px;">
        © 2026 - Analyse du Risque de Crédit
    </p>
    """,
    unsafe_allow_html=True
)