illerimport streamlit as st
import numpy as np
import pandas as pd 
from joblib import dump, load


# Chargeons le modèle entrainé
model = load('regression_model_saved.pkl')

# Fonction de prédiction et de probabilité

def prediction_faux_billets(features):
    X = pd.DataFrame([features])
    predictions = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return predictions, proba

# Créons notre interface Streamlit
def main():
    st.title('Détection de faux billet')
    st.write('Pouvez-vous entrer les caractéristiques du billet à identifier :')

    # Entrons des caractéristiques du billet
    diagonal = st.number_input('Diagonal', min_value=0.0)
    height_left = st.number_input('Hauteur gauche', min_value=0.0)
    height_right = st.number_input('Hauteur droite', min_value=0.0)
    margin_low = st.number_input('Marge inférieure', min_value=0.0)
    margin_up = st.number_input('Marge supérieure', min_value=0.0)
    length = st.number_input('Longueur', min_value=0.0)

    # Bouton prédire
    if st.button('Prédire'):
        features = [diagonal, height_left, height_right, margin_low, margin_up, length]
        predictions, proba = prediction_faux_billets(features)
        if predictions == True:
            st.write('Le billet est bon.')
        else:
            st.write('Le billet est faux.')
        st.write(f'Probabilité d\'obtention d\'un vrai billet : {proba[1]}')
        st.write(f'Probabilité d\'obtention d\'un faux billet : {proba[0]}')
        
        # Ajout de la jauge pour afficher la probabilité
        st.progress(proba[1])
        st.write(f'Probabilité d\'obtention d\'un vrai billet : {proba[1]}')

if __name__ == '__main__':
    main()
