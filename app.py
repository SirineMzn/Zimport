import os
import uuid
import pandas as pd
import openai  # Bibliothèque pour l'API OpenAI
import streamlit as st
import sys
import csv
import chardet
from collections import Counter
# Assurez-vous que votre clé OpenAI est dans les secrets de Streamlit
openai.api_key = st.secrets["openai_api_key"]

# Initialisation de l'état de session Streamlit
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.docs_loaded = False

session_id = st.session_state.id
# Augmente la limite de taille de champ
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)
# fonction pour convertir un fichier csv en txt
def convert_csv_to_txt(csv_file_path, txt_file_path,separateur_choisi,encodage, new_line):
    try:
        # Ouverture du fichier CSV en lecture
        with open(csv_file_path, mode='r', encoding=encodage) as fichier_csv:
            lecteur_csv = csv.reader(fichier_csv)
    
            # Ouverture du fichier texte en écriture
            with open(txt_file_path, mode='w', encoding=encodage) as fichier_txt:
                for ligne in lecteur_csv:
                    # Écriture de chaque ligne du CSV dans le fichier texte avec le séparateur choisi
                    fichier_txt.write(separateur_choisi.join(ligne) + new_line)
    
        print(f"Le fichier a été copié avec succès de {csv_file_path} à {txt_file_path}")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
# fonction pour vérifier l'extension d'un fichier ( csv ou txt )
def verifier_extension_fichier(fichier_path):
    _, extension = os.path.splitext(fichier_path)
    return extension.lower()
# fonction pour supprimer l'extension d'un fichier
def remove_file_extension(file_name):
    # Utilise os.path.splitext pour séparer le nom de fichier et son extension
    file_base_name = os.path.splitext(file_name)[0]
    return file_base_name
# détecter encodage
def open_file_with_auto_and_manual_encodings(fichier, encodings_list=None):
    """
    Détecte l'encodage automatiquement puis essaie d'ouvrir un fichier avec différents encodages.
    
    Paramètres :
    fichier : str
        Chemin vers le fichier à lire.
    encodings_list : list, optionnel
        Liste d'encodages à essayer après celui détecté automatiquement. Si None, une liste par défaut est utilisée.
    
    Retourne :
    content : str
        Le contenu du fichier si un encodage fonctionne.
    encodage_utilisé : str
        L'encodage qui a fonctionné.
    """
    if encodings_list is None:
        encodings_list = ['utf-8', 'latin1', 'cp1252','ascii']  # Liste par défaut d'encodages à essayer

    # D'abord, détecter l'encodage automatiquement
    with open(fichier, 'rb') as file:
        raw_data = file.read(1000)  # Lire tout le fichier pour plus de précision
    result = chardet.detect(raw_data)
    detected_encoding = result['encoding']
    print(f"Encodage détecté : {detected_encoding}")
    
    # Si l'encodage détecté n'est pas déjà dans la liste, l'ajouter en première position
    if detected_encoding and detected_encoding not in encodings_list:
        encodings_list.insert(0, detected_encoding)
    
    # Essayer les encodages dans la liste, en commençant par celui détecté
    for encodage in encodings_list:
        try:
            with open(fichier, 'r', encoding=encodage) as file:
                content = file.read()
            # Si l'ouverture et la lecture réussissent, on retourne le contenu et l'encodage utilisé
            return content, encodage
        except UnicodeDecodeError as e:
            print(f"L'encodage {encodage} a échoué avec une erreur : {e}")
        except Exception as e:
            print(f"Une autre erreur est survenue avec l'encodage {encodage} : {e}")
    
    # Si aucun encodage ne fonctionne, on lève une exception
    raise ValueError("Aucun encodage dans la liste n'a permis d'ouvrir correctement le fichier.")

#détécter le séparateur et un séparateur qui ne figure pas dans le fichier, si possible
def detect_delimiter(filename, res):
    with open(filename, 'r',encoding=res) as csvfile:      

        # Lire les premières lignes pour avoir un échantillon du fichier

        contenu = csvfile.read()    

        # Compter les occurrences de quelques séparateurs communs
        delimiter_counts = Counter()
        delimiters = [',', ';', '\t', '|']  # Liste de séparateurs courants à tester       
        for delim in delimiters:
            delimiter_counts[delim] += contenu.count(delim)

        print(f"Delimiter counts: {delimiter_counts}")

        # Sélectionner le séparateur le plus fréquent dans l'échantillon

    probable_delimiter = max(delimiter_counts, key=delimiter_counts.get)

    separateur_inexistant = None
    minimum = min(delimiter_counts, key=delimiter_counts.get)
# Vérifier si un séparateur de la liste est absent dans le fichier
    if delimiter_counts[minimum] ==0 :
        separateur_inexistant = minimum
    else:
        if minimum == '|': 
            contenu = contenu.replace('|', '\ ')
            with open(filename, 'w', encoding=res) as csvfile:
                csvfile.write(contenu)
                separateur_inexistant = minimum
        if minimum == '\t':
            contenu = contenu.replace('\t', ' ')  
            with open(filename,'+w', encoding=res) as csvfile:
                csvfile.write(contenu)
                separateur_inexistant = minimum
                          
            
    return probable_delimiter,separateur_inexistant
# Fonction pour lire le fichier et séparer les données en deux populations
# Uploader de fichier Excel
uploaded_file = st.file_uploader("Please upload a csv or Text file only", type=["csv", "txt"])
   

