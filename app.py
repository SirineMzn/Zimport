import os
import uuid
import pandas as pd
import openai  # Bibliothèque pour l'API OpenAI
import streamlit as st
import sys
import csv
import chardet
from collections import Counter
from io import StringIO
import io
import random
import bcrypt
# Assurez-vous que votre clé OpenAI est dans les secrets de Streamlit
openai.api_key = st.secrets["API_key"]["openai_api_key"]

hashed_password_user1 = st.secrets["auth"]["HASHED_PASSWORD_Tim"]
hashed_password_user2 = st.secrets["auth"]["HASHED_PASSWORD_Hugo"]
hashed_password_user3 = st.secrets["auth"]["HASHED_PASSWORD_Thomas_TR"]
hashed_password_user4 = st.secrets["auth"]["HASHED_PASSWORD_Paul"]
hashed_password_user5 = st.secrets["auth"]["HASHED_PASSWORD_Auriane"]
hashed_password_user6 = st.secrets["auth"]["HASHED_PASSWORD_Jordan"]
hashed_password_user7 = st.secrets["auth"]["HASHED_PASSWORD_Ben_Loyez"]
hashed_password_user8 = st.secrets["auth"]["HASHED_PASSWORD_Robin"]
hashed_password_user9 = st.secrets["auth"]["HASHED_PASSWORD_Baptiste"]
hashed_password_user10 = st.secrets["auth"]["HASHED_PASSWORD_Bouchra"]
hashed_password_user11 = st.secrets["auth"]["HASHED_PASSWORD_Sophia"]
hashed_password_user12 = st.secrets["auth"]["HASHED_PASSWORD_Thomas_TRA"]
hashed_password_user13 = st.secrets["auth"]["HASHED_PASSWORD_Solene"]
hashed_password_user14 = st.secrets["auth"]["HASHED_PASSWORD_Sisi"]
hashed_password_user15 = st.secrets["auth"]["HASHED_PASSWORD_Zaira"]
hashed_password_user16 = st.secrets["auth"]["HASHED_PASSWORD_François"]


# Simuler une base d'utilisateurs
USERS = {
    "Timothée Verluise": hashed_password_user1,
    "Hugo Andries": hashed_password_user2,
    "Thomas Tremblay": hashed_password_user3,
    "Paul Laloue": hashed_password_user4,
    "Auriane Giovannoni": hashed_password_user5,
    "Jordan Widom": hashed_password_user6,
    "Benjamin Loyez": hashed_password_user7,
    "Robin David": hashed_password_user8,
    "Baptiste Leroux": hashed_password_user9,
    "Bouchra Demanne": hashed_password_user10,
    "Sophia Mana": hashed_password_user11,
    "Thomas Tran": hashed_password_user12,
    "Solène Claudel": hashed_password_user13,
    "Sisi": hashed_password_user14,
    "Zaira Cosman": hashed_password_user15,
    "François Fanuel": hashed_password_user16
}
# Fonction pour vérifier le mot de passe
def verifier_mot_de_passe(mot_de_passe_saisi, hashed_password):
    return bcrypt.checkpw(mot_de_passe_saisi.encode('utf-8'), hashed_password.encode('utf-8'))


# Vérifier si l'utilisateur est déjà authentifié
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Entrer le nom d'utilisateur et le mot de passe si non authentifié
if not st.session_state.authenticated:
    
# Interface Streamlit pour l'authentification
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS:
            hashed_password = USERS[username]
            if verifier_mot_de_passe(password, hashed_password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome {username} !")
                
            else:
                st.error("Incorrect password.")
        else:
            st.error("ncorrect username.")

# Si authentifié, afficher l'interface de l'application
if st.session_state.authenticated:
    # Initialisation de l'état de session Streamlit
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
        st.session_state.file_name = None
        st.session_state.file_content = None
        st.session_state.file_extension = None
        st.session_state.docs_loaded = False
        st.session_state.data_processed = False

    session_id = st.session_state.id if "id" in st.session_state else uuid.uuid4()
    # Augmente la limite de taille de champ
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    # fonction pour détecter l'encodage d'un fichier
    def detect_encoding(content):
        """Détecte l'encodage à partir d'un contenu en mémoire."""
        result = chardet.detect(content)
        return result['encoding']
    # fonction pour convertir un fichier csv en txt
    def convert_csv_to_txt(csv_file, separateur_choisi, encodage, new_line):
        try:
            txt_buffer = io.StringIO()
            csv_reader = csv.reader(io.TextIOWrapper(csv_file, encoding=encodage))
            for ligne in csv_reader:
                txt_buffer.write(separateur_choisi.join(ligne) + new_line)
            txt_buffer.seek(0)
            return txt_buffer
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
            return None
    # fonction pour vérifier l'extension d'un fichier ( csv ou txt )
    def verifier_extension_fichier(fichier_path):
        _, extension = os.path.splitext(fichier_path)
        return extension.lower()
    # fonction pour supprimer l'extension d'un fichier
    def remove_file_extension(file_name):
        # Utilise os.path.splitext pour séparer le nom de fichier et son extension
        return os.path.splitext(file_name)[0]
    # détecter encodage
    def open_file_with_auto_and_manual_encodings(content, encodings_list=None):
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
        #Essaye de décoder le contenu en utilisant plusieurs encodages.
        if encodings_list is None:
            encodings_list = ['utf-8', 'latin1', 'cp1252', 'ascii']
        
        # Détection automatique de l'encodage
        detected_encoding = detect_encoding(content)
        if detected_encoding and detected_encoding not in encodings_list:
            encodings_list.insert(1, detected_encoding)
        
        # Essai de décodage avec chaque encodage
        for encodage in encodings_list:
            try:
                decoded_content = content.decode(encodage)
                return decoded_content, encodage
            except UnicodeDecodeError:
                pass
        
        # Si aucun encodage ne fonctionne
        raise ValueError("Aucun encodage dans la liste n'a permis de décoder le contenu.")

    #détécter le séparateur et un séparateur qui ne figure pas dans le fichier, si possible
    def detect_delimiter(encoded_content, encoding):
        """
        Détecte le délimiteur le plus probable et un séparateur inexistant dans le contenu encodé en mémoire.
        
        Arguments :
        - encoded_content (bytes) : Contenu encodé du fichier CSV.
        - encoding (str) : Encodage du contenu.

        Retourne :
        - probable_delimiter (str) : Le délimiteur le plus probable dans le fichier.
        - separateur_inexistant (str ou None) : Un séparateur inexistant dans le fichier, s'il y en a un.
        - modified_content (str) : Contenu modifié en mémoire si un remplacement a été nécessaire, sinon le contenu d'origine.
        """
        # Décoder le contenu pour l'analyse
        decoded_content = encoded_content.decode(encoding)
        
        # Compter les occurrences des délimiteurs communs
        delimiters = [',', ';', '\t', '|']  # Liste de séparateurs courants
        delimiter_counts = Counter({delim: decoded_content.count(delim) for delim in delimiters})

        # Déterminer le délimiteur le plus fréquent
        probable_delimiter = max(delimiter_counts, key=delimiter_counts.get)

        # Identifier un séparateur inexistant dans le fichier, si possible
        separateur_inexistant = next((delim for delim, count in delimiter_counts.items() if count == 0), None)
        
        # Vérifier si un remplacement est nécessaire et modifier le contenu si besoin
        modified_content = decoded_content
        if separateur_inexistant is None:
            # Si tous les séparateurs existent, identifier celui qui est le moins fréquent pour le remplacer
            least_frequent_delimiter = min(delimiter_counts, key=delimiter_counts.get)
            
            if least_frequent_delimiter == '|':
                # Remplacer '|' par un espace protégé si besoin
                modified_content = modified_content.replace('|', r'\ ')
                separateur_inexistant = '|'
            
            elif least_frequent_delimiter == '\t':
                # Remplacer '\t' par un espace pour éviter les conflits
                modified_content = modified_content.replace('\t', ' ')
                separateur_inexistant = '\t'

        return probable_delimiter, separateur_inexistant, modified_content
    def display_correction_table(malformed_rows, header,types):
        """
        Displays an interactive table of malformed rows for correction, with validation and restore buttons.
        Adds options to select multiple rows and columns to perform actions on selected cells.
        """
        # Ajout du CSS personnalisé pour réduire la taille du texte dans la barre latérale
        st.markdown(
            """
            <style>
            .sidebar-text {
                font-size: 12px; /* Ajustez la taille selon vos besoins */
                color: #333333; /* Facultatif: changez la couleur du texte */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Afficher le texte avec une classe CSS personnalisée
        if malformed_rows:
            st.write("## Detected malformed rows")
            
            max_columns = max(len(row) for row in malformed_rows)
            extended_header = header + [f"Unnamed_Column_{i+1}" for i in range(len(header), max_columns)]
            if len(types) == len(header):
                extended_types = types + [""] * (max_columns - len(header))  # Types pour les colonnes supplémentaires
            else:
                extended_types = [""] * max_columns
            # Adjust rows to match the maximum number of columns and replace None with ""
            adjusted_rows = [
                [field if field is not None else "" for field in (row + [None] * (max_columns - len(row)))]
                for row in malformed_rows
            ]

            # Create DataFrame for display
            df_display = pd.DataFrame(adjusted_rows)

            # Fusionner les types et noms des colonnes pour affichage
            merged_headers = [
                f"{col}\n({type_})" if type_ else col  # Ajouter type entre parenthèses
                for col, type_ in zip(extended_header, extended_types)
            ]
            df_display.columns = merged_headers  # Appliquer les en-têtes fusionnés
            # Décaler l'index pour commencer à 1
            df_display.index = df_display.index + 1
            if df_display.empty:
                st.warning("The table is empty and cannot be displayed.")
                return None, corrected

            # Initialize original values and modified state if not already done
            if 'cell_initial' not in st.session_state:
                if not df_display.empty:
                    st.session_state.cell_initial = {(i, j): df_display.iat[i, j] for i in range(df_display.shape[0]) for j in range(df_display.shape[1])}
                    st.session_state.cell_current = df_display.copy()
                    st.session_state.modified_cells = [(1,1)]  # Initialisation par défaut si le tableau n'est pas vide
                else:
                    return None, None  # Retourne None si le tableau est vide
                
                st.session_state.corrected = []
            if 'validated_rows' not in st.session_state:
                st.session_state.validated_rows = []
            if 'selection_rows' not in st.session_state:
                st.session_state.selection_rows = list(range(1,df_display.shape[0]+1))
            # Display the interactive table
            corrected_df = st.data_editor(st.session_state.cell_current, use_container_width=True)

            # Detect changes to record modified cells
            for i in range(corrected_df.shape[0]):
                for j in range(corrected_df.shape[1]):
                    current_value = corrected_df.iat[i, j]
                    if current_value != st.session_state.cell_current.iat[i, j]:
                        st.session_state.modified_cells.append((i, j))
                        st.session_state.cell_current.iat[i, j] = current_value  # Update modified value

            # Row and column selection inputs in the sidebar
            if not corrected_df.empty:
                selected_rows = st.sidebar.multiselect(
                    "Select multiple/single row numbers.",
                    options=st.session_state.selection_rows)
                    
                col_selection = st.sidebar.selectbox(
                    "Select a single column",
                    options=list(range(df_display.shape[1])),
                    format_func=lambda x: df_display.columns[x]
                )
                st.sidebar.write('<p class="sidebar-text">After cells selection, click on Merge button to merge the selected cells with the next cell on the right.</p>', unsafe_allow_html=True)
                st.sidebar.write('<p class="sidebar-text">After cells selection, click on Delete button to delete the selected cells.</p>', unsafe_allow_html=True)
                st.sidebar.write('<p class="sidebar-text">Both Merge and Delete operations result in shifting the rest of the table to the left.</p>', unsafe_allow_html=True)
                # Convert input text to index lists
            else:
                st.sidebar.write("The table is empty. No operations can be performed.")
                selected_rows = []
                col_selection = []
                return st.session_state.cell_current, st.session_state.corrected
            if st.sidebar.button("Merge"):
                current_indices = st.session_state.cell_current.index.tolist()  # Récupérer les indices actuels

                for row_index in selected_rows:
                    if row_index not in current_indices:
                        st.sidebar.error(f"Row {row_index} is invalid or has been deleted.")
                        continue
                    
                    # Convertir l'index sélectionné en position réelle
                    pos = current_indices.index(row_index)

                    
                    if col_selection < st.session_state.cell_current.shape[1] - 1:  # Vérifier que la fusion est possible
                        try:
                                # Effectuer la fusion
                            st.session_state.cell_current.iat[pos, col_selection] = (
                                str(st.session_state.cell_current.iat[pos, col_selection]) +
                                    " " +
                                    str(st.session_state.cell_current.iat[pos, col_selection + 1])
                                )
                                # Décaler les cellules vers la gauche
                            st.session_state.cell_current.iloc[pos, col_selection + 1:] = (
                                    st.session_state.cell_current.iloc[pos, col_selection + 2:].tolist() + [""]
                                )
                        except IndexError:
                            st.sidebar.warning(f"Invalid operation on row {row_index}, column {col_selection}.")
                            continue
                st.sidebar.success("Merge completed.")
                st.rerun()  # Redémarrer pour appliquer les modifications


            # Delete selected cells and shift remaining cells accordingly
            if st.sidebar.button("Delete"):
                current_indices = st.session_state.cell_current.index.tolist()  # Récupérer les indices actuels

                for row_index in selected_rows:
                    if row_index not in current_indices:
                        st.sidebar.error(f"Row {row_index} is invalid or has been deleted.")
                        continue

                    # Convertir l'index sélectionné en position réelle
                    pos = current_indices.index(row_index)

                    try:
                            # Supprimer la cellule et décaler les cellules suivantes vers la gauche
                        st.session_state.cell_current.iloc[pos, col_selection:] = (
                            st.session_state.cell_current.iloc[pos, col_selection + 1:].tolist() + [""]
                            )
                    except IndexError:
                        st.sidebar.warning(f"Invalid operation on row {row_index}, column {col_selection}.")
                        continue
                st.sidebar.success("Delete completed.")
                st.rerun()  # Redémarrer pour appliquer les modifications
            st.sidebar.write('<p class="sidebar-text">Press the validation button to send edited rows to storage.</p>', unsafe_allow_html=True)

            # Validation des lignes modifiées
            if st.sidebar.button("Validation"):
                if st.session_state.modified_cells or len(st.session_state.cell_current) == 1:
                    
                    valid_rows = []  # Liste pour stocker les lignes à supprimer après validation

                    # Assurez-vous que les indices sélectionnés correspondent aux indices actuels
                    current_indices = st.session_state.cell_current.index.tolist()

                    for row_index in selected_rows:
                        if row_index not in current_indices:
                            st.sidebar.warning(f"Row {row_index} is invalid or has been deleted.")
                            continue

                        # Convertir l'index sélectionné en position réelle
                        pos = current_indices.index(row_index)

                        total_fields_count = len(corrected_df.iloc[pos, :len(header)])
                        extra_fields_empty = all(
                            str(field).strip() == "" or pd.isna(field)
                            for field in corrected_df.iloc[pos, len(header):]
                        )

                        if total_fields_count == len(header) and extra_fields_empty:
                            valid_row = corrected_df.iloc[pos, :len(header)].tolist()
                            st.session_state.corrected.append(valid_row)
                            valid_rows.append(pos)  # Utiliser la position, pas l'index
                            st.sidebar.success(f"Row {row_index} validated successfully!")
                        else:
                            st.sidebar.warning(f"Row {row_index} contains filled extra columns and cannot be validated.")

                    if valid_rows:
                        # Convertir les positions en indices actuels
                        rows_to_drop = st.session_state.cell_current.index[valid_rows]
                        
                        # Supprimer les lignes validées en utilisant les indices actuels
                        st.session_state.cell_current.drop(rows_to_drop, inplace=True)

                        # Réinitialiser l'index pour commencer à 1
                        st.session_state.cell_current.reset_index(drop=True, inplace=True)
                        st.session_state.cell_current.index = st.session_state.cell_current.index + 1
                        st.session_state.selection_rows =list(range(1,st.session_state.cell_current.shape[0]+1)) # Mettre à jour les lignes sélectionnées
                        st.rerun()  # Redémarrer pour forcer la mise à jour
            return st.session_state.cell_current,None
        else:
            st.success("No malformed rows detected!")
            return None, None



    # Étape 1 : Affichage de l'uploader si aucun fichier n'est chargé
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("Please upload a csv or Text file only", type=["csv", "txt"])
    
        st.info("Welcome to Zimport! Please upload a csv or Text file only, your file will be analyzed immediately.")

        if uploaded_file:
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_content = uploaded_file.read()
            st.session_state.file_extension = verifier_extension_fichier(uploaded_file.name)
            st.session_state.docs_loaded = True
            
            st.rerun()

    # Étape 2 : Affichage et traitement après téléchargement du fichier
    if st.session_state.file_uploaded:
        st.write(f"### File Loaded: {st.session_state.file_name}")
        content = st.session_state.file_content
        extension = st.session_state.file_extension
        base_file_name = remove_file_extension(st.session_state.file_name)

        
        # Détecter l'encodage
        try:
            decoded_content, encodage = open_file_with_auto_and_manual_encodings(content)
        except ValueError as e:
            st.error(f"Erreur d'encodage : {e}")
            st.stop()
        # détécter le caractère de fin de ligne
        
        newlines = {'\n': decoded_content.count('\n'), '\r\\n': decoded_content.count('\r\n'), '\r': decoded_content.count('\r')}
        for key, value in newlines.items():
            if value != 0 :
                new_line = key
        number_lines = newlines[new_line]
            # Appeler la fonction pour détecter le délimiteur et obtenir le contenu modifié si nécessaire
        delimiter, sep_inexistant, modified_content = detect_delimiter(content, encodage)

        # Vérifier si une conversion est nécessaire

        # Si le fichier est un CSV, le convertir en TXT
        if extension == '.csv':
            # Utiliser `modified_content` pour la conversion
            texte = convert_csv_to_txt(modified_content, sep_inexistant or delimiter, encodage, new_line)
            delimiter = sep_inexistant or delimiter
        else:
            # Sinon, utiliser le contenu tel quel
            texte = modified_content

        # Boutons pour télécharger ou réinitialiser
        if st.button("Reset File Upload"):
            for key in list(st.session_state.keys()):
                if key != "authenticated":
                    del st.session_state[key]

            st.rerun()

        # Découper le contenu en lignes
        lines = modified_content.splitlines()
        total_lines = len(lines)    
        # Initialiser les listes pour les lignes "saines" et "malades"
        saines = []
        malades = []
        store_list = []  # Stocke les lignes incomplètes pour compléter les lignes suivantes
        logs_pi = []
        logs_sigma = []
        # Lire la première ligne pour obtenir le nombre attendu de colonnes
        first_line = lines[0].split(delimiter)
        number = len(first_line)  # Nombre attendu de colonnes
        
        most_common_count = number  # Utilisez la première ligne comme référence pour la structure
        # Initialiser les variables
        index = 1  # Commence après l'en-tête
        store_list = []
        old_index_buffer = [] 
        saines_pi = [] # Liste tampon pour capturer tous les indices d'origine
        nb_malades = 0  
        nb_saines = 0
        nb_saines_pi = 0
        line_breaks = 0
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.split(delimiter)  # Diviser chaque ligne selon le délimiteur

            # Ajouter l'index actuel au buffer pour cette itération
            old_index_buffer.append(index)

            if len(line) > number:
                # Trop de champs dans la ligne -> Ajouter à "malades"
                malades.append(line)
                nb_malades += 1
                logs_sigma.append((old_index_buffer.copy()[0], 0))

                old_index_buffer.clear()
                index += 1
            elif len(line) == number:
                # Ligne correcte -> Ajouter à "saines"
                saines.append(line)
                nb_saines += 1
                old_index_buffer.clear()

                index += 1  # Passer à la ligne suivante

            elif len(line) < number:
                line_breaks += 1
                # Ligne trop courte -> Fusionner avec la suivante
                if store_list:
                    next_line = line
                    line = store_list
                else:
                    # Charger la ligne suivante pour tenter une fusion
                    next_index = index + 1
                    if next_index < len(lines):
                        next_line = lines[next_index].split(delimiter)
                        old_index_buffer.append(next_index)  # Ajouter la ligne suivante au buffer
                        index += 1  # Ligne suivante utilisée, donc incrémenter
                    else:
                        next_line = []

                # Fusionner les colonnes
                if next_line:
                    milieu = line[-1] + next_line[0]
                    result_line = line[:-1] + [milieu] + next_line[1:]
                else:
                    result_line = line

                # Vérifier la ligne après fusion
                if len(result_line) == number:
                    # Ligne correcte après fusion -> Ajouter à "saines"
                    saines_pi.append(result_line)
                    nb_saines_pi += 1
                    # Enregistrer le mapping old_index_buffer -> new_index
                    logs_pi.append((old_index_buffer.copy()[0], 0))
                    old_index_buffer.clear()

                    store_list = []  # Réinitialiser le stockage temporaire
                elif len(result_line) < number:
                    # Si encore trop courte -> Stocker pour une future fusion
                    store_list = result_line
                elif len(result_line) > number:
                    # Si trop longue -> Ajouter à "malades"
                    malades.append(result_line)
                    nb_malades += 1
                    logs_sigma.append((old_index_buffer.copy(), 0))

                    # Enregistrer les indices et effacer le buffer
                    old_index_buffer.clear()

                # Avancer l'index après traitement
                index += 1
            else:
                # En cas d'anomalie inattendue
                st.warning(f"Unexpected case at line {index}. Skipping.")
                old_index_buffer.clear()
                index += 1
        
        def déterminer_types(first_line, lignes_aleatoires):
            types = []
            prompt = (
                f"Tu es un assistant qui détermine les types de données des champs d'un fichier CSV. "
                f"Voici l'entête des données : {first_line}. "
                f"Voici quelques exemples de données correctes : {lignes_aleatoires}. "
                f"si tu trouves des champs date, tu peux les identifier comme tels. "
                f"Réponds seulement avec une liste de types de données pour chaque champ dans l'ordre d'apparition des champs."
            )
            
            # Utiliser la nouvelle interface API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            # Extraction et nettoyage de la réponse
            raw_output = response['choices'][0]['message']['content'].strip()
            
            # Nettoyer les balises Markdown et autres caractères indésirables
            cleaned_output = raw_output.replace("```plaintext\n", "").replace("\n```", "").replace("[", "").replace("]", "").strip()
            
            # Transformation en liste
            types = [t.strip("'\" ") for t in cleaned_output.split(", ")]
            
            return types
        if len(saines)>10:
            lignes_aleatoires = random.sample(saines, 10)
        else: 
            lignes_aleatoires = random.sample(saines, len(saines))   
        types = déterminer_types(first_line,lignes_aleatoires)
        #types = ["String", "String", "Date", "String", "String", "String", "Date", "String", "String", "String", "Date", "String", "String"]
        st.info(f"Number of columns detected: {number},\n"
                f"Number of good lines: {nb_saines},\n"
                f"Number of line breaks: {line_breaks},\n"
                f"Number of bad lines: {nb_malades}")
        
        corrected_df,corrected=display_correction_table(malades, first_line,types)
        if corrected :
            # Convertir les lignes "saines" en DataFrame
            df_Pi = pd.DataFrame(saines_pi, columns=first_line)
            df_sigma = pd.DataFrame(corrected, columns=first_line)
            df_saines = pd.DataFrame(saines, columns=first_line)
            df_final = pd.concat([df_saines,df_Pi,df_sigma],axis=0)
                # Récupérer le nom de fichier original sans extension
                    # Calculer les indices de départ
            start_index_pi = nb_saines + 1  # Commence après les lignes "saines"
            start_index_sigma = nb_saines + nb_saines_pi + 1  # Commence après "saines" et "saines_pi"

            # Mettre à jour logs_pi (calcul automatique des nouveaux indices)
            for i in range(len(logs_pi)):
                ancien_index, nouveau_index = logs_pi[i]
                if nouveau_index == 0:  # Si le nouveau index est 0, calculer
                    logs_pi[i] = (ancien_index, start_index_pi)
                    start_index_pi += 1  # Incrémenter le nouvel index

            # Mettre à jour logs_sigma (calcul automatique des nouveaux indices)
            for i in range(len(logs_sigma)):
                ancien_index, nouveau_index = logs_sigma[i]
                if nouveau_index == 0:  # Si le nouveau index est 0, calculer
                    logs_sigma[i] = (ancien_index, start_index_sigma)
                    start_index_sigma += 1  # Incrémenter le nouvel index

            def generate_log_file(logs_pi, logs_sigma,total_lines,nb_saines,nb_saines_pi,nb_malades):
                content = "Summary of the file conversion:\n"
                content += f"Total lines in the original file: {total_lines}\n"
                content += f"Number of 'healthy' lines: {nb_saines}\n"
                content += f"Number of 'healthy' lines after merging: {nb_saines_pi}\n"
                content += f"Number of 'unhealthy' lines: {nb_malades}\n\n"
                content += f"Number of lines in the new file: {total_lines - nb_malades}\n\n"
                content += "Logs after treatement:\n"
                content += "Logs for Pi:\n"
                for ancien_index, nouveau_index in logs_pi:
                    content += f"Row: {ancien_index}, has became: {nouveau_index}\n"

                content += "\n"  # Ligne vide pour séparer les sections

                content += "Logs for Sigma:\n"
                for ancien_index, nouveau_index in logs_sigma:
                    content += f"Row: {ancien_index}, has became: {nouveau_index}\n"

                return content

            # Générer le contenu du fichier texte
            file_content = generate_log_file(logs_pi, logs_sigma,total_lines,nb_saines,nb_saines_pi,nb_malades)

            # Afficher un bouton de téléchargement
            st.download_button(
                label="Download logs",
                data=file_content,
                file_name=f"{base_file_name}_logs.txt",
                mime="text/plain"
            )        
                # Générer un fichier CSV téléchargeable avec le nom original
            csv_data = df_final.to_csv(index=False).encode(encodage)
            st.download_button(
                    label="Download your file as CSV",
                    data=csv_data,
                    file_name=f"{base_file_name}_saines.csv",
                    mime="text/csv"
                )

                # Générer un fichier TXT téléchargeable avec le nom original
            txt_data = df_final.to_csv(index=False, sep=delimiter).encode(encodage)
            st.download_button(
                    label="Download your file as TXT",
                    data=txt_data,
                    file_name=f"{base_file_name}_saines.txt",
                    mime="text/plain"
                )
