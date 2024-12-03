import os
import uuid
import pandas as pd
import openai  # Bibliothèque pour l'API OpenAI
import streamlit as st
import sys
import csv
import chardet
from collections import Counter
from io import StringIO,BytesIO
import io
import bcrypt
import time
import gc
import csv
openai.api_key = st.secrets["API_key"]["openai_api_key"]
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/zimport/uploads")
DATA_FOLDER = os.getenv("DATA_FOLDER", "/zimport/data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
uploads = os.listdir(UPLOAD_FOLDER)
contents = os.listdir(DATA_FOLDER)
for upload in uploads:
    st.write(upload)

 
# Créer le dossier si nécessaire
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
hashed_password_user16 = st.secrets["auth"]["HASHED_PASSWORD_Francois"]
 
 
# Simuler une base d'utilisateurs
USERS = {
    "TVE": hashed_password_user1,
    "HAN": hashed_password_user2,
    "TTR": hashed_password_user3,
    "PLA": hashed_password_user4,
    "AGI": hashed_password_user5,
    "JWI": hashed_password_user6,
    "BLO": hashed_password_user7,
    "RDA": hashed_password_user8,
    "BLE": hashed_password_user9,
    "BDE": hashed_password_user10,
    "SMA": hashed_password_user11,
    "TTV": hashed_password_user12,
    "SCL": hashed_password_user13,
    "Sisi": hashed_password_user14,
    "ZCO": hashed_password_user15,
    "FFA": hashed_password_user16
}
# Fonction pour vérifier le mot de passe
def verifier_mot_de_passe(mot_de_passe_saisi, hashed_password):
    return bcrypt.checkpw(mot_de_passe_saisi.encode('utf-8'), hashed_password.encode('utf-8'))
 
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory ,filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Supprime le fichier ou le lien symbolique
                    st.write(f"Le dossier {file_path} a été supprimé.")
                    time.sleep(2)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Supprime un sous-dossier vide
                    st.write(f"Le dossier {file_path} a été supprimé.")
                    time.sleep(2)
            except Exception as e:
                print(f"Erreur lors de la suppression {file_path}: {e}")
# Vérifier si l'utilisateur est déjà authentifié
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "malades" not in st.session_state:
    st.session_state.malades = False
if "saines" not in st.session_state:
    st.session_state.saines = False
 
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
                time.sleep(2)
                st.rerun()
            else:
                st.error("Incorrect password.")
        else:
            st.error("Incorrect username.")
 
# Si authentifié, afficher l'interface de l'application
if st.session_state.authenticated:
    # Initialisation de l'état de session Streamlit
   
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
       
 
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
 
    # fonction pour vérifier l'extension d'un fichier ( csv ou txt )
 
   
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
    def detect_delimiter(decoded_content):
        """
        Détecte le délimiteur le plus probable et un séparateur inexistant dans le contenu encodé en mémoire.
       
        Arguments :
        - encoded_content (bytes) : Contenu encodé du fichier CSV.
        - encoding (str) : Encodage du contenu.
 
        Retourne :
        - probable_delimiter (str) : Le délimiteur le plus probable dans le fichier.
        - separateur_inexistant (str ou None) : Un séparateur inexistant dans le fichier, s'il y en a un.
        - content (str) : Contenu modifié en mémoire si un remplacement a été nécessaire, sinon le contenu d'origine.
        """
       
       
        # Compter les occurrences des délimiteurs communs
        delimiters = [',', ';', '\t', '|']  # Liste de séparateurs courants
        delimiter_counts = Counter({delim: decoded_content.count(delim) for delim in delimiters})
 
        # Déterminer le délimiteur le plus fréquent
        probable_delimiter = max(delimiter_counts, key=delimiter_counts.get)
 
        # Identifier un séparateur inexistant dans le fichier, si possible
        separateur_inexistant = next((delim for delim, count in delimiter_counts.items() if count == 0), None)
       
        if separateur_inexistant is None:
            # Si tous les séparateurs existent, identifier celui qui est le moins fréquent pour le remplacer
            least_frequent_delimiter = min(delimiter_counts, key=delimiter_counts.get)
           
            if least_frequent_delimiter == '|':
                # Remplacer '|' par un espace protégé si besoin
                decoded_contentt = decoded_content.replace('|', r'\ ')
                separateur_inexistant = '|'
           
            elif least_frequent_delimiter == '\t':
                # Remplacer '\t' par un espace pour éviter les conflits
                decoded_content = decoded_content.replace('\t', ' ')
                separateur_inexistant = '\t'
 
        return probable_delimiter, separateur_inexistant, decoded_content
   
    if "file_path" not in st.session_state:
        st.session_state.file_path = None
    if "filename" not in st.session_state:
        st.session_state.filename = None

    # Étape 1 : Affichage de l'uploader si aucun fichier n'est chargé
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("Please upload a CSV or text file only.", type=["csv", "txt"])
            # Exporter les lignes malformées dans un fichier Excel
       
        st.info("Welcome to Zimport! Please upload a csv or Text file.The analysis will start automatically.")
        if uploaded_file:
            files_base = os.path.splitext(uploaded_file.name)[0]
            file_path =os.path.join(UPLOAD_FOLDER,uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File uploaded and saved at: {file_path}")
            st.session_state.file_path = file_path    
            st.session_state.filename = files_base        
            del uploaded_file
            gc.collect()
            st.session_state.file_uploaded = True
            st.rerun()
 
    # Étape 2 : Affichage et traitement après téléchargement du fichier
    if st.session_state.file_uploaded:
 
        # Boutons pour télécharger ou réinitialiser

 
        saines_path = os.path.join(DATA_FOLDER,st.session_state.filename +"_saines.csv")
        malades_path = os.path.join(DATA_FOLDER,st.session_state.filename+ "_malades.csv")
        @st.cache_data
        def load_and_process_data(file_path,saines_path,malades_path):
            """
            Charge et traite un fichier local (CSV ou TXT) et écrit les résultats directement dans des fichiers.
 
            Paramètres :
            - file_path : str
                Chemin vers le fichier à traiter.
            - output_dir : str
                Répertoire de sortie pour les fichiers traités.
 
            Retourne :
            - new_line : str
                Caractère de fin de ligne détecté.
            - delimiter : str
                Délimiteur détecté dans le fichier.
            - number : int
                Nombre attendu de colonnes basé sur l'en-tête.
            - logs_pi : list
                Logs des lignes fusionnées pour correction.
            - logs_sigma : list
                Logs des lignes malformées nécessitant une révision manuelle.
            - nb_saines : int
                Nombre de lignes valides.
            - nb_malades : int
                Nombre de lignes malformées.
            - nb_saines_pi : int
                Nombre de lignes fusionnées valides.
            - line_breaks : int
                Nombre de sauts de ligne corrigés.
            - first_line : list
                Liste des colonnes en-tête détectées.
            """
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
 
            # Créer le répertoire de sortie
           
 
            # Détecter l'extension
            _, extension = os.path.splitext(file_path)
 
            # Lire le fichier
            with open(file_path, "rb") as f:
                content = f.read()
 
            # Détecter l'encodage
            try:
                content, encodage = open_file_with_auto_and_manual_encodings(content)
            except ValueError as e:
                raise ValueError(f"Erreur d'encodage : {e}")
 
            # Détecter le caractère de fin de ligne
            newlines = {'\n': content.count('\n'), '\r\n': content.count('\r\n'), '\r': content.count('\r')}
            new_line = max(newlines, key=newlines.get)
 
            # Détecter le délimiteur
            delimiter, sep_inexistant, content = detect_delimiter(content)
 
            # Conversion CSV -> TXT si nécessaire
            if extension.lower() == '.csv':
                delimiter = sep_inexistant or delimiter
            # Libérer la mémoire
            del content
            gc.collect()
 
            # Variables de suivi
            index = 1  # Commence après l'en-tête
            store_list = []
            old_index_buffer = []
            nb_saines = 0
            nb_malades = 0
            nb_saines_pi = 0
            line_breaks = 0
            logs_pi = []
            logs_sigma = []
 
            # Ouvrir les fichiers de sortie
            saines_file = open(saines_path, mode="w", newline=new_line, encoding=encodage)
            malades_file = open(malades_path, mode="w", newline=new_line, encoding=encodage)
 
            saines_writer = csv.writer(saines_file)
            malades_writer = csv.writer(malades_file)
 
 
 
            with open(file_path, "r", encoding=encodage) as input_file, \
         open(saines_path, "w", newline=new_line, encoding=encodage) as saines_file, \
         open(malades_path, "w", newline=new_line, encoding=encodage) as malades_file:
            # Lire l'en-tête
                reader = csv.reader(input_file, delimiter=delimiter)
                counting = 0
                first_line = next(reader)
                number = len(first_line)
                # Écrire l'en-tête dans les fichiers
                saines_writer.writerow(first_line)
            # Traiter chaque ligne
                for line in reader:
                    if len(line) > number:
                        nb_malades += 1
                        malades_writer.writerow(line)
                    if len(line) == number:
                        nb_saines += 1
                        saines_writer.writerow(line)
                    elif len(line) < number:
                        if store_list:
                            next_line = line
                            line = store_list
                        else:
                            try:
                                next_line = next(reader)
                            except StopIteration:
                                # Dernière ligne atteinte
                                nb_malades += 1
                                malades_writer.writerow(line)
                                break

                        if not next_line:  # Vérifiez si `next_line` est vide
                            nb_malades += 1
                            malades_writer.writerow(line)
                            break
                        milieu = line[-1] + next_line[0]
                        result_line = line[:-1] + [milieu] + next_line[1:]
                        if len(result_line) == number:
                            line_breaks += 1
                            saines_writer.writerow(result_line)
                            store_list = []
                        if len(result_line) < number:
                            store_list = result_line
                        if len(result_line) > number:
                            nb_malades += 1
                            malades_writer.writerow(result_line)
                    counting += 1
            # Fermer les fichiers de sortie
            saines_file.close()
            malades_file.close()
 
 
            return counting,encodage,new_line, delimiter, number, logs_pi, logs_sigma, nb_saines, nb_malades, nb_saines_pi, line_breaks, first_line
 
        counting,encodage,new_line,delimiter,number, logs_pi, logs_sigma, nb_saines, nb_malades, nb_saines_pi, line_breaks, first_line = load_and_process_data(st.session_state.file_path,saines_path,malades_path)
        st.cache_data.clear()
        st.info(f"File Analysis Summary: Columns Detected: {number},\n"
                f"Valid Rows: {nb_saines} (lines without errors),\n"
                f"Line Breaks Fixed: {line_breaks} (extra line breaks detected and removed),\n"
                f"Invalid Rows: {nb_malades} (manual review required for rows with too many columns).")
        st.write(f"Les fichiers valides sont enregistrés dans : {saines_path}")
        st.write(f"Les fichiers malformés sont enregistrés dans : {malades_path}")
        st.session_state.saines_path = saines_path
        if nb_malades > 0:
            if "encodage" not in st.session_state:
                
                st.session_state.encodage = encodage
                st.session_state.new_line = new_line
                st.session_state.first_line = first_line
                st.session_state.malades_path = malades_path
                st.session_state.delimiter = delimiter  
                st.session_state.counting = counting
            st.session_state.malades = True
    if st.session_state.malades:
        if st.button("RE-UPLOAD FILE"):
            st.session_state.file_uploaded = False
            clear_directory(UPLOAD_FOLDER)
            st.session_state.malades = False
            st.rerun()
        st.session_state.file_uploaded = False
 
        def display_correction_table(malades_path, saines_path, header,new_line,encodage,delimiter,counting):
            """
            Displays an interactive table of malformed rows for correction, with validation and restore buttons.
            Handles data through file paths.
            """
            st.info(f"There is a total of {counting} after treat.")
 
               
            def lire_par_blocs(fichier_path, taille_bloc):
                """
                Lit un fichier par blocs de lignes et sépare les champs par un séparateur.
 
                Args:
                    fichier_path (str): Chemin du fichier à lire.
                    taille_bloc (int): Nombre de lignes à lire par bloc.
                    separateur (str): Caractère séparant les champs dans une ligne.
 
                Yields:
                    list: Liste de listes, où chaque sous-liste contient les champs d'une ligne.
                """
                with open(fichier_path, "r",encoding=encodage) as fichier:
                    while True:
                        # Lire un bloc de `taille_bloc` lignes
                        bloc = [fichier.readline().strip() for _ in range(taille_bloc)]
                        # Supprimer les lignes vides
                        bloc = [ligne for ligne in bloc if ligne]
                        if not bloc:
                            break
                        # Séparer les champs pour chaque ligne
                        bloc_separe = [ligne.split(",") for ligne in bloc]
                        yield bloc_separe
 
            if "bloc" not in st.session_state:
                st.session_state.bloc = False
               
                    # Initialize session state for managing blocks
            if 'current_bloc' not in st.session_state:
                st.session_state.current_bloc = 0
                # Read the current block
            blocs = list(lire_par_blocs(malades_path, taille_bloc=10))
            if st.session_state.current_bloc >= len(blocs):
                st.success("All rows have been processed.")
                return
            else:
                current_bloc = blocs[st.session_state.current_bloc]

                # Bouton pour passer au prochain bloc
                if st.button("Next Block"):
                    st.session_state.current_bloc += 1
                    st.session_state.bloc = True
                    st.rerun()
            
            max_columns = max(len(row) for row in current_bloc)
            # Calculer le nombre maximum de colonnes
           
            extended_header = header + [f"Unnamed_Column_{i+1}" for i in range(len(header), max_columns)]
            adjusted_rows = []
            # Compléter les lignes pour qu'elles aient le même nombre de colonnes
            for row in current_bloc:
                adjusted_rows.append(row + [""] * (max_columns - len(row)))
           
            # Créer le DataFrame pour l'affichage
            df_display = pd.DataFrame(adjusted_rows, columns=extended_header)
            
 
            # Décaler l'index pour commencer à 1
            df_display.index = df_display.index + 1
            
               
 
            # Initialize original values and modified state if not already done
            if 'cell_initial' not in st.session_state or st.session_state.bloc:
                    st.session_state.cell_initial = {(i, j): df_display.iat[i, j] for i in range(df_display.shape[0]) for j in range(df_display.shape[1])}
                    st.session_state.modified_cells = [(1,1)]  # Initialisation par défaut si le tableau n'est pas vide
            if 'corrected' not in st.session_state or st.session_state.bloc:
               
                st.session_state.corrected = []
            if 'cell_current' not in st.session_state or st.session_state.bloc:  
                st.session_state.cell_current = df_display
 
            if 'validated_rows' not in st.session_state or st.session_state.bloc:
                st.session_state.validated_rows = []
            if 'selection_rows' not in st.session_state or st.session_state.bloc:
                st.session_state.selection_rows = list(range(1,df_display.shape[0]+1))
                st.session_state.bloc = False
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
                    "Select a column",
                    options=list(range(df_display.shape[1])),
                    format_func=lambda x: df_display.columns[x]
                )
                st.sidebar.write('<p class="sidebar-text">After selecting the cells, click the Merge button to combine them with the adjacent cell on the right.</p>', unsafe_allow_html=True)
                st.sidebar.write('<p class="sidebar-text">After selecting the cells, click the Delete button to remove the selected cells.</p>', unsafe_allow_html=True)
                st.sidebar.write('<p class="sidebar-text">Both Merge and Delete actions will shift the remaining table content to the left.</p>', unsafe_allow_html=True)
           

               
                 
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
            st.sidebar.write('<p class="sidebar-text">Click the Validate button to save the edited rows to storage.</p>', unsafe_allow_html=True)
 
            # Validation des lignes modifiées
            if st.sidebar.button("Validate"):
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
                        with open(saines_path, "a", newline=new_line, encoding=encodage) as file:
                            writer = csv.writer(file,delimiter=delimiter)
                            for row in valid_rows:
                                writer.writerow(st.session_state.cell_current.iloc[row, :len(header)].tolist())
                        # Supprimer les lignes validées en utilisant les indices actuels
                        st.session_state.cell_current.drop(rows_to_drop, inplace=True)
 
                        # Réinitialiser l'index pour commencer à 1
                        st.session_state.cell_current.reset_index(drop=True, inplace=True)
                        st.session_state.cell_current.index = st.session_state.cell_current.index + 1
                        st.session_state.selection_rows =list(range(1,st.session_state.cell_current.shape[0]+1)) # Mettre à jour les lignes sélectionnées
                        st.rerun()  # Redémarrer pour forcer la mise à jour
 
 
        clear_directory(UPLOAD_FOLDER)
        display_correction_table(st.session_state.malades_path,st.session_state.saines_path, st.session_state.first_line,st.session_state.new_line,st.session_state.encodage,st.session_state.delimiter,st.session_state.counting)
        if st.button("Download "):
            with open(st.session_state.saines_path, "r", encoding=st.session_state.encodage) as f:
                
                data_f = f.read()
            st.download_button(
                    label="Download your file as TXT",
                    data = data_f,
                    file_name=f"{st.session_state.saines_path}",
                    mime="text/plain"
                )


 