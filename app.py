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

def display_correction_table(malformed_rows, header, saines):
    """
    Displays an interactive table of malformed rows for correction, with validation and restore buttons.
    Adds options to select multiple rows and columns to perform actions on selected cells.
    """
    
    if malformed_rows:
        
        st.write("## Detected malformed rows")
        
        max_columns = max(len(row) for row in malformed_rows)
        extended_header = header + [f"Unnamed_Column_{i+1}" for i in range(len(header), max_columns)]
        
        # Adjust rows to match the maximum number of columns and replace None with ""
        adjusted_rows = [
            [field if field is not None else "" for field in (row + [None] * (max_columns - len(row)))]
            for row in malformed_rows
        ]
        
        # Create DataFrame for display
        df_display = pd.DataFrame(adjusted_rows, columns=extended_header)
        if df_display.empty:
            st.warning("The table is empty and cannot be displayed.")
            return None, None
        # Initialize original values and modified state if not already done
        if 'cell_initial' not in st.session_state:
# Vérifie si le tableau n'est pas vide avant d'initialiser les valeurs
            
            if not df_display.empty:
                st.session_state.cell_initial = {(i, j): df_display.iat[i, j] for i in range(df_display.shape[0]) for j in range(df_display.shape[1])}
                st.session_state.cell_current = df_display.copy()

                st.session_state.modified_cells = [(0, 0)]  # Initialisation par défaut si le tableau n'est pas vide
            else:
                return None, None# Liste vide si le tableau est vide
            
            st.session_state.corrected = []
        if 'validated_rows' not in st.session_state:
            st.session_state.validated_rows = []

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
            row_selection = st.sidebar.text_input("Enter multiple/single row numbers.")
            col_selection = st.sidebar.multiselect(
                "Select columns", 
                options=list(range(df_display.shape[1])), 
                format_func=lambda x: extended_header[x]
            )

            # Convert input text to index lists
            selected_rows = [int(x.strip()) for x in row_selection.split(",") if x.strip().isdigit()]

            # Verify selected rows exist
            max_index = corrected_df.shape[0] - 1
            if any(row_index > max_index for row_index in selected_rows):
                st.sidebar.error("No more data to edit.")
        else:
            st.sidebar.write("The table is empty. No operations can be performed.")
            selected_rows = []
            col_selection = []
            return st.session_state.cell_current, st.session_state.corrected
 

        # Merge selected cells with the next cell
        if st.sidebar.button("Merge"):
            for i in selected_rows:
                for j in col_selection:
                    if j < corrected_df.shape[1] - 1:  # Check if merge is possible
                        st.session_state.cell_current.iat[i, j] = str(corrected_df.iat[i, j]) + " " + str(corrected_df.iat[i, j + 1])
                        st.session_state.cell_current.iloc[i, j+1:] = corrected_df.iloc[i, j+2:].tolist() + [""]
            st.sidebar.write("Selected cells were merged.")
            st.rerun()  # Restart to force update

        # Delete selected cells and shift remaining cells accordingly
        if st.sidebar.button("Delete"):
            for i in selected_rows:
                for j in col_selection:
                    st.session_state.cell_current.iloc[i, j:] = corrected_df.iloc[i, j+1:].tolist() + [""]
            st.sidebar.write("Selected cells were deleted.")
            st.rerun()  # Restart to force update
        # Validation des lignes modifiées
        if st.sidebar.button("Validation"):
            if st.session_state.modified_cells or len(st.session_state.cell_current) == 1:
                rows_to_validate = sorted(set(i for i, _ in st.session_state.modified_cells))
                valid_rows = []  # Liste pour stocker les lignes à supprimer après validation

                for row_index in rows_to_validate:
                    # Comparer la longueur totale de la ligne avec celle de l'en-tête
                    total_fields_count = len(corrected_df.iloc[row_index, :len(header)])
                            # Vérifier que les champs au-delà de l'en-tête sont vides
                    extra_fields_empty = all(
                        field == "" or pd.isna(field)
                        for field in corrected_df.iloc[row_index, len(header):]
                    )
                    if total_fields_count == len(header)  and extra_fields_empty:
                        valid_row = corrected_df.iloc[row_index, :len(header)].tolist()
                        st.session_state.corrected.append(valid_row)
                        
                        # Ajouter l'index à la liste pour suppression ultérieure
                        valid_rows.append(row_index)
                    else:
                        st.sidebar.write(f"Row {row_index+1} contains filled extra columns and cannot be validated.")
                
                # Supprimer toutes les lignes validées en une seule fois
                if valid_rows:
                    st.session_state.cell_current.drop(valid_rows, inplace=True)
                    st.session_state.cell_current.reset_index(drop=True, inplace=True)
                
                # Réinitialiser modified_cells après validation                
                st.rerun()
            
        return st.session_state.cell_current, st.session_state.corrected
    else:
        st.success("No malformed rows detected!")
        return None, None



# Uploader de fichier Excel
uploaded_file= st.file_uploader("Please upload a csv or Text file only", type=["csv", "txt"])
if uploaded_file :
    content = uploaded_file.read()

    # Détecter l'encodage
    try:
        decoded_content, encodage = open_file_with_auto_and_manual_encodings(content)
        st.write(f"Encodage détecté et utilisé : {encodage}")
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
    extension = verifier_extension_fichier(uploaded_file.name)

    # Si le fichier est un CSV, le convertir en TXT
    if extension == '.csv':
        # Utiliser `modified_content` pour la conversion
        texte = convert_csv_to_txt(modified_content, sep_inexistant or delimiter, encodage, new_line)
        delimiter = sep_inexistant or delimiter
    else:
        # Sinon, utiliser le contenu tel quel
        texte = modified_content



    # Découper le contenu en lignes
    lines = modified_content.splitlines()

    # Initialiser les listes pour les lignes "saines" et "malades"
    saines = []
    malades = []
    store_list = []  # Stocke les lignes incomplètes pour compléter les lignes suivantes

    # Lire la première ligne pour obtenir le nombre attendu de colonnes
    first_line = lines[0].split(delimiter)
    saines.append(first_line)  # Ajouter la première ligne aux "saines"
    number = len(first_line)  # Nombre attendu de colonnes
    most_common_count = number  # Utilisez la première ligne comme référence pour la structure

    # Parcourir le reste des lignes
    for raw_line in lines[1:]:
        # Diviser la ligne en colonnes en utilisant le délimiteur
        line = raw_line.split(delimiter)
        
        # Si la ligne dépasse le nombre de colonnes attendu, on la stocke dans malades
        if len(line) > most_common_count:
            malades.append(line)
        
        # Si la ligne a le bon nombre de colonnes, on l'ajoute directement aux "saines"
        elif len(line) == number:
            saines.append(line)
        
        # Si la ligne a moins de colonnes que le nombre attendu
        elif len(line) < most_common_count:
            if store_list:
                # Si une ligne est stockée, on la combine avec la ligne actuelle
                next_line = line
                line = store_list
            else:
                # Si aucune ligne n'est en mémoire, on essaie d'en obtenir une nouvelle
                try:
                    next_line = lines[lines.index(raw_line) + 1].split(delimiter)
                except IndexError:
                    # Si on atteint la fin du fichier, on sort de la boucle
                    break
            
            if next_line:
                # Combinaison de la fin de la ligne actuelle et du début de la ligne suivante
                milieu = line[-1] + next_line[0]
                result_line = line[:-1] + [milieu] + next_line[1:]

                # Enregistrement ou ajustement de la ligne combinée en fonction de la taille attendue
                if len(result_line) == most_common_count:
                    saines.append(result_line)  # Ligne complète, enregistrée dans saines
                    store_list = []  # Réinitialiser le stockage
                elif len(result_line) < most_common_count:
                    store_list = result_line  # Stocker pour la prochaine itération
                elif len(result_line) > most_common_count:
                    malades.append(result_line)  # Ligne trop longue, enregistrée dans malades
                next_line = None
                # Display `saines` to confirm rows were added
    corrected_df,corrected=display_correction_table(malades, first_line,saines)
    if corrected :
        # Convertir les lignes "saines" en DataFrame
        df_Pi = pd.DataFrame(corrected, columns=first_line)
        df_saines = pd.DataFrame(saines, columns=first_line)
        df_final = pd.concat([df_saines,df_Pi],axis=0)
            # Récupérer le nom de fichier original sans extension
        base_file_name = remove_file_extension(uploaded_file.name)

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
