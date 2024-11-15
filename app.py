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
        extended_types = types + [""] * (max_columns - len(header))  # Types pour les colonnes supplémentaires

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

        if df_display.empty:
            st.warning("The table is empty and cannot be displayed.")
            return None, corrected

        # Initialize original values and modified state if not already done
        if 'cell_initial' not in st.session_state:
            if not df_display.empty:
                st.session_state.cell_initial = {(i, j): df_display.iat[i, j] for i in range(df_display.shape[0]) for j in range(df_display.shape[1])}
                st.session_state.cell_current = df_display.copy()
                st.session_state.modified_cells = [(0, 0)]  # Initialisation par défaut si le tableau n'est pas vide
            else:
                return None, None  # Retourne None si le tableau est vide
            
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
                "Select a single column",
                options=list(range(df_display.shape[1])),
                format_func=lambda x: df_display.columns[x]
            )
            st.sidebar.write('<p class="sidebar-text">After cells selection, click on Merge button to merge the selected cells with the next cell on the right.</p>', unsafe_allow_html=True)
            st.sidebar.write('<p class="sidebar-text">After cells selection, click on Delete button to delete the selected cells.</p>', unsafe_allow_html=True)
            st.sidebar.write('<p class="sidebar-text">Both Merge and Delete operations result in shifting the rest of the table to the left.</p>', unsafe_allow_html=True)
            st.sidebar.write('<p class="sidebar-text">Press the validation button to send edited rows to storage.</p>', unsafe_allow_html=True)
            # Convert input text to index lists
            selected_rows = [int(x.strip()) for x in row_selection.split(",") if x.strip().isdigit()]

            # Verify selected rows exist
            max_index = corrected_df.shape[0] 
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
                
                valid_rows = []  # Liste pour stocker les lignes à supprimer après validation

                for row_index in selected_rows:
                    total_fields_count = len(corrected_df.iloc[row_index, :len(header)])
                    extra_fields_empty = all(
                        str(field).strip() == "" or pd.isna(field)
                        for field in corrected_df.iloc[row_index, len(header):]
                    )

                    if total_fields_count == len(header) and extra_fields_empty:
                        valid_row = corrected_df.iloc[row_index, :len(header)].tolist()
                        st.session_state.corrected.append(valid_row)
                        valid_rows.append(row_index)
                        st.sidebar.success(f"Row {row_index+1} validated successfully!")
                    else:
                        st.sidebar.write(f"Row {row_index} contains filled extra columns and cannot be validated.")
                
                # Supprimer toutes les lignes validées en une seule fois
                if valid_rows:
                    st.sidebar.write(f"Validated rows: {valid_rows}")
                    st.session_state.cell_current.drop(valid_rows, inplace=True)
                    st.session_state.cell_current.reset_index(drop=True, inplace=True)
                    st.rerun()  # Restart to force update

        return st.session_state.cell_current,None
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
    new_index = 1
    store_list = []
    old_index_buffer = [] 
    saines_pi = [] # Liste tampon pour capturer tous les indices d'origine
    nb_malades = 0  
    nb_saines = 0
    nb_saines_pi = 0
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
            new_index += 1
        elif len(line) == number:
            # Ligne correcte -> Ajouter à "saines"
            saines.append(line)
            nb_saines += 1
            # Enregistrer le mapping old_index -> new_index
            old_index_buffer.clear()

            index += 1  # Passer à la ligne suivante
            new_index += 1  # Incrémenter le nouvel index

        elif len(line) < number:
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
                new_index += 1  # Incrémenter le nouvel index
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

    types = ["String", "String", "Date", "String", "String", "String", "Date", "String", "String", "String", "Date", "String", "String"]

    corrected_df,corrected=display_correction_table(malades, first_line,types)
    if corrected :
        # Convertir les lignes "saines" en DataFrame
        df_Pi = pd.DataFrame(saines_pi, columns=first_line)
        df_sigma = pd.DataFrame(corrected, columns=first_line)
        df_saines = pd.DataFrame(saines, columns=first_line)
        df_final = pd.concat([df_saines,df_Pi,df_sigma],axis=0)
            # Récupérer le nom de fichier original sans extension
        base_file_name = remove_file_extension(uploaded_file.name)
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
