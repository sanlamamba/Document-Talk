import os
import datetime
import pickle
import tempfile
import pandas as pd
import streamlit as st
import zipfile
from io import StringIO

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

st.set_page_config(page_title="📚 Assistant de Conversation sur Documents", page_icon="📚", layout="wide")

# ================================
# Chemins et répertoires
# ================================
DB_PATH = "embeddings/faiss_index"
METADATA_PATH = "embeddings/metadata.pkl"

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

# ================================
# Fonctions utilitaires
# ================================
def save_metadata(metadata):
    """Enregistre les métadonnées dans un fichier pickle."""
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_metadata():
    """Charge les métadonnées à partir d'un fichier pickle."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return {}

@st.cache_resource
def initialize_db():
    """Initialise la base FAISS. Si elle n'existe pas, la crée à partir d'un texte vide."""
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "votre_clé_api_openai")
    )
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_texts([" "], embeddings)
    return db

def load_file(file_path, file_name):
    """Charge un fichier en objets Document. Gère TXT, CSV, Excel, PDF ou utilise un chargeur non structuré."""
    file_type = file_name.split(".")[-1].lower()
    documents = []

    if file_type == "txt":
        documents.extend(TextLoader(file_path).load())
    elif file_type == "csv":
        df = pd.read_csv(file_path)
        nb_rows = len(df)
        for _, row in df.iterrows():
            text = " ".join(map(str, row.values))
            documents.append(Document(page_content=text, metadata={"source": file_name, "type": "csv", "rows": nb_rows}))
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
        nb_rows = len(df)
        for _, row in df.iterrows():
            text = " ".join(map(str, row.values))
            documents.append(Document(page_content=text, metadata={"source": file_name, "type": "excel", "rows": nb_rows}))
    elif file_type == "pdf":
        reader = PdfReader(file_path)
        all_docs = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            doc = Document(page_content=page_text, metadata={"source": file_name, "type": "pdf", "page": i+1})
            all_docs.append(doc)
        documents.extend(all_docs)
    else:
        docs = UnstructuredFileLoader(file_path).load()
        for d in docs:
            d.metadata["source"] = file_name
            d.metadata["type"] = "other"
        documents.extend(docs)

    return documents

def process_and_add_documents(files, db):
    """Traite les fichiers téléchargés (y compris les archives ZIP) et les ajoute à l'index FAISS."""
    all_documents = []
    metadata = load_metadata()

    for file in files:
        file_type = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_type) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        if file_type == "zip":
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                extract_dir = tempfile.mkdtemp()
                zip_ref.extractall(extract_dir)

                for root, dirs, extracted_files in os.walk(extract_dir):
                    for extracted_file in extracted_files:
                        extracted_path = os.path.join(root, extracted_file)
                        docs = load_file(extracted_path, extracted_file)
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(docs)
                        all_documents.extend(chunks)
                        metadata[extracted_file] = {"added_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        else:
            docs = load_file(tmp_file_path, file.name)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            all_documents.extend(chunks)
            metadata[file.name] = {"added_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    db.add_documents(all_documents)
    db.save_local(DB_PATH)
    save_metadata(metadata)

def clear_all_documents():
    """Efface tous les documents et l'index, permettant de repartir à zéro."""
    if os.path.exists(DB_PATH):
        for root, dirs, files in os.walk(DB_PATH, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(DB_PATH)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
    st.success("Tous les documents ont été effacés ! Veuillez rafraîchir la page.")


def summarize_all_documents(qa_chain):
    """Crée un résumé global de tous les documents importés."""
    summary_query = "Donne-moi un résumé global de tous les documents importés."
    response = qa_chain({"query": summary_query})
    return response["result"]

def download_chat_history():
    """Permet à l'utilisateur de télécharger l'historique de conversation."""
    if "chat_history" not in st.session_state or not st.session_state["chat_history"]:
        st.warning("Aucun historique à télécharger.")
        return

    output = StringIO()
    for q, a, sources, timestamp in st.session_state["chat_history"]:
        output.write(f"Vous ({timestamp}): {q}\n")
        output.write(f"Assistant: {a}\n")
        output.write("-" * 50 + "\n")
    output_str = output.getvalue()
    st.download_button("Télécharger l'historique", data=output_str.encode('utf-8'), file_name="historique_conversation.txt")

# ================================
# Initialisation
# ================================
db = initialize_db()

st.sidebar.title("📂 Gérer vos documents")

model_name = st.sidebar.selectbox("Choisissez le modèle OpenAI :", ["gpt-4", "gpt-3.5-turbo"])

uploaded_files = st.sidebar.file_uploader(
    "Choisissez des documents à importer (plusieurs fichiers, ou un ZIP pour un import massif).",
    accept_multiple_files=True
)
if uploaded_files:
    with st.spinner("Traitement des documents en cours..."):
        process_and_add_documents(uploaded_files, db)
    st.sidebar.success("Les documents ont été ajoutés avec succès !")

if st.sidebar.button("Résumer tous les documents"):
    metadata = load_metadata()
    if len(metadata) == 0:
        st.sidebar.warning("Aucun document disponible pour le résumé.")
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        db_reload = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db_reload.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        summary = summarize_all_documents(qa_chain)
        st.sidebar.write("### Résumé des documents :")
        st.sidebar.write(summary)

if st.sidebar.button("Effacer tous les documents"):
    clear_all_documents()

if st.sidebar.button("Effacer l'historique de conversation"):
    st.session_state["chat_history"] = []
    st.sidebar.success("Historique de conversation effacé !")

download_chat_history()

# ================================
# Initialisation du LLM et QA Chain avec le modèle choisi
# ================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "votre_clé_api_openai":
    st.error("Clé API OpenAI introuvable. Veuillez définir la variable d'environnement OPENAI_API_KEY.")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

if os.path.exists(DB_PATH):
    db_reload = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db_reload.as_retriever(search_type="similarity", search_kwargs={"k": 5})
else:
    retriever = None

llm = ChatOpenAI(
    model_name=model_name,
    temperature=0,
    openai_api_key=api_key
)

qa_chain = None
if retriever is not None:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# ================================
# Interface principale
# ================================
st.title("📚 Assistant de Conversation sur Documents")

st.write(
    "Bienvenue dans l'Assistant de Conversation sur Documents ! "
    "Voici ce que vous pouvez faire :\n"
    "- Importer un ou plusieurs documents (ou une archive ZIP) via la barre latérale.\n"
    "- Poser des questions sur le contenu des documents importés.\n"
    "- Résumer tous les documents d'un seul coup.\n"
    "- Effacer l'historique de conversation ou tous les documents si vous voulez repartir de zéro.\n"
    "- Choisir le modèle OpenAI que vous souhaitez utiliser (gpt-4, gpt-3.5-turbo).\n"
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.write("### Interface de Discussion")
st.markdown(
    "Saisissez votre question ci-dessous. L'assistant tentera de répondre en utilisant les documents importés. "
    "Déroulez la section 'Sources' sous chaque réponse pour voir les documents utilisés."
)

query = st.text_input("Posez une question :", placeholder="Par exemple : Quelles sont les idées principales des documents importés ?")

if query:
    metadata = load_metadata()
    if len(metadata) == 0:
        st.warning("Aucun document disponible. Veuillez importer des documents avant de poser une question.")
    else:
        if qa_chain is None:
            st.error("Le système n'est pas encore prêt. Réessayez dans un instant.")
        else:
            with st.spinner("Réflexion en cours..."):
                response = qa_chain({"query": query})
                answer = response["result"]
                sources = response["source_documents"]
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state["chat_history"].append((query, answer, sources, timestamp))

for i, (q, a, sources, timestamp) in enumerate(st.session_state["chat_history"]):
    st.markdown(f"**Vous ({timestamp}) :** {q}")
    st.markdown(f"**Assistant :** {a}")
    with st.expander("Sources"):
        for source in sources:
            source_name = source.metadata.get('source', 'inconnu')
            doc_type = source.metadata.get('type', 'inconnu')
            excerpt = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content

            additional_info = []
            if doc_type == "pdf":
                page_num = source.metadata.get('page', '?')
                additional_info.append(f"Page: {page_num}")
            elif doc_type in ["csv", "excel"]:
                rows_count = source.metadata.get('rows', '?')
                additional_info.append(f"Lignes dans le fichier: {rows_count}")

            info_str = ", ".join(additional_info)
            if info_str:
                st.write(f"- **{source_name}** ({info_str}) : {excerpt}")
            else:
                st.write(f"- **{source_name}** : {excerpt}")
