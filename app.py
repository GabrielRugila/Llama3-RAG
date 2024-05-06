import streamlit as st
from chains import Model
from utils import Document_Tools

st.set_page_config(page_title="LLAMA3-RAG Chat", 
                   page_icon="🦙", 
                   layout="wide")

# --- Sidebar --- #
st.sidebar.title("LLAMA3-RAG Chat")
model_temperature = st.sidebar.slider("Temperature", 
                                      0.0, 1.0, 0.2, 0.1, 
                                      key="temperature", 
                                      help="Temperature controls how 'random' the model responds. \n\nLower values tend to be more deterministic, while higher values are more creative. \n\n Warning: High values may lead to hallucinations.")

## --- Document Control --- ##
uploaded_file = st.sidebar.file_uploader("Upload a document", 
                                         type=["pdf", "csv", "txt"],
                                         accept_multiple_files=False,
                                         help="File Naming Convention: \n- Length must be between 3 to 63 characters \n- Must start and end with a letter or number \n- Can contain dots, dashes, and underscores")

if uploaded_file is not None:
    doc_ = Document_Tools()
    doc_.add_document(uploaded_file)


st.sidebar.markdown("Manage documents in teh database:")
docs = Document_Tools()
files_list = docs.get_documents_list()
selected_files = st.sidebar.multiselect("Select documents", files_list)

if "delete_clicked" not in st.session_state:
    st.session_state.delete_clicked = False

if selected_files:
    if st.sidebar.button("Delete Documents"):
        st.session_state.delete_clicked = True
    if st.session_state.delete_clicked:
        col1, col2 = st.sidebar.columns(2, gap="small")
        confirm_placeholder = col1.empty()
        cancel_placeholder = col2.empty()
        if confirm_placeholder.button("Confirm"):
            for file in selected_files:
                docs.remove_document(file)
            st.session_state.delete_clicked = False
            st.toast("Documents deleted successfully!", icon="✅")
            confirm_placeholder.empty()
            cancel_placeholder.empty()
            selected_files = [file for file in selected_files if file not in files_list]
        elif cancel_placeholder.button("Cancel"):
            st.session_state.delete_clicked = False
            st.toast("Deletion cancelled", icon="🚨")
            confirm_placeholder.empty()
            cancel_placeholder.empty()

# --- Main Chat Interface --- #
model = Model(temperature=model_temperature)
def send_prompt(prompt):
    response = model.query_with_memory(prompt)
    return response

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "bot", "message": "Ask anything to get started!"},
    ]

if prompt := st.chat_input("User:"):
    st.session_state.messages.append({"role": "user", "message": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["message"])

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("bot"):
        with st.spinner("Processing..."):
            response = send_prompt(prompt)
            st.write(f"AI: \n\n{response}")
            message = {"role": "bot", "message": response}
            st.session_state.messages.append(message)

