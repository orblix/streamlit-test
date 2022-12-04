from pyserini.search.faiss import FaissSearcher
from streamlit_chat import message
import json
import openai
import streamlit as st

FIRST_MESSAGE = "Enter your openai public key here."
SECOND_MESSAGE = "Enter your question here."
IS_FIRST = True

st.set_page_config(page_title="Company Asker - Imperva Demo", page_icon=":robot:")

st.header("Company Asker - Imperva Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Load index
searcher = FaissSearcher("./index/", "castorini/tct_colbert-v2-hnp-msmarco")    


def get_first_text():
    input_text = st.text_input("You: ", FIRST_MESSAGE, key="input")
    return input_text

def get_text():
    input_text = st.text_input("You: ", SECOND_MESSAGE, key="input")
    return input_text


def get_raw_text(id):
    with open("./imperva_raw_documents.jsonl", "r") as f:
        # Read line by line:
        for line in f:
            # Load json:
            data = json.loads(line)
            if data["id"] == id:
                return data["contents"]
        return "CONTEXT_NOT_FOUND"


def get_contexts(question: str):
    hits = searcher.search(question)
    contexts = []
    for i in range(0, 10):
        id = hits[i].docid
        url = hits[i].docid.split("$$")[0].replace("__", "/")
        raw_text = get_raw_text(id)
        contexts.append(f"CONTEXT {i+1}:\n" + raw_text)

def get_answer(question: str, contexts: list, optional_args = {}):
    """Generate code by completing given prompt.

    Args:
        prompt: initiate generation with this prompt

    Returns:
        statistics, generated code
    """
    context = "\n\n".join(contexts)
    prompt = f"{context}\n\nConsider the above contexts, answer the following question:\n{question}\nANSWER:"
    try:
        kwargs = optional_args
        response = openai.Completion.create(
            model="text-davinci-003", **kwargs, prompt=prompt
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"Error querying OpenAI: {e}"        

def answer_question(question: str):
    contexts = get_contexts(question)
    answer = get_answer(question, contexts, {"max_tokens": 1000, "tempature": 0})
    return answer

def set_key(key):
    openai.api_key = key

if IS_FIRST:
    user_input = get_first_text()
else:
    user_input = get_text()

if user_input:
    # Check using regex if user input equals to key:API_KEY:
    if user_input == r"key:.*":
        set_key(user_input)
        output = "Key set."
    elif user_input == SECOND_MESSAGE:
        if not IS_FIRST:
            output = answer_question(user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
