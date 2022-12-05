from pyserini.search.faiss import FaissSearcher
from streamlit_chat import message
from transformers import GPT2TokenizerFast
import json
import openai
import streamlit as st

FIRST_MESSAGE = "Enter your openai public key here. Use 'key:YOUR_KEY'"
SECOND_MESSAGE = "Enter your question here."

if "is_key_set" not in st.session_state:
    st.session_state["is_key_set"] = False

st.set_page_config(page_title="Company Asker - Imperva Demo", page_icon=":robot:")

st.header("Company Asker - Imperva Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = [FIRST_MESSAGE]

if "past" not in st.session_state:
    st.session_state["past"] = [""]

# Load index
searcher = FaissSearcher("./index/", "castorini/tct_colbert-v2-hnp-msmarco")    

input_container = st.empty()
def get_text():
    input_text = input_container.text_input("You: ", key="input")
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

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def estimate_prompt_tokens(prompt: str):
    """Get chunk size making sure we can also fit the prompt in."""
    prompt_tokens = tokenizer(prompt)
    return len(prompt_tokens["input_ids"])

def get_contexts(question: str):
    hits = searcher.search(question)
    contexts = []
    for i in range(0, 10):
        id = hits[i].docid
        url = hits[i].docid.split("$$")[0].replace("__", "/")
        raw_text = get_raw_text(id)
        contexts.append(f"CONTEXT {i+1}:\n" + raw_text)
    return contexts

def get_answer(question: str, contexts: list, optional_args = {}):
    """Generate code by completing given prompt.

    Args:
        prompt: initiate generation with this prompt

    Returns:
        statistics, generated code
    """
    context = "\n\n".join(contexts)
    prompt = f"Contexts about the Imperva company:\n\n{context}\n\nConsider the above contexts, answer the following question:\n{question}\nANSWER:"
    prompt_tokens = estimate_prompt_tokens(prompt)
    print("Prompt:")
    print(prompt)
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Sending request with max_tokens={4096 - prompt_tokens - 1}")
    try:
        kwargs = {**{"max_tokens": 4096 - prompt_tokens - 1}, **optional_args}
        response = openai.Completion.create(
            model="text-davinci-003", **kwargs, prompt=prompt
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"Error querying OpenAI: {e}"        

def answer_question(question: str):
    contexts = get_contexts(question)
    answer = get_answer(question, contexts, {"temperature": 0})
    return answer

def set_key(key):
    openai.api_key = key
    st.session_state["is_key_set"] = True
    print("Key set.")

user_input = get_text()

if user_input:
    output = "PLACEHOLDER"
    produced_answer = False
    if user_input.startswith("key:"):
        set_key(user_input[4:])
        output = "Key set. What's your question?"
        produced_answer = True
    elif st.session_state["is_key_set"]:
        output = answer_question(user_input)
        produced_answer = True

    if produced_answer:
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

if st.session_state["generated"]:
    if not st.session_state["is_key_set"]:
        message(st.session_state["generated"][0], key="0")
    else:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            if i < len(st.session_state["past"]) and st.session_state["past"][i] != "":
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")