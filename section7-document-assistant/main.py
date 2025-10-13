import streamlit as st
from backend.core import run_llm

# Streamlitアプリは、ユーザの操作があるたびに最初から最後まで再実行される(Eventが発生するたびに、かな)
# 変数の状態の毎回初期化されるので、st.session_stateを使って状態を保存する
st.header("Document Assistant")

prompt = st.text_input(
    "Enter your question about LangChain:", placeholder="What is LangChain?"
)

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return "No sources found."
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "\n".join([f"- {url}" for url in sources_list])
    return sources_string


if prompt:
    with st.spinner("Generating answer..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f""
            f"**Answer:**\n{generated_response['result']}\n\n"
            f"**Sources:**\n{create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

        # import pdb

        # pdb.set_trace()

if st.session_state["user_prompt_history"]:
    for user_prompt, generated_answer in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        st.chat_message("user").markdown(user_prompt)
        st.chat_message("assistant").markdown(generated_answer)
