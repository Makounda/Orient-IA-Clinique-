"""Interface Streamlit Orient'IA."""

from __future__ import annotations

import json

import streamlit as st

from orientia.agent.orchestrator import DISCLAIMER, run_agent
from orientia.agent.tools import list_tools
from orientia.rag.index import build_index, load_index


def main() -> None:
    st.set_page_config(page_title="Orient'IA", page_icon="🎓", layout="centered")
    st.title("Orient'IA")
    st.caption("Assistant d'aide à l'orientation pédagogique — ISPM")
    st.info(DISCLAIMER)

    with st.sidebar:
        st.header("Système")
        if st.button("Reconstruire l'index RAG"):
            report = build_index()
            st.success(f"Index OK — {report['n_chunks']} chunks")
        st.subheader("Outils disponibles")
        for t in list_tools():
            st.markdown(f"- `{t['name']}` — {t['description']}")
        st.caption("Sources: corpus ISPM + modèle ML (Random Forest).")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # garantir index au démarrage
    try:
        load_index()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Index RAG indisponible: {exc}")
        if st.button("Construire maintenant"):
            build_index()
            st.rerun()
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Traces / outils"):
                    st.json(msg["meta"])

    prompt = st.chat_input("Posez une question d'orientation…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = run_agent(
            prompt,
            profile=st.session_state.profile,
            session_id=st.session_state.session_id,
        )
        st.session_state.session_id = result["session_id"]
        st.session_state.profile = result["profile"]

        meta = {
            "intent": result["intent"],
            "tools_called": result["tools_called"],
            "citations": result["citations"],
            "elapsed_ms": result["elapsed_ms"],
            "trace_id": result["trace_id"],
            "refusal": result["refusal"],
        }
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "meta": meta}
        )
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            with st.expander("Traces / outils"):
                st.json(meta)

    st.divider()
    st.subheader("Profil courant (extrait)")
    st.code(json.dumps(
        {k: v for k, v in st.session_state.profile.items() if k.startswith(("score_", "comp_", "interet_")) and v},
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
