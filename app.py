"""Streamlit chat UI for the SQL analyst.

Run with:
    streamlit run app.py

The UI mirrors the conversational style of GitHub Copilot / Claude Code:
  - chat history in the main area
  - approval cards rendered inline (plan / SQL) with Approve / Edit / Cancel buttons
  - sidebar with toggles, controls, and the trainer
"""
from __future__ import annotations
import uuid
import os
import subprocess
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from config import Config
from graph import build_graph
from tools.db import get_engine, test_connection
from tools.ollama_client import OllamaClient
from tools.embeddings import EmbeddingsCache
from tools.examples_store import ExamplesStore
from tools.schema import load_or_build, all_table_names
from tools.indexer import build_index


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="SQL Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Bootstrap
# ============================================================

@st.cache_resource
def _bootstrap():
    """One-time setup: load config, build graph, init engine and caches."""
    cfg = Config.load("config.yaml")
    client = OllamaClient(
        host=cfg.ollama_host,
        default_model=cfg.model,
        default_num_ctx=cfg.num_ctx,
    )
    engine = get_engine(cfg.db_connection_string, query_timeout_sec=cfg.query_timeout_sec)
    embeddings = EmbeddingsCache(cfg.embeddings_cache_path, cfg.embedding_model, client)
    examples_store = ExamplesStore(cfg.examples_db_path, embeddings, user_id=cfg.user_id)
    schema_chunks = load_or_build(cfg.ddl_file, cfg.notes_file, cfg.schema_cache_path, embeddings)
    graph = build_graph(cfg, client, engine, schema_chunks, embeddings, examples_store)
    return cfg, client, engine, embeddings, examples_store, schema_chunks, graph


def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None
    if "want_chart" not in st.session_state:
        st.session_state.want_chart = False
    if "want_explanation" not in st.session_state:
        st.session_state.want_explanation = True
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []


cfg, client, engine, embeddings, examples_store, schema_chunks, graph = _bootstrap()
_init_session()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### Setup")
    st.caption(f"Model: `{cfg.model}`")
    st.caption(f"Embeddings: `{cfg.embedding_model}`")
    st.caption(f"{len(schema_chunks)} tables in schema")

    st.divider()
    st.markdown("### This run")
    st.session_state.want_chart = st.toggle(
        "Generate chart",
        value=st.session_state.want_chart,
        help="Also produce a Plotly chart from the results.",
    )
    st.session_state.want_explanation = st.toggle(
        "Plain-English summary",
        value=st.session_state.want_explanation,
    )

    st.divider()
    st.markdown("### Tools")

    if st.button("Test DB connection", use_container_width=True):
        ok, msg = test_connection(engine)
        (st.success if ok else st.error)(msg)

    if st.button("Re-embed schema", use_container_width=True,
                 help="Re-parse DDL + notes and re-compute embeddings."):
        with st.spinner("Re-embedding schema..."):
            new_chunks = load_or_build(
                cfg.ddl_file, cfg.notes_file, cfg.schema_cache_path,
                embeddings, force_rebuild=True,
            )
            schema_chunks.clear()
            schema_chunks.extend(new_chunks)
        st.success(f"Re-embedded {len(schema_chunks)} tables.")

    if st.button("Reindex shared drive", use_container_width=True,
                 help="Walk the shared drive and rebuild the file index. Can be slow."):
        with st.spinner("Indexing files..."):
            try:
                n = build_index(cfg.shared_drive_root, cfg.file_index_db)
                st.success(f"Indexed {n} files.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.button("Open output folder", use_container_width=True):
        try:
            os.startfile(cfg.output_folder)
        except (AttributeError, OSError):
            try:
                subprocess.Popen(["xdg-open", cfg.output_folder])
            except Exception:
                st.error("Couldn't open the folder. Path: " + cfg.output_folder)

    st.divider()
    if st.button("New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_interrupt = None
        st.session_state.conversation_context = []
        st.rerun()

    st.divider()
    st.markdown("### Trainer")
    st.caption("Promote a past query to a verified example. Verified examples rank higher in future retrievals.")
    recent = examples_store.recent(limit=20)
    if not recent:
        st.caption("No runs yet.")
    else:
        for r in recent[:10]:
            badge = "✅" if r["verified"] else ("✓" if r["success"] else "✗")
            label = f"{badge} {r['question'][:50]}{'...' if len(r['question']) > 50 else ''}"
            with st.expander(label):
                st.code(r["sql"], language="sql")
                cols = st.columns(2)
                if not r["verified"]:
                    if cols[0].button("Mark verified", key=f"verify_{r['id']}"):
                        examples_store.mark_verified(r["id"])
                        st.rerun()
                edited_sql = cols[1].text_area(
                    "Edit + verify",
                    value=r["sql"], height=120,
                    key=f"edit_{r['id']}", label_visibility="collapsed",
                )
                if cols[1].button("Save edited + verify", key=f"save_{r['id']}"):
                    examples_store.mark_verified(r["id"], new_sql=edited_sql)
                    st.rerun()


# ============================================================
# Main area: chat
# ============================================================

st.title("SQL Analyst")
st.caption(
    "Ask in plain English. The agent plans, you approve, it writes T-SQL, you approve, "
    "it runs read-only, and pops the result into Excel."
)


# ---------- helpers ----------

def _push_message(role: str, kind: str, payload):
    """Append a message to the chat history."""
    st.session_state.messages.append({"role": role, "kind": kind, "payload": payload})


def _render_message(msg: dict):
    role = msg["role"]
    kind = msg["kind"]
    payload = msg["payload"]
    with st.chat_message(role):
        if kind == "text":
            st.markdown(payload)
        elif kind == "plan":
            st.markdown("**Plan**")
            st.markdown(payload)
        elif kind == "sql":
            st.markdown("**SQL**")
            st.code(payload, language="sql")
        elif kind == "result":
            _render_result(payload)
        elif kind == "files":
            _render_file_results(payload)
        elif kind == "error":
            st.error(payload)
        elif kind == "cancelled":
            st.warning(payload)


def _render_result(payload: dict):
    rc = payload.get("row_count", 0)
    truncated = payload.get("truncated", False)
    runtime = payload.get("runtime_ms", 0)
    excel_path = payload.get("excel_path")

    st.markdown(f"**Done** — {rc:,} rows{' (truncated)' if truncated else ''} in {runtime} ms.")
    if excel_path:
        st.caption(f"Saved: `{excel_path}`")

    df = payload.get("dataframe")
    if df is not None and not df.empty:
        st.dataframe(df.head(20), use_container_width=True)
        if len(df) > 20:
            st.caption(f"(showing first 20 of {len(df):,} rows)")

    if payload.get("explanation"):
        st.info(payload["explanation"])

    if payload.get("chart_html_path"):
        chart_path = Path(payload["chart_html_path"])
        if chart_path.exists():
            st.caption(f"Chart: `{chart_path}`")
            # Streamlit can also render plotly figures directly; for now, surface
            # the saved HTML path. Live render below if we have the figure.


def _render_file_results(files: list[dict]):
    if not files:
        st.info("No matching files found.")
        return
    st.markdown(f"**{len(files)} file(s) matched**")
    for f in files[:20]:
        st.markdown(f"- `{f['name']}` — {f['path']}")


def _approval_card(kind: str, payload: dict):
    """Render an approval card inline. Updates pending_interrupt on click."""
    with st.chat_message("assistant"):
        if kind == "plan_approval":
            st.markdown("**Proposed plan — please review:**")
            edited = st.text_area(
                "Plan", value=payload.get("plan", ""), height=220,
                key=f"plan_edit_{st.session_state.thread_id}",
                label_visibility="collapsed",
            )
            cols = st.columns([1, 1, 1, 4])
            if cols[0].button("✅ Approve", key="plan_approve", type="primary"):
                _resume_graph({"action": "approve", "content": None})
            if cols[1].button("✏️ Approve edited", key="plan_edit"):
                _resume_graph({"action": "edit", "content": edited})
            if cols[2].button("✖ Cancel", key="plan_cancel"):
                _resume_graph({"action": "cancel", "content": None})

        elif kind == "sql_approval":
            st.markdown("**Proposed SQL — please review:**")
            edited = st.text_area(
                "SQL", value=payload.get("sql", ""), height=260,
                key=f"sql_edit_{st.session_state.thread_id}",
                label_visibility="collapsed",
            )
            cols = st.columns([1, 1, 1, 4])
            if cols[0].button("▶ Run", key="sql_approve", type="primary"):
                _resume_graph({"action": "approve", "content": None})
            if cols[1].button("✏️ Run edited", key="sql_edit"):
                _resume_graph({"action": "edit", "content": edited})
            if cols[2].button("✖ Cancel", key="sql_cancel"):
                _resume_graph({"action": "cancel", "content": None})


# ---------- graph driving ----------

def _stream_until_interrupt_or_end(invoke_payload):
    """Run/resume the graph and stop at the next interrupt or terminal state."""
    cfg_conf = {"configurable": {"thread_id": st.session_state.thread_id}}

    interrupt_payload = None
    final_state = None
    try:
        for chunk in graph.stream(invoke_payload, config=cfg_conf, stream_mode="values"):
            final_state = chunk
        # After stream completes, check for interrupt
        snapshot = graph.get_state(cfg_conf)
        if snapshot.next:
            # We're paused at an interrupt; pull the interrupt payload from tasks.
            for task in snapshot.tasks:
                if task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break
    except Exception as e:
        _push_message("assistant", "error", f"Graph error: {e}")
        st.session_state.pending_interrupt = None
        return

    if interrupt_payload:
        st.session_state.pending_interrupt = interrupt_payload
    else:
        st.session_state.pending_interrupt = None
        _finalize_run(final_state or {})


def _resume_graph(response_value: dict):
    st.session_state.pending_interrupt = None
    _stream_until_interrupt_or_end(Command(resume=response_value))
    st.rerun()


def _finalize_run(state: dict):
    """Render the final outputs and record context for follow-ups."""
    status = state.get("status", "ok")
    intent = state.get("intent")

    if status == "cancelled":
        _push_message("assistant", "cancelled", "Cancelled. No SQL was executed.")
        return

    if state.get("error_message"):
        _push_message("assistant", "error", state["error_message"])
        return

    if intent == "FILE":
        _push_message("assistant", "files", state.get("file_results") or [])
        return

    # DB result
    if state.get("validation_errors") and not state.get("excel_path"):
        errs = "\n".join(state["validation_errors"])
        _push_message("assistant", "error",
                      f"Couldn't produce valid SQL after retries.\n\n```\n{errs}\n```")
        return

    if state.get("sql"):
        _push_message("assistant", "sql", state["sql"])

    _push_message("assistant", "result", {
        "row_count": state.get("row_count", 0),
        "truncated": state.get("truncated", False),
        "runtime_ms": state.get("runtime_ms", 0),
        "excel_path": state.get("excel_path"),
        "dataframe": state.get("dataframe"),
        "explanation": state.get("explanation"),
        "chart_html_path": state.get("chart_html_path"),
    })

    # Save into the in-session conversation context for follow-ups.
    if state.get("sql"):
        st.session_state.conversation_context.append({
            "question": state.get("question", ""),
            "sql": state["sql"],
            "row_count": state.get("row_count"),
        })
        # Keep last 5 turns
        st.session_state.conversation_context = st.session_state.conversation_context[-5:]


# ---------- render history ----------

for msg in st.session_state.messages:
    _render_message(msg)

# ---------- render any pending approval card ----------

if st.session_state.pending_interrupt is not None:
    p = st.session_state.pending_interrupt
    _approval_card(p.get("type"), p)


# ============================================================
# Input
# ============================================================

input_disabled = st.session_state.pending_interrupt is not None
prompt = st.chat_input(
    "Ask a question about the data..." if not input_disabled else "Approve or cancel the proposal above first.",
    disabled=input_disabled,
)

if prompt:
    _push_message("user", "text", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_state = {
        "question": prompt,
        "conversation_context": list(st.session_state.conversation_context),
        "want_chart": st.session_state.want_chart,
        "want_explanation": st.session_state.want_explanation,
        "retry_count": 0,
        "validation_errors": [],
        "schema_chunks": [],
        "example_pairs": [],
        "plan_approved": False,
        "sql_approved": False,
        "truncated": False,
        "status": "ok",
    }
    with st.spinner("Thinking..."):
        _stream_until_interrupt_or_end(initial_state)
    st.rerun()

