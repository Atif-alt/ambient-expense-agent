# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ambient Expense-Approval Agent — ADK 2.0 graph Workflow
========================================================

Graph topology
--------------

  START
    │
    ▼
  ┌─────────────┐
  │ parse_expense│  FunctionNode — decode base64 or plain JSON, extract fields
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │route_expense│  FunctionNode — pure-Python threshold check (NO LLM)
  └──────┬──────┘
         │
    ─────┴──────────────────────────
    │ "auto_approve"                │ "llm_review"
    ▼                               ▼
  ┌────────────┐          ┌──────────────────┐
  │auto_approve│          │ llm_risk_review  │  LlmAgent — risk factors + alert
  └─────┬──────┘          └────────┬─────────┘
        │                          │
        │                 ┌────────▼──────────────┐
        │                 │request_human_approval  │  FunctionNode + RequestInput HITL
        │                 └────────┬──────────────┘
        │                          │  (resumes on human reply — "approved" / "rejected")
        │                          │
    ────┴──────────────────────────┘
                    │   (both paths converge)
                    ▼
          ┌─────────────────┐
          │  record_outcome │  FunctionNode — write to ledger, emit ExpenseOutcome
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  format_output  │  Generator node — render markdown to ADK web UI
          └─────────────────┘

Key design decisions
--------------------
* Threshold routing is 100% Python (route_expense) — the LLM never sees cheap items.
* The LlmAgent (llm_risk_review) is used only for risk *assessment*, not for routing.
  Routing after HITL is also Python (parse the "approve"/"reject" string).
* RequestInput suspends the Workflow; ResumabilityConfig(handle_user_turns_as_resumption=True)
  means the next message in the playground automatically resumes the suspended session.
* record_outcome is a shared convergence point for both branches: it inspects node_input
  for a "status" key (auto path) vs. "decision" key (human path).
* All node I/O is typed with Pydantic models from expense_agent/schemas.py.
  ADK FunctionNodes auto-convert dict inputs to the annotated type.
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from datetime import date

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow
from google.genai import types

from expense_agent.config import config
from expense_agent.schemas import (
    ExpenseOutcome,
    ParsedExpense,
    RiskAssessment,
    WorkflowOutput,
)

# ── Authentication ─────────────────────────────────────────────────────────────
# load_dotenv() reads .env from the current working directory (project root).
# PATH A: GOOGLE_GENAI_USE_VERTEXAI=False + GOOGLE_API_KEY=<AI Studio key>
# PATH B: GOOGLE_GENAI_USE_VERTEXAI=True  + gcloud ADC credentials
load_dotenv()

_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("1", "true")
if _USE_VERTEXAI:
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 — parse_expense
# ══════════════════════════════════════════════════════════════════════════════
# Receives:  types.Content  (START output when no input_schema is set)
# Returns:   ParsedExpense  (auto-emitted as Event(output=...))
#
# Handles two envelope shapes so the same agent works in production (Pub/Sub)
# and locally (plain JSON in the playground):
#
#   Pub/Sub:     {"data": "<base64-encoded JSON>"}
#   Local/test:  {"amount": 42.0, "submitter": "alice@example.com", ...}
#                {"data": "{\"amount\": 42.0, ...}"}   ← plain-JSON "data" string


def parse_expense(
    node_input: types.Content,
) -> ParsedExpense:  # ← auto-wrapped as FunctionNode
    """Decode the incoming event and extract structured expense fields."""
    text = node_input.parts[0].text.strip() if node_input.parts else "{}"

    # ── Outer parse ────────────────────────────────────────────────────────
    try:
        raw: dict = json.loads(text)
    except json.JSONDecodeError:
        # Treat plain text as a description with no amount
        raw = {
            "description": text,
            "amount": 0.0,
            "submitter": "unknown",
            "category": "Other",
        }

    # ── Unwrap "data" envelope (Pub/Sub or forwarded webhook) ──────────────
    if "data" in raw and isinstance(raw["data"], str):
        data_field: str = raw["data"]
        # Try base64 first (real Pub/Sub adds padding issues — append "==" to be safe)
        try:
            inner = base64.b64decode(data_field + "==").decode("utf-8")
            raw = json.loads(inner)
        except Exception:
            # Fall back: treat "data" as a plain-JSON string (local shortcut)
            try:
                raw = json.loads(data_field)
            except Exception:
                raw["description"] = data_field  # last resort — use as description

    return ParsedExpense(
        amount_usd=float(raw.get("amount", raw.get("amount_usd", 0.0))),
        submitter=raw.get("submitter", raw.get("submitted_by", "unknown")),
        category=raw.get("category", "Other"),
        description=raw.get("description", ""),
        expense_date=raw.get("date", raw.get("expense_date", date.today().isoformat())),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 — route_expense
# ══════════════════════════════════════════════════════════════════════════════
# Receives:  ParsedExpense  (ADK auto-converts dict from parse_expense output)
# Returns:   Event with route="auto_approve" or "llm_review"
#
# This is the ONLY routing decision in the graph. It is 100% Python.
# No model is called here.  The threshold comes from config.auto_approve_threshold.


def route_expense(ctx: Context, node_input: ParsedExpense) -> Event:
    """Pure-Python threshold routing — no LLM involved."""
    route = (
        "auto_approve"
        if node_input.amount_usd < config.auto_approve_threshold
        else "llm_review"
    )
    return Event(
        output=node_input.model_dump(),
        actions=EventActions(
            route=route,
            state_delta={
                "expense": node_input.model_dump(),
            },
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node 3a — auto_approve  (reached via "auto_approve" edge)
# ══════════════════════════════════════════════════════════════════════════════
# Receives:  dict  (ParsedExpense fields — no ctx needed)
# Returns:   ExpenseOutcome  (ADK serialises Pydantic models to dict automatically)


def auto_approve(node_input: dict) -> ExpenseOutcome:
    """Instantly approve expenses below config.auto_approve_threshold."""
    return ExpenseOutcome(
        expense_id=f"AUTO-{uuid.uuid4().hex[:8].upper()}",
        status="auto_approved",
        amount_usd=node_input["amount_usd"],
        submitter=node_input["submitter"],
        category=node_input["category"],
        description=node_input["description"],
        expense_date=node_input["expense_date"],
        reviewer_note=(
            f"Auto-approved: USD {node_input['amount_usd']:.2f} "
            f"is below the ${config.auto_approve_threshold:.0f} threshold."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node 3b — llm_risk_review  (reached via "llm_review" edge)
# ══════════════════════════════════════════════════════════════════════════════
# An LlmAgent node — auto-wrapped by the Workflow engine.
# Receives:  the ParsedExpense dict from route_expense as its conversation input.
# Returns:   RiskAssessment  (via output_schema — ADK enforces the JSON shape)
#            Also written to ctx.state["risk_assessment"] via output_key.
#
# IMPORTANT: the LLM is only asked to *assess risk*.
#            It never approves or rejects — that decision belongs to the human.

llm_risk_review = LlmAgent(
    name="llm_risk_review",
    model=config.risk_review_model,  # ← set in config.py / env var RISK_REVIEW_MODEL
    instruction="""\
You are a corporate expense risk analyst. You receive expense details as a JSON object.

Analyse the expense for risk factors such as:
- Amount unusually high for the stated category (e.g. > $500 for a single meal)
- Vague, generic, or unverifiable description
- Weekend or public-holiday submission dates
- Possible personal expense (spa, luxury hotel upgrade, alcohol)
- Round-number amounts that suggest estimation rather than a real receipt
- Missing or implausible category for the described item

Respond ONLY with a valid JSON object matching this exact schema — no markdown, no extra text:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "risk_factors": ["<factor>", ...],    // empty array if none
  "recommendation": "<brief note for the human reviewer — max 2 sentences>",
  "alert_message": "<one-line summary shown at the top of the review card>"
}

Do NOT approve or reject. Only assess risk. Be concise and factual.\
""",
    output_schema=RiskAssessment,  # ← structured output enforced by ADK
    output_key="risk_assessment",  # ← also stored in ctx.state["risk_assessment"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Node 4 — request_human_approval  (HITL gate, reached after llm_risk_review)
# ══════════════════════════════════════════════════════════════════════════════
# Receives:  RiskAssessment  (ADK auto-converts dict from LlmAgent output)
# Yields:    multiple Events — an async generator node
#
# Turn 1  (interrupt_id absent in ctx.resume_inputs):
#   • Emit a formatted review card as content (visible in ADK web UI).
#   • Yield RequestInput — this suspends the Workflow and returns control to
#     the user.  The session stays alive in state.
#
# Turn 2  (interrupt_id present — the human has replied):
#   • ctx.resume_inputs["human_approval"] contains the raw reply string.
#   • Parse "approve..." / "reject..." and emit an Event with the decision.
#   • Execution resumes and flows to record_outcome.
#
# ResumabilityConfig(handle_user_turns_as_resumption=True) on the App means
# the next playground message automatically resumes the suspended session —
# no special client-side resume call needed.


async def request_human_approval(ctx: Context, node_input: RiskAssessment):
    """Pause for human approval; resume when the reviewer replies."""
    interrupt_id = "human_approval"
    expense: dict = ctx.state.get("expense", {})

    # ── Turn 1: display and suspend ────────────────────────────────────────
    if interrupt_id not in (ctx.resume_inputs or {}):
        risk_icon = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}.get(
            node_input.risk_level, "⚪"
        )
        factors_md = (
            "\n".join(f"  - {f}" for f in node_input.risk_factors)
            if node_input.risk_factors
            else "  - None identified"
        )
        message = (
            f"## ⚠️ Expense Approval Required\n\n"
            f"> **{node_input.alert_message}**\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Submitter | {expense.get('submitter', 'N/A')} |\n"
            f"| Amount | USD {expense.get('amount_usd', 0):.2f} |\n"
            f"| Category | {expense.get('category', 'N/A')} |\n"
            f"| Date | {expense.get('expense_date', 'N/A')} |\n"
            f"| Description | {str(expense.get('description', 'N/A'))[:120]} |\n\n"
            f"**Risk Level:** {risk_icon} {node_input.risk_level}\n\n"
            f"**Risk Factors:**\n{factors_md}\n\n"
            f"**LLM Recommendation:** {node_input.recommendation}\n\n"
            f"---\n"
            f"Reply **`approve`** (optionally with a note) or **`reject <reason>`** to continue."
        )

        # Emit content — rendered in ADK web UI immediately
        yield Event(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=message)],
            )
        )
        # Suspend the Workflow here — execution resumes on the next user message
        yield RequestInput(interrupt_id=interrupt_id, message=message)
        return  # ← workflow pauses; the lines below only run on resume

    # ── Turn 2: parse the human's reply ────────────────────────────────────
    raw_reply: str = str(ctx.resume_inputs[interrupt_id]).strip()
    lower = raw_reply.lower()

    if lower.startswith("approve"):
        note = raw_reply[len("approve") :].strip(" :,.") or "Approved by reviewer."
        decision = "approved"
    else:
        note = (
            re.sub(r"^reject\s*:?\s*", "", raw_reply, flags=re.IGNORECASE).strip()
            or "Rejected by reviewer."
        )
        decision = "rejected"

    # No route label — both "approved" and "rejected" flow unconditionally
    # to record_outcome.  The decision is carried in the output dict.
    yield Event(
        output={"decision": decision, "reviewer_note": note},
        actions=EventActions(
            state_delta={
                "human_decision": {
                    "decision": decision,
                    "reviewer_note": note,
                }
            }
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node 5 — record_outcome  (shared convergence point for both branches)
# ══════════════════════════════════════════════════════════════════════════════
# Receives:  dict — but the shape differs by branch:
#   • auto_approve path:  ExpenseOutcome fields (has key "status")
#   • human review path:  {"decision": "approved"|"rejected", "reviewer_note": "..."}
#
# The discriminator is the presence of "status" in node_input.
# Expense details for the human path are recovered from ctx.state["expense"].


def record_outcome(ctx: Context, node_input: dict) -> Event:
    """Persist the final expense record; emit a complete ExpenseOutcome."""
    if "status" in node_input:
        # ── Fast path: auto_approve already built the full outcome ─────────
        outcome = ExpenseOutcome(**node_input)
    else:
        # ── Human-review path: reconstruct from state ──────────────────────
        expense: dict = ctx.state.get("expense", {})
        risk: dict = ctx.state.get("risk_assessment", {})
        outcome = ExpenseOutcome(
            expense_id=f"EXP-{uuid.uuid4().hex[:8].upper()}",
            status=node_input["decision"],  # "approved" or "rejected"
            amount_usd=expense.get("amount_usd", 0.0),
            submitter=expense.get("submitter", ""),
            category=expense.get("category", ""),
            description=expense.get("description", ""),
            expense_date=expense.get("expense_date", ""),
            reviewer_note=node_input.get("reviewer_note", ""),
            risk_assessment=risk,
        )

    # Append to in-memory ledger — swap for Firestore / BigQuery in production
    ledger: list = ctx.state.get("ledger", [])
    ledger.append(outcome.model_dump())

    return Event(
        output=outcome.model_dump(),
        actions=EventActions(
            state_delta={
                "ledger": ledger,
                "last_outcome": outcome.model_dump(),
            }
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node 6 — format_output  (terminal node)
# ══════════════════════════════════════════════════════════════════════════════
# Receives:  dict  (ExpenseOutcome fields from record_outcome)
# Yields:    Event(content=...)  — rendered in ADK web UI
#            Event(output=...)   — structured payload for programmatic callers
#
# In ADK 2.0, event.output is internal; only event.content is visible in the
# web playground.  We emit both so the agent is useful in both contexts.


def format_output(node_input: dict):
    """Render the workflow result to the ADK web UI and emit structured output."""
    status = node_input.get("status", "unknown")
    expense_id = node_input.get("expense_id", "N/A")
    amount = float(node_input.get("amount_usd", 0))
    submitter = node_input.get("submitter", "?")
    note = node_input.get("reviewer_note", "")
    category = node_input.get("category", "")
    expense_date = node_input.get("expense_date", "")

    icon, headline = {
        "auto_approved": ("✅", "Auto-Approved"),
        "approved": ("✅", "Approved by Reviewer"),
        "rejected": ("❌", "Rejected"),
    }.get(status, ("⚪", "Unknown status"))

    message = (
        f"{icon} **{headline}**\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| ID | `{expense_id}` |\n"
        f"| Submitter | {submitter} |\n"
        f"| Amount | USD {amount:.2f} |\n"
        f"| Category | {category} |\n"
        f"| Date | {expense_date} |\n"
        f"| Note | {note} |"
    )

    # ① Visible in ADK web UI
    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )
    )
    # ② Structured payload for API / eval callers
    yield Event(
        output=WorkflowOutput(
            expense_id=expense_id,
            status=status,  # type: ignore[arg-type]
            summary=message,
        ).model_dump()
    )


# ══════════════════════════════════════════════════════════════════════════════
# Workflow graph
# ══════════════════════════════════════════════════════════════════════════════
#
# Edges are declared as (source, target) or (source, target, "route_label").
# Plain functions and LlmAgent instances are auto-wrapped by the Workflow engine.
#
#  ① START         → parse_expense                  (always)
#  ② parse_expense → route_expense                  (always)
#  ③ route_expense → auto_approve         "auto_approve"
#  ④ route_expense → llm_risk_review      "llm_review"
#  ⑤ llm_risk_review → request_human_approval       (always)
#  ⑥ auto_approve  → record_outcome                 (always — fast-path merge)
#  ⑦ request_human_approval → record_outcome        (always — human path merge)
#  ⑧ record_outcome → format_output                 (always)

root_agent = Workflow(
    name="expense_approval_workflow",
    description=(
        "Ambient expense-approval agent: auto-approves expenses below "
        f"${config.auto_approve_threshold:.0f}, uses an LLM to assess risk for "
        "larger amounts, then gates on human approval via RequestInput."
    ),
    edges=[
        ("START", parse_expense),
        (parse_expense, route_expense),
        (route_expense, {"auto_approve": auto_approve}),
        (route_expense, {"llm_review": llm_risk_review}),
        (llm_risk_review, request_human_approval),
        (auto_approve, record_outcome),
        (request_human_approval, record_outcome),
        (record_outcome, format_output),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# App — wraps the Workflow for agents-cli playground and FastAPI
# ══════════════════════════════════════════════════════════════════════════════
#
# ResumabilityConfig(handle_user_turns_as_resumption=True):
#   When the Workflow is suspended at a RequestInput, the next message the user
#   sends in the playground is automatically treated as a resume input rather
#   than a new session.  This is what makes the HITL gate work seamlessly
#   without any client-side code changes.

app = App(
    name="app",  # must match agent_directory in agents-cli-manifest.yaml
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(
        handle_user_turns_as_resumption=True,
    ),
)
