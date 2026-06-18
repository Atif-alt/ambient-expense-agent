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

"""Pydantic schemas for every node boundary in the expense-approval workflow.

Each schema represents the *typed contract* between two adjacent nodes.
ADK FunctionNodes auto-convert dict inputs to the annotated Pydantic type,
so downstream nodes receive a proper model instance rather than a raw dict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Input envelope (raw event from Pub/Sub or local test) ─────────────────────


class RawEvent(BaseModel):
    """Minimal shape expected at START.  parse_expense handles all decoding."""

    data: str | None = None
    """Base-64 encoded JSON (real Pub/Sub) or plain JSON string (local tests)."""


# ── After parse_expense ───────────────────────────────────────────────────────


class ParsedExpense(BaseModel):
    """Normalised expense — the canonical representation throughout the graph."""

    amount_usd: float = Field(description="Expense amount in US dollars.")
    submitter: str = Field(description="Name or email of the submitter.")
    category: str = Field(description="Expense category, e.g. Travel, Meals.")
    description: str = Field(
        description="Free-text description from the receipt / form."
    )
    expense_date: str = Field(description="ISO-8601 date (YYYY-MM-DD).")


# ── After llm_risk_review (LlmAgent output_schema) ────────────────────────────


class RiskAssessment(BaseModel):
    """Risk analysis produced by the LLM — shown verbatim to the human reviewer."""

    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Specific concerns found (empty list if none).",
    )
    recommendation: str = Field(description="Brief guidance for the human reviewer.")
    alert_message: str = Field(
        description="One-line summary shown at the top of the review card."
    )


# ── Final outcome (written to ledger, passed to format_output) ────────────────


class ExpenseOutcome(BaseModel):
    """Complete record of what happened to an expense."""

    expense_id: str
    status: Literal["auto_approved", "approved", "rejected"]
    amount_usd: float
    submitter: str
    category: str
    description: str
    expense_date: str
    reviewer_note: str = ""
    risk_assessment: dict = Field(
        default_factory=dict,
        description="Persisted RiskAssessment dict (empty for auto-approved expenses).",
    )


# ── Structured output returned to API callers ─────────────────────────────────


class WorkflowOutput(BaseModel):
    """Final structured payload emitted by format_output for programmatic use."""

    expense_id: str
    status: Literal["auto_approved", "approved", "rejected"]
    summary: str = Field(description="Human-readable markdown summary.")
