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

"""Pydantic schemas for the ambient-expense-agent workflow nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Input / intermediate schemas
# ---------------------------------------------------------------------------


class RawExpenseInput(BaseModel):
    """Raw expense data submitted by the user or an event trigger."""

    description: str = Field(
        description="Free-text description of the expense (e.g. receipt text, email body)."
    )
    amount_str: str = Field(
        description="Amount as a raw string, e.g. '$42.50' or '42.50 USD'."
    )
    submitted_by: str = Field(
        default="unknown",
        description="Name or email of the person submitting the expense.",
    )
    date_str: str = Field(
        default="",
        description="Date string from the source document; empty means today.",
    )


class ParsedExpense(BaseModel):
    """Normalised expense after parse_expense node."""

    description: str
    amount_usd: float
    submitted_by: str
    expense_date: str  # ISO-8601 YYYY-MM-DD
    currency: str = "USD"


class ClassifiedExpense(BaseModel):
    """Expense enriched with AI-derived classification from classify_expense LlmAgent."""

    description: str
    amount_usd: float
    submitted_by: str
    expense_date: str
    currency: str
    category: str = Field(
        description="Expense category, e.g. Travel, Meals, Software, Hardware, Other."
    )
    policy_compliant: bool = Field(
        description="True if the expense appears to comply with standard expense policy."
    )
    compliance_notes: str = Field(
        description="Brief explanation of any policy concerns, or 'OK' if fully compliant."
    )
    suggested_gl_code: str = Field(
        description="Suggested general-ledger account code, e.g. '6010-TRAVEL'."
    )


class ReviewDecision(BaseModel):
    """Decision captured from the human-in-the-loop review step."""

    decision: Literal["approved", "rejected"]
    reviewer_note: str = ""


class RecordedExpense(BaseModel):
    """Expense entry written to the ledger / output store."""

    expense_id: str
    description: str
    amount_usd: float
    submitted_by: str
    expense_date: str
    currency: str
    category: str
    gl_code: str
    status: Literal["approved", "rejected"]
    reviewer_note: str


class WorkflowOutput(BaseModel):
    """Final structured output emitted to the ADK web UI and callers."""

    expense_id: str
    status: Literal["approved", "rejected"]
    summary: str
