# AI-Powered Expense Approval Agent

## Overview

AI-Powered Expense Approval Agent is a business workflow automation project built using Google ADK 2.0 concepts.

The system automatically processes expense submissions, approves low-risk expenses, and routes higher-value expenses through a review workflow. It demonstrates AI agent orchestration, decision routing, human-in-the-loop approval, and business process automation.

## Track

Agents for Business

## Problem Statement

Organizations often spend significant time reviewing employee expense claims manually. This creates delays, increases operational workload, and reduces scalability.

This project automates expense approvals while maintaining human oversight for higher-risk transactions.

## Features

* Automated expense processing
* Rule-based approval workflow
* Human review for high-value expenses
* Dashboard integration
* Google ADK 2.0 workflow concepts
* Agent-based decision routing

## System Workflow

1. User submits an expense.
2. Expense data is processed by the agent.
3. Expenses below $100 are automatically approved.
4. Expenses above $100 are routed for review.
5. Human approval is requested when required.
6. Final decision is generated and recorded.

## Technology Stack

* Google ADK 2.0
* Gemini
* Python
* HTML
* CSS
* JavaScript
* GitHub

## Architecture

User → Dashboard → Expense Agent → Decision Engine

* Amount < $100 → Auto Approval
* Amount ≥ $100 → Risk Review → Human Approval

## Repository Structure

* expense_agent/ — Agent workflow implementation
* app/ — Application layer
* tests/ — Testing utilities
* config/ — Configuration management

## Future Enhancements

* Database integration
* Cloud deployment
* Real-time notifications
* Advanced risk scoring
* Multi-user support

## Author

MD Atifuddin

GITAM University

Built as part of the Google × Kaggle 5-Day AI Agents Intensive Course.
