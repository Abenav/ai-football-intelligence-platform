# AI-Powered Football Intelligence Platform

An AI-powered football analytics platform that combines Large Language Models (LLMs), structured football datasets, and intelligent workflows to generate actionable insights.

## Overview

This project enables users to interact with football data using natural language instead of static dashboards or manual spreadsheet analysis.

Users can ask questions such as:

- Who played the most minutes this season?
- Which players outperformed expected goals (xG)?
- Suggest the best XI against a specific opponent.
- Compare players using advanced metrics.

## Key Features

- Conversational football analytics
- Player performance insights
- Opponent-aware lineup recommendations
- Automated trend detection
- Multi-agent analytics workflows
- Structured CSV data analysis

## Tech Stack

- Python
- Pandas
- LangChain
- Google Gemini API
- BeautifulSoup
- StatsBombPy

## Project Structure

```text
ai-football-intelligence-platform/
├── README.md
├── requirements.txt
├── main.py
├── agents/
├── data/
├── screenshots/
└── docs/
```

## Installation

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
export GOOGLE_API_KEY=your_key_here
```

Run the project:

```bash
python main.py
```

## Future Improvements

- Live football API integrations
- Web dashboard / chatbot UI
- Cloud deployment
- Multi-league support
- Advanced scouting recommendations

## License

MIT License
