import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "API_KEY"

DATA_DIR = Path("data")
META_DIR = DATA_DIR / "metadata"

TABLE_INDEX_PATH = META_DIR / "tables.json"

# ---------------------------------------------------------------------
# LOAD METADATA
# ---------------------------------------------------------------------
with open(TABLE_INDEX_PATH, "r", encoding="utf-8") as f:
    TABLE_INDEX = json.load(f)


# ---------------------------------------------------------------------
# TOOL 1: TABLE RESOLVER
# ---------------------------------------------------------------------
class TableResolverTool(BaseTool):
    name = "resolve_tables"
    description = (
        "Given a football analysis question, return the relevant FBref tables "
        "needed to answer it."
    )

    def _run(self, query: str):
        query_lower = query.lower()
        matching_tables = []

        for table_name, meta in TABLE_INDEX.items():
            keywords = meta.get("keywords", [])
            if any(k.lower() in query_lower for k in keywords):
                matching_tables.append(meta["file"])

        if not matching_tables:
            return "No relevant tables found."

        return ", ".join(sorted(set(matching_tables)))

    async def _arun(self, query: str):
        return self._run(query)


# ---------------------------------------------------------------------
# TOOL 2: FOOTBALL ANALYTICS (PANDAS)
# ---------------------------------------------------------------------
class TottenhamAnalyticsTool(BaseTool):
    name = "tottenham_analysis"
    description = (
        "Run pandas-based analysis on Tottenham FBref CSV tables to answer "
        "questions about minutes, xG, possession, etc."
    )

    # ------------- helpers -------------

    def _load_player_table(self, filename: str, player_col: str) -> pd.DataFrame:
        """Load a stats_* CSV and clean out header/junk rows."""
        df = pd.read_csv(DATA_DIR / filename)

        # drop header/total rows
        junk = {"Player", "Squad Total", "Opponent Total"}
        df = df[~df[player_col].isin(junk)]
        df = df.dropna(subset=[player_col])

        return df

    # ------------- main logic -------------

    def _run(self, query: str):
        q = query.lower()

        # ---------------------------------------------------
        # 1) Who played the most minutes for Tottenham?
        #    -> stats_playing_time_9.csv
        # ---------------------------------------------------
        if "minute" in q or "played the most" in q or "most minutes" in q:
            player_col = "Unnamed: 0_level_0_Player"
            df = self._load_player_table("stats_playing_time_9.csv", player_col)

            minutes_col = "Playing Time_Min"
            starts_col = "Starts_Starts"

            top = (
                df.sort_values(minutes_col, ascending=False)
                  .loc[:, [player_col, minutes_col, starts_col]]
                  .head(5)
            )

            top = top.rename(
                columns={
                    player_col: "Player",
                    minutes_col: "Minutes",
                    starts_col: "Starts",
                }
            )

            return (
                "Top Spurs players by minutes played (Premier League 2025-26):\n\n"
                f"{top.to_string(index=False)}"
            )

        # ---------------------------------------------------
        # 2) Which players overperformed their xG?
        #    -> stats_standard_9.csv  (Performance_Gls, Expected_xG)
        # ---------------------------------------------------
        if "xg" in q and ("overperform" in q or "outperform" in q or "over performed" in q):
            player_col = "Unnamed: 0_level_0_Player"
            df = self._load_player_table("stats_standard_9.csv", player_col)

            goals_col = "Performance_Gls"
            xg_col = "Expected_xG"

            df["xG_diff"] = df[goals_col] - df[xg_col]

            top = (
                df.sort_values("xG_diff", ascending=False)
                  .loc[:, [player_col, goals_col, xg_col, "xG_diff"]]
                  .head(5)
            )

            top = top.rename(
                columns={
                    player_col: "Player",
                    goals_col: "Gls",
                    xg_col: "xG",
                    "xG_diff": "Gls - xG",
                }
            )

            return (
                "Top Spurs xG overperformers (goals minus expected goals):\n\n"
                f"{top.to_string(index=False)}"
            )

        # Also handle simpler “xG” question without the word “overperform”
        if "xg" in q:
            player_col = "Unnamed: 0_level_0_Player"
            df = self._load_player_table("stats_standard_9.csv", player_col)

            goals_col = "Performance_Gls"
            xg_col = "Expected_xG"

            df["xG_diff"] = df[goals_col] - df[xg_col]

            top = (
                df.sort_values("xG_diff", ascending=False)
                  .loc[:, [player_col, goals_col, xg_col, "xG_diff"]]
                  .head(5)
            )

            top = top.rename(
                columns={
                    player_col: "Player",
                    goals_col: "Gls",
                    xg_col: "xG",
                    "xG_diff": "Gls - xG",
                }
            )

            return (
                "Spurs players sorted by goals minus xG (Gls - xG):\n\n"
                f"{top.to_string(index=False)}"
            )

        # ---------------------------------------------------
        # 3) Who are the top possession contributors?
        #    -> stats_possession_9.csv (Touches_Touches, Carries_PrgC)
        # ---------------------------------------------------
        if "possession" in q or "touches" in q or "on the ball" in q:
            player_col = "Unnamed: 0_level_0_Player"
            df = self._load_player_table("stats_possession_9.csv", player_col)

            touches_col = "Touches_Touches"
            prgc_col = "Carries_PrgC"

            top = (
                df.sort_values(touches_col, ascending=False)
                  .loc[:, [player_col, touches_col, prgc_col]]
                  .head(5)
            )

            top = top.rename(
                columns={
                    player_col: "Player",
                    touches_col: "Touches",
                    prgc_col: "Prog Carries",
                }
            )

            return (
                "Top Spurs possession contributors (touches & progressive carries):\n\n"
                f"{top.to_string(index=False)}"
            )

        # ---------------------------------------------------
        # fallback
        # ---------------------------------------------------
        return (
            "I couldn’t match this question to a rule yet.\n"
            "Right now I can answer:\n"
            "- Who played the most minutes for Tottenham?\n"
            "- Which players overperformed their xG?\n"
            "- Who are the top possession contributors?\n"
        )

    async def _arun(self, query: str):
        return self._run(query)


class PlayerProfileTool(BaseTool):
    name = "player_profiles"
    description = (
        "Returns structured, position-aware performance profiles for "
        "Tottenham players to support reasoning and evaluation."
    )

    def _run(self, query: str):  # ✅ IMPORTANT: must be named, non-underscore
        profiles = {}

        # ---------- Standard ----------
        std = pd.read_csv(DATA_DIR / "stats_standard_9.csv")
        for _, r in std.iterrows():
            player = r["Unnamed: 0_level_0_Player"]
            if player in {"Player", "Squad Total", "Opponent Total"}:
                continue

            profiles[player] = {
                "position": r["Unnamed: 2_level_0_Pos"],
                "age": r["Unnamed: 3_level_0_Age"],
                "minutes": r["Playing Time_Min"],
                "starts": r["Playing Time_Starts"],
                "goals": r["Performance_Gls"],
                "assists": r["Performance_Ast"],
                "xg": r["Expected_xG"],
                "xag": r["Expected_xAG"],
                "prog_carries": r["Progression_PrgC"],
                "prog_passes": r["Progression_PrgP"],
            }

        # ---------- Possession ----------
        poss = pd.read_csv(DATA_DIR / "stats_possession_9.csv")
        for _, r in poss.iterrows():
            player = r["Unnamed: 0_level_0_Player"]
            if player in profiles:
                profiles[player].update({
                    "touches": r["Touches_Touches"],
                    "progressive_carries": r["Carries_PrgC"],
                    "progressive_receives": r["Receiving_PrgR"],
                })

        # ---------- Defense ----------
        defense = pd.read_csv(DATA_DIR / "stats_defense_9.csv")
        for _, r in defense.iterrows():
            player = r["Unnamed: 0_level_0_Player"]
            if player in profiles:
                profiles[player].update({
                    "tackles": r["Tackles_Tkl"],
                    "interceptions": r["Unnamed: 17_level_0_Int"],
                    "errors": r["Unnamed: 20_level_0_Err"],
                })

        # ---------- Remove tiny samples ----------
        profiles = {
            p: v for p, v in profiles.items()
            if v.get("minutes", 0) >= 200
        }

        return json.dumps(profiles, indent=2)

    async def _arun(self, query: str):
        return self._run(query)


class WorkloadProfileTool(BaseTool):
    name = "workload_profiles"
    description = (
        "Provides workload and usage context for Tottenham players to support "
        "relative injury-risk reasoning (non-medical)."
    )

    def _run(self, query: str):
        profiles = {}

        playing = pd.read_csv(DATA_DIR / "stats_playing_time_9.csv")
        standard = pd.read_csv(DATA_DIR / "stats_standard_9.csv")

        for _, r in playing.iterrows():
            player = r["Unnamed: 0_level_0_Player"]
            if player in {"Player", "Squad Total", "Opponent Total"}:
                continue

            profiles[player] = {
                "minutes": r["Playing Time_Min"],
                "starts": r["Starts_Starts"],
                "minutes_per_start": r["Starts_Mn/Start"],
                "ninety_equivalents": r["Playing Time_90s"],
            }

        # merge age & position
        for _, r in standard.iterrows():
            player = r["Unnamed: 0_level_0_Player"]
            if player in profiles:
                profiles[player].update({
                    "age": r["Unnamed: 3_level_0_Age"],
                    "position": r["Unnamed: 2_level_0_Pos"],
                })

        # filter low sample players
        profiles = {
            p: v for p, v in profiles.items()
            if v.get("minutes", 0) >= 300
        }

        return json.dumps(profiles, indent=2)

    async def _arun(self, query: str):
        return self._run(query)


SYSTEM_PROMPT = """
You are a professional football performance analyst.

You MUST:
- Base conclusions strictly on the provided data
- Evaluate players relative to others in the same position
- Consider minutes played when judging performance
- Identify underperformance using contribution vs opportunity
- Explain conclusions clearly using evidence from the data

DO NOT:
- Use external football knowledge
- Make claims not grounded in the data
- Assume fixed performance thresholds
"""


# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    system_message=SYSTEM_PROMPT
)

# ---------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------
tools = [
    TableResolverTool(),
    TottenhamAnalyticsTool(),
    PlayerProfileTool(),
    WorkloadProfileTool()
]

# ---------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ---------------------------------------------------------------------
# TEST QUERIES
# ---------------------------------------------------------------------
if __name__ == "__main__":
    queries = [
        "Who is the most underperforming player?",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print("QUESTION:", q)
        answer = agent.run(q)
        print("\nANSWER:\n", answer)
