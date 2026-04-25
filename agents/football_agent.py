import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "API_KEY"  # <-- put your key

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "fbref" / "premier_league_2025_26"

# known team folder names (keys are folder names)
VALID_TEAMS = {"spurs", "brentford"}

PLAYER_COL = "Unnamed: 0_level_0_Player"


# ---------------------------------------------------------------------
# HELPERS: TEAM / QUERY PARSING
# ---------------------------------------------------------------------
def detect_team_and_opponent(query: str) -> Dict[str, Any]:
    """
    Very simple heuristic parser to guess team and opponent from natural
    language like:
      - "Suggest the best starting XI for Spurs against Brentford"
      - "Give me Brentford's XI vs Spurs"
    """
    q = query.lower()

    team = None
    opponent = None

    # detect mentions
    mentions_spurs = any(w in q for w in ["spurs", "tottenham"])
    mentions_brentford = "brentford" in q

    # try 'for X against Y' patterns
    if "against brentford" in q or "vs brentford" in q or "v brentford" in q:
        opponent = "brentford"
        if mentions_spurs:
            team = "spurs"
    if "against spurs" in q or "vs spurs" in q or "v spurs" in q or "tottenham" in q:
        if opponent is None:
            opponent = "spurs"
        if mentions_brentford and team is None:
            team = "brentford"

    # fallback: if only one team mentioned, assume that's the team
    if team is None:
        if mentions_spurs and not mentions_brentford:
            team = "spurs"
        elif mentions_brentford and not mentions_spurs:
            team = "brentford"

    # default hard fallback
    if team is None:
        team = "spurs"
    if opponent is None:
        opponent = "brentford" if team == "spurs" else "spurs"

    return {"team": team, "opponent": opponent}


def detect_formation(query: str) -> str:
    """
    Detects a formation from the query if explicitly mentioned,
    otherwise returns 'auto' (to be interpreted as default in the tool).
    """
    q = query.replace(" ", "")
    for f in ["4-2-3-1", "4-3-3", "3-4-3", "4-4-2", "3-5-2"]:
        if f.replace("-", "") in q:
            return f
    return "auto"


def get_team_path(team_key: str) -> Path:
    """
    Map a team key like 'spurs' to its folder under DATA_ROOT.
    """
    key = team_key.lower()
    if key not in VALID_TEAMS:
        raise ValueError(f"Unknown team '{team_key}'. Known: {VALID_TEAMS}")
    path = DATA_ROOT / key
    if not path.exists():
        raise FileNotFoundError(f"Data folder for team '{team_key}' not found at {path}")
    return path


def load_team_csv(team_key: str, filename: str) -> pd.DataFrame:
    folder = get_team_path(team_key)
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV '{filename}' not found for team '{team_key}' at {path}")
    return pd.read_csv(path)


def clean_player_df(df: pd.DataFrame, player_col: str = PLAYER_COL) -> pd.DataFrame:
    """
    Remove header and total rows for player-based tables.
    """
    if player_col not in df.columns:
        return df
    junk = {"Player", "Squad Total", "Opponent Total"}
    df = df[~df[player_col].isin(junk)]
    df = df.dropna(subset=[player_col])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# TOOL: CLUB PROFILES (GENERIC, ANY TEAM)
# ---------------------------------------------------------------------
class ClubProfilesTool(BaseTool):
    name = "club_profiles"
    description = (
        "Builds structured performance profiles for players of a club "
        "based on FBref stats. The club is inferred from the question "
        "(supports Spurs and Brentford). Returns JSON."
    )

    def _run(self, query: str) -> str:
        info = detect_team_and_opponent(query)
        team = info["team"]

        std = clean_player_df(load_team_csv(team, "stats_standard_9.csv"))
        pos = clean_player_df(load_team_csv(team, "stats_possession_9.csv"))
        dfn = clean_player_df(load_team_csv(team, "stats_defense_9.csv"))
        play = clean_player_df(load_team_csv(team, "stats_playing_time_9.csv"))

        profiles: Dict[str, Dict[str, Any]] = {}

        # ---------- Standard ----------
        for _, r in std.iterrows():
            player = r[PLAYER_COL]
            profiles[player] = {
                "team": team,
                "position": r["Unnamed: 2_level_0_Pos"],
                "age": r["Unnamed: 3_level_0_Age"],
                "minutes": float(r.get("Playing Time_Min", 0) or 0),
                "starts": float(r.get("Playing Time_Starts", 0) or 0),
                "goals": float(r.get("Performance_Gls", 0) or 0),
                "assists": float(r.get("Performance_Ast", 0) or 0),
                "xg": float(r.get("Expected_xG", 0) or 0),
                "xag": float(r.get("Expected_xAG", 0) or 0),
                "prog_carries": float(r.get("Progression_PrgC", 0) or 0),
                "prog_passes": float(r.get("Progression_PrgP", 0) or 0),
            }

        # ---------- Possession ----------
        for _, r in pos.iterrows():
            player = r[PLAYER_COL]
            if player not in profiles:
                continue
            profiles[player].update(
                {
                    "touches": float(r.get("Touches_Touches", 0) or 0),
                    "progressive_carries": float(r.get("Carries_PrgC", 0) or 0),
                    "progressive_receives": float(r.get("Receiving_PrgR", 0) or 0),
                }
            )

        # ---------- Defense ----------
        for _, r in dfn.iterrows():
            player = r[PLAYER_COL]
            if player not in profiles:
                continue
            profiles[player].update(
                {
                    "tackles": float(r.get("Tackles_Tkl", 0) or 0),
                    "interceptions": float(r.get("Unnamed: 17_level_0_Int", 0) or 0),
                    "errors": float(r.get("Unnamed: 20_level_0_Err", 0) or 0),
                }
            )

        # ---------- Playing time ----------
        max_minutes = 0.0
        minutes_map: Dict[str, float] = {}
        n90_map: Dict[str, float] = {}

        for _, r in play.iterrows():
            player = r[PLAYER_COL]
            minutes = float(r.get("Playing Time_Min", 0) or 0)
            n90 = float(r.get("Playing Time_90s", 0) or 0)
            minutes_map[player] = minutes
            n90_map[player] = n90
            if minutes > max_minutes:
                max_minutes = minutes

        for player, prof in profiles.items():
            mins = minutes_map.get(player, prof.get("minutes", 0))
            n90 = n90_map.get(player, 0)
            prof["minutes"] = mins
            prof["ninety_equivalents"] = n90
            prof["workload_ratio"] = mins / max_minutes if max_minutes > 0 else 0.0

            # per 90 metrics
            denom = n90 if n90 > 0 else 1.0
            prof["attack_per90"] = (prof.get("goals", 0) + prof.get("assists", 0)) / denom
            prof["defend_per90"] = (
                prof.get("tackles", 0) + prof.get("interceptions", 0)
            ) / denom
            prof["carry_per90"] = (
                prof.get("progressive_carries", 0) + prof.get("progressive_receives", 0)
            ) / denom

        # filter players with tiny samples
        profiles = {
            p: v for p, v in profiles.items() if v.get("minutes", 0) >= 200
        }

        return json.dumps(
            {
                "team": team,
                "players": profiles,
            },
            indent=2,
        )

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ---------------------------------------------------------------------
# TOOL: WORKLOAD PROFILES (GENERIC, ANY TEAM)
# ---------------------------------------------------------------------
class WorkloadProfilesTool(BaseTool):
    name = "workload_profiles"
    description = (
        "Provides workload and usage context for a club's players "
        "to support injury-risk reasoning (non-medical). "
        "Club is inferred from the question."
    )

    def _run(self, query: str) -> str:
        info = detect_team_and_opponent(query)
        team = info["team"]

        play = clean_player_df(load_team_csv(team, "stats_playing_time_9.csv"))
        std = clean_player_df(load_team_csv(team, "stats_standard_9.csv"))

        profiles: Dict[str, Dict[str, Any]] = {}
        max_minutes = 0.0

        for _, r in play.iterrows():
            player = r[PLAYER_COL]
            mins = float(r.get("Playing Time_Min", 0) or 0)
            n90 = float(r.get("Playing Time_90s", 0) or 0)
            profiles[player] = {
                "team": team,
                "minutes": mins,
                "starts": float(r.get("Starts_Starts", 0) or 0),
                "minutes_per_start": float(r.get("Starts_Mn/Start", 0) or 0),
                "ninety_equivalents": n90,
            }
            if mins > max_minutes:
                max_minutes = mins

        for _, r in std.iterrows():
            player = r[PLAYER_COL]
            if player not in profiles:
                continue
            profiles[player].update(
                {
                    "age": r.get("Unnamed: 3_level_0_Age"),
                    "position": r.get("Unnamed: 2_level_0_Pos"),
                }
            )

        for p, v in profiles.items():
            mins = v.get("minutes", 0.0)
            v["workload_ratio"] = mins / max_minutes if max_minutes > 0 else 0.0

        # filter low sample players
        profiles = {
            p: v for p, v in profiles.items() if v.get("minutes", 0.0) >= 300
        }

        return json.dumps(
            {
                "team": team,
                "players": profiles,
            },
            indent=2,
        )

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ---------------------------------------------------------------------
# TOOL: MATCH PLANNER (WORKLOAD-AWARE XI)
# ---------------------------------------------------------------------
class MatchPlannerTool(BaseTool):
    name = "match_planner"
    description = (
        "Plans a workload-aware starting XI using FBref CSV data. "
        "Input is a natural language question (team & opponent inferred)."
    )

    def _run(self, query: str) -> str:
        # ------------------------------------------------------------
        # 1) Detect teams & formation intent
        # ------------------------------------------------------------
        info = detect_team_and_opponent(query)
        team = info["team"]
        opponent = info["opponent"]
        formation_hint = detect_formation(query)

        base = get_team_path(team)

        # ------------------------------------------------------------
        # 2) Load & clean data
        # ------------------------------------------------------------
        std = clean_player_df(pd.read_csv(base / "stats_standard_9.csv"))
        play = clean_player_df(pd.read_csv(base / "stats_playing_time_9.csv"))
        poss = clean_player_df(pd.read_csv(base / "stats_possession_9.csv"))
        dfn = clean_player_df(pd.read_csv(base / "stats_defense_9.csv"))

        # ------------------------------------------------------------
        # 3) Build unified player profiles
        # ------------------------------------------------------------
        players: Dict[str, Dict[str, float]] = {}

        for _, r in std.iterrows():
            p = r[PLAYER_COL]
            players[p] = {
                "pos": r["Unnamed: 2_level_0_Pos"],
                "goals": float(r["Performance_Gls"] or 0),
                "xg": float(r["Expected_xG"] or 0),
                "xag": float(r.get("Expected_xAG", 0) or 0),
            }

        for _, r in play.iterrows():
            p = r[PLAYER_COL]
            if p in players:
                players[p].update({
                    "minutes": float(r["Playing Time_Min"] or 0),
                    "starts": float(r["Starts_Starts"] or 0),
                })

        for _, r in poss.iterrows():
            p = r[PLAYER_COL]
            if p in players:
                players[p].update({
                    "touches": float(r["Touches_Touches"] or 0),
                    "prog_carries": float(r["Carries_PrgC"] or 0),
                })

        for _, r in dfn.iterrows():
            p = r[PLAYER_COL]
            if p in players:
                players[p].update({
                    "tackles": float(r["Tackles_Tkl"] or 0),
                    "interceptions": float(r["Unnamed: 17_level_0_Int"] or 0),
                })

        # ------------------------------------------------------------
        # 4) Filter match-ready players
        # ------------------------------------------------------------
        players = {
            p: v for p, v in players.items()
            if v.get("minutes", 0) >= 300
        }

        max_minutes = max(v["minutes"] for v in players.values())

        # ------------------------------------------------------------
        # 5) Formation selection
        # ------------------------------------------------------------
        avg_load = sum(v["minutes"] for v in players.values()) / len(players)

        if formation_hint != "auto":
            formation = formation_hint
            formation_reason = f"Formation explicitly requested: {formation_hint}"
        else:
            if avg_load > 800:
                formation = "4-3-3"
                formation_reason = "Balanced usage suggests 4-3-3 with rotation support"
            else:
                formation = "4-2-3-1"
                formation_reason = "Allows workload management across attacking lines"

        # ------------------------------------------------------------
        # 6) Scoring logic
        # ------------------------------------------------------------
        def workload_penalty(p):
            return (p["minutes"] / max_minutes) * 1.5

        def score(p, weights):
            return (
                sum(p.get(k, 0) * w for k, w in weights.items())
                - workload_penalty(p)
            )

        role_defs = {
            "GK": ("GK", {"minutes": 1}),
            "RB": ("DF", {"touches": 0.6, "prog_carries": 0.4}),
            "CB1": ("DF", {"tackles": 0.6, "interceptions": 0.4}),
            "CB2": ("DF", {"tackles": 0.6, "interceptions": 0.4}),
            "LB": ("DF", {"touches": 0.6, "prog_carries": 0.4}),
            "CM1": ("MF", {"touches": 0.5, "prog_carries": 0.5}),
            "CM2": ("MF", {"touches": 0.5, "prog_carries": 0.5}),
            "AM": ("MF", {"xg": 0.6, "xag": 0.4}),
            "RW": ("FW", {"xg": 0.4, "prog_carries": 0.6}),
            "LW": ("FW", {"xg": 0.4, "prog_carries": 0.6}),
            "ST": ("FW", {"goals": 0.7, "xg": 0.3}),
        }

        used_players = set()
        xi = {}
        decisions = {}

        # ------------------------------------------------------------
        # 7) Select XI (NO DUPLICATES)
        # ------------------------------------------------------------
        for role, (pos_tag, weights) in role_defs.items():
            pool = {
                p: score(v, weights)
                for p, v in players.items()
                if pos_tag in v["pos"] and p not in used_players
            }

            if not pool:
                xi[role] = None
                continue

            ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)
            chosen, chosen_score = ranked[0]
            used_players.add(chosen)

            decisions[role] = {
                "selected": chosen,
                "score": round(chosen_score, 2),
                "workload_minutes": players[chosen]["minutes"],
                "key_metrics": {k: players[chosen].get(k, 0) for k in weights},
                "alternatives": {
                    p: round(s, 2) for p, s in ranked[1:3]
                }
            }

            xi[role] = chosen

        # ------------------------------------------------------------
        # 8) Return explainable output
        # ------------------------------------------------------------
        return json.dumps({
            "team": team,
            "opponent": opponent,
            "formation": formation,
            "formation_reason": formation_reason,
            "workload_aware": True,
            "starting_xi": xi,
            "decisions": decisions,
        }, indent=2)

    async def _arun(self, query: str) -> str:
        return self._run(query)



# ---------------------------------------------------------------------
# SYSTEM PROMPT & LLM
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a professional football performance and tactics analyst.

You MUST:
- Base conclusions strictly on the tool outputs (CSV-derived data).
- Never invent players or stats that are not in the data.
- Consider workload_ratio when judging fatigue/injury risk.
- Prefer players with high contribution per 90 but not extreme workload.
- Explain your reasoning when asked, referencing data points.

You MAY:
- Recommend formations like 4-2-3-1, 4-3-3, etc., but if the user or tool
  specifies one, respect it.

DO NOT:
- Use outside football knowledge (no real-world transfers/injuries).
- Mention players who are not in the club CSV files.
"""


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    system_message=SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------
tools = [
    ClubProfilesTool(),
    WorkloadProfilesTool(),
    MatchPlannerTool(),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    queries = [
        "As a football performance analyst, suggest the best starting XI for Spurs against Brentford.Specify the formation, justify each player’s selection using performance and workload data, and explain why other viable options were not selected",
        # you can add more:
        # "Which Brentford players are most overloaded by minutes?",
        # "Give me a workload-aware XI for Brentford vs Spurs in a 4-3-3",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print("QUESTION:", q)
        answer = agent.run(q)
        print("\nANSWER:\n", answer)
