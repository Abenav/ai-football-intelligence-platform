import os
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "API_KEY"

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "fbref" / "premier_league_2025_26"
VALID_TEAMS = {"spurs", "brentford"}
PLAYER_COL = "Unnamed: 0_level_0_Player"

# ---------------------------------------------------------------------
# SHARED UTILITIES
# ---------------------------------------------------------------------
def detect_team_from_query(query: str) -> str:
    """Extract team name from query."""
    q = query.lower()
    if any(w in q for w in ["spurs", "tottenham"]):
        return "spurs"
    if "brentford" in q:
        return "brentford"
    return "spurs"  # default

def get_team_path(team_key: str) -> Path:
    """Get the data folder path for a team."""
    if team_key.lower() not in VALID_TEAMS:
        raise ValueError(f"Unknown team '{team_key}'. Known: {VALID_TEAMS}")
    path = DATA_ROOT / team_key.lower()
    if not path.exists():
        raise FileNotFoundError(f"Data folder not found at {path}")
    return path

def clean_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove header and total rows."""
    if PLAYER_COL not in df.columns:
        return df
    junk = {"Player", "Squad Total", "Opponent Total"}
    df = df[~df[PLAYER_COL].isin(junk)]
    return df.dropna(subset=[PLAYER_COL]).reset_index(drop=True)

# Add this helper near the top of the file (after imports)
def _extract_query_from_call(*args, **kwargs) -> str:
    """
    LangChain may call tool._run with:
      - a single positional argument (the input string), OR
      - keyword args like {'query': '...', 'team_name': '...'}
    This helper normalizes both cases into a single query string.
    """
    # If agent passed a positional arg (common), use it
    if args:
        # If the agent passed a dict-like structure, try to convert to string
        first = args[0]
        # sometimes it's a dict/JSON-like; prefer 'query' or 'team_name' if present
        if isinstance(first, dict):
            return first.get("query") or first.get("team_name") or str(first)
        return str(first)

    # Otherwise fallback to known keywords or a stringified kwargs
    return kwargs.get("query") or kwargs.get("team_name") or str(kwargs) or ""

# ---------------------------------------------------------------------
# TOOL 1: DATA LOADER
# ---------------------------------------------------------------------
class LoadTeamDataTool(BaseTool):
    name = "load_team_data"
    description = (
        "Loads raw CSV data for a team (Spurs or Brentford). "
        "Returns available data files and basic team info. "
        "Input should be a natural language query mentioning the team name."
    )

    def _run(self, *args, **kwargs) -> str:
        # Handle both direct string input and keyword arguments
        query = kwargs.get('query') or kwargs.get('team_name') or str(kwargs)
        team = detect_team_from_query(query)
        team_path = get_team_path(team)
        
        csv_files = list(team_path.glob("*.csv"))
        
        return json.dumps({
            "team": team,
            "data_path": str(team_path),
            "available_files": [f.name for f in csv_files],
            "total_files": len(csv_files)
        }, indent=2)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# ---------------------------------------------------------------------
# TOOL 2: PLAYER PERFORMANCE PROFILES
# ---------------------------------------------------------------------
class PlayerPerformanceTool(BaseTool):
    name = "player_performance"
    description = (
        "Analyzes individual player performance metrics including goals, assists, "
        "xG, xAG, progressive actions, touches, and per-90 stats. "
        "Input should be a natural language query mentioning the team name."
    )

    def _run(self, *args, **kwargs) -> str:
        query = kwargs.get('query') or kwargs.get('team_name') or str(kwargs)
        team = detect_team_from_query(query)
        team_path = get_team_path(team)
        
        # Load data
        std = clean_player_df(pd.read_csv(team_path / "stats_standard_9.csv"))
        pos = clean_player_df(pd.read_csv(team_path / "stats_possession_9.csv"))
        play = clean_player_df(pd.read_csv(team_path / "stats_playing_time_9.csv"))
        
        profiles = {}
        
        for _, r in std.iterrows():
            player = r[PLAYER_COL]
            profiles[player] = {
                "position": r["Unnamed: 2_level_0_Pos"],
                "age": r["Unnamed: 3_level_0_Age"],
                "goals": float(r.get("Performance_Gls", 0) or 0),
                "assists": float(r.get("Performance_Ast", 0) or 0),
                "xg": float(r.get("Expected_xG", 0) or 0),
                "xag": float(r.get("Expected_xAG", 0) or 0),
                "prog_carries": float(r.get("Progression_PrgC", 0) or 0),
                "prog_passes": float(r.get("Progression_PrgP", 0) or 0),
            }
        
        # Add possession stats
        for _, r in pos.iterrows():
            player = r[PLAYER_COL]
            if player in profiles:
                profiles[player].update({
                    "touches": float(r.get("Touches_Touches", 0) or 0),
                    "progressive_carries": float(r.get("Carries_PrgC", 0) or 0),
                    "progressive_receives": float(r.get("Receiving_PrgR", 0) or 0),
                })
        
        # Add playing time and calculate per-90 stats
        for _, r in play.iterrows():
            player = r[PLAYER_COL]
            if player in profiles:
                mins = float(r.get("Playing Time_Min", 0) or 0)
                n90 = float(r.get("Playing Time_90s", 0) or 0)
                profiles[player]["minutes"] = mins
                profiles[player]["ninety_equivalents"] = n90
                
                if n90 > 0:
                    prof = profiles[player]
                    prof["goals_per90"] = prof["goals"] / n90
                    prof["assists_per90"] = prof["assists"] / n90
                    prof["attack_contribution_per90"] = (prof["goals"] + prof["assists"]) / n90
        
        # Filter low-sample players
        profiles = {p: v for p, v in profiles.items() if v.get("minutes", 0) >= 200}
        
        return json.dumps({
            "team": team,
            "total_players": len(profiles),
            "players": profiles
        }, indent=2)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# ---------------------------------------------------------------------
# TOOL 3: WORKLOAD ANALYSIS
# ---------------------------------------------------------------------
class WorkloadAnalysisTool(BaseTool):
    name = "workload_analysis"
    description = (
        "Analyzes player workload including minutes played, starts, "
        "workload ratio (relative to most-used player), and fatigue risk indicators. "
        "Helps identify overloaded or fresh players. Input should be a natural language query."
    )

    def _run(self, *args, **kwargs) -> str:
        query = kwargs.get('query') or kwargs.get('team_name') or str(kwargs)
        team = detect_team_from_query(query)
        team_path = get_team_path(team)
        
        play = clean_player_df(pd.read_csv(team_path / "stats_playing_time_9.csv"))
        std = clean_player_df(pd.read_csv(team_path / "stats_standard_9.csv"))
        
        profiles = {}
        max_minutes = 0.0
        
        for _, r in play.iterrows():
            player = r[PLAYER_COL]
            mins = float(r.get("Playing Time_Min", 0) or 0)
            profiles[player] = {
                "minutes": mins,
                "starts": float(r.get("Starts_Starts", 0) or 0),
                "minutes_per_start": float(r.get("Starts_Mn/Start", 0) or 0),
                "ninety_equivalents": float(r.get("Playing Time_90s", 0) or 0),
            }
            max_minutes = max(max_minutes, mins)
        
        # Add position and age
        for _, r in std.iterrows():
            player = r[PLAYER_COL]
            if player in profiles:
                profiles[player]["position"] = r.get("Unnamed: 2_level_0_Pos")
                profiles[player]["age"] = r.get("Unnamed: 3_level_0_Age")
        
        # Calculate workload metrics
        for p, v in profiles.items():
            v["workload_ratio"] = v["minutes"] / max_minutes if max_minutes > 0 else 0
            
            # Categorize workload risk
            if v["workload_ratio"] > 0.85:
                v["workload_category"] = "High Risk - Heavy Load"
            elif v["workload_ratio"] > 0.65:
                v["workload_category"] = "Moderate Load"
            else:
                v["workload_category"] = "Fresh - Low Load"
        
        # Filter and sort by workload
        profiles = {p: v for p, v in profiles.items() if v["minutes"] >= 300}
        sorted_players = sorted(profiles.items(), key=lambda x: x[1]["workload_ratio"], reverse=True)
        
        return json.dumps({
            "team": team,
            "max_minutes_player": max(profiles.items(), key=lambda x: x[1]["minutes"])[0] if profiles else None,
            "max_minutes": max_minutes,
            "players": dict(sorted_players)
        }, indent=2)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# ---------------------------------------------------------------------
# TOOL 4: DEFENSIVE ANALYSIS
# ---------------------------------------------------------------------
class DefensiveAnalysisTool(BaseTool):
    name = "defensive_analysis"
    description = (
        "Analyzes defensive performance including tackles, interceptions, "
        "blocks, errors, and defensive actions per 90. "
        "Input should be a natural language query mentioning the team."
    )

    def _run(self, *args, **kwargs) -> str:
        query = kwargs.get('query') or kwargs.get('team_name') or str(kwargs)
        team = detect_team_from_query(query)
        team_path = get_team_path(team)
        
        dfn = clean_player_df(pd.read_csv(team_path / "stats_defense_9.csv"))
        play = clean_player_df(pd.read_csv(team_path / "stats_playing_time_9.csv"))
        std = clean_player_df(pd.read_csv(team_path / "stats_standard_9.csv"))
        
        profiles = {}
        
        for _, r in dfn.iterrows():
            player = r[PLAYER_COL]
            profiles[player] = {
                "tackles": float(r.get("Tackles_Tkl", 0) or 0),
                "tackles_won": float(r.get("Tackles_TklW", 0) or 0),
                "interceptions": float(r.get("Unnamed: 17_level_0_Int", 0) or 0),
                "blocks": float(r.get("Blocks_Blocks", 0) or 0),
                "errors": float(r.get("Unnamed: 20_level_0_Err", 0) or 0),
            }
        
        # Add position
        for _, r in std.iterrows():
            player = r[PLAYER_COL]
            if player in profiles:
                profiles[player]["position"] = r.get("Unnamed: 2_level_0_Pos")
        
        # Add per-90 stats
        for _, r in play.iterrows():
            player = r[PLAYER_COL]
            if player in profiles:
                n90 = float(r.get("Playing Time_90s", 0) or 0)
                if n90 > 0:
                    prof = profiles[player]
                    prof["tackles_per90"] = prof["tackles"] / n90
                    prof["interceptions_per90"] = prof["interceptions"] / n90
                    prof["defensive_actions_per90"] = (prof["tackles"] + prof["interceptions"]) / n90
                    prof["ninety_equivalents"] = n90
        
        # Filter and focus on defenders
        profiles = {
            p: v for p, v in profiles.items() 
            if v.get("ninety_equivalents", 0) >= 3 and 
            ("DF" in str(v.get("position", "")) or "MF" in str(v.get("position", "")))
        }
        
        return json.dumps({
            "team": team,
            "players_analyzed": len(profiles),
            "players": profiles
        }, indent=2)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# ---------------------------------------------------------------------
# TOOL 5: FORMATION SUGGESTER
# ---------------------------------------------------------------------
class FormationSuggestionTool(BaseTool):
    name = "formation_suggestion"
    description = (
        "Suggests optimal formation based on team squad composition, "
        "player workload, and tactical considerations. "
        "Can also evaluate a specific formation if mentioned in query. "
        "Input should be a natural language query mentioning the team."
    )

    def _run(self, *args, **kwargs) -> str:
        query = kwargs.get('query') or kwargs.get('team_name') or str(kwargs)
        team = detect_team_from_query(query)
        team_path = get_team_path(team)
        
        # Check if specific formation mentioned
        q = str(query).replace(" ", "")
        requested_formation = None
        for f in ["4-2-3-1", "4-3-3", "3-4-3", "4-4-2", "3-5-2"]:
            if f.replace("-", "") in q:
                requested_formation = f
                break
        
        # Load data for analysis
        std = clean_player_df(pd.read_csv(team_path / "stats_standard_9.csv"))
        play = clean_player_df(pd.read_csv(team_path / "stats_playing_time_9.csv"))
        
        # Count available players by position
        position_counts = {}
        avg_workload = 0
        total_players = 0
        
        for _, r in std.iterrows():
            pos = r.get("Unnamed: 2_level_0_Pos", "")
            for p in ["GK", "DF", "MF", "FW"]:
                if p in str(pos):
                    position_counts[p] = position_counts.get(p, 0) + 1
        
        for _, r in play.iterrows():
            mins = float(r.get("Playing Time_Min", 0) or 0)
            if mins >= 300:
                avg_workload += mins
                total_players += 1
        
        avg_workload = avg_workload / total_players if total_players > 0 else 0
        
        # Formation logic
        if requested_formation:
            suggestion = requested_formation
            reason = f"Formation {requested_formation} was explicitly requested"
        else:
            # High workload = need rotation-friendly formation
            if avg_workload > 800:
                suggestion = "4-3-3"
                reason = "High average workload suggests 4-3-3 for better rotation options"
            elif position_counts.get("MF", 0) > 6:
                suggestion = "4-3-3"
                reason = "Strong midfield depth favors 4-3-3"
            elif position_counts.get("FW", 0) >= 4:
                suggestion = "4-2-3-1"
                reason = "Multiple attacking options suit 4-2-3-1 flexibility"
            else:
                suggestion = "4-2-3-1"
                reason = "Balanced formation for workload management"
        
        return json.dumps({
            "team": team,
            "suggested_formation": suggestion,
            "reason": reason,
            "squad_composition": position_counts,
            "average_workload_minutes": round(avg_workload, 1),
            "alternative_formations": ["4-3-3", "4-2-3-1", "3-4-3"]
        }, indent=2)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# ---------------------------------------------------------------------
# TOOL 6: STARTING XI SELECTOR
# ---------------------------------------------------------------------
class StartingXISelectorTool(BaseTool):
    name = "starting_xi_selector"
    description = (
        "Selects the optimal starting XI based on performance metrics, "
        "workload considerations, and formation requirements. "
        "Provides detailed justification for each selection. "
        "Input should be a natural language query with team, opponent, and/or formation."
    )

    def _run(self, *args, **kwargs) -> str:
        query = kwargs.get('query') or str(kwargs)
        team = detect_team_from_query(query)
        
        # Detect opponent
        q = str(query).lower()
        opponent = "brentford" if team == "spurs" else "spurs"
        if "against" in q or "vs" in q:
            if "brentford" in q and team != "brentford":
                opponent = "brentford"
            elif any(w in q for w in ["spurs", "tottenham"]) and team != "spurs":
                opponent = "spurs"
        
        team_path = get_team_path(team)
        
        # Load all data
        std = clean_player_df(pd.read_csv(team_path / "stats_standard_9.csv"))
        play = clean_player_df(pd.read_csv(team_path / "stats_playing_time_9.csv"))
        poss = clean_player_df(pd.read_csv(team_path / "stats_possession_9.csv"))
        dfn = clean_player_df(pd.read_csv(team_path / "stats_defense_9.csv"))
        
        # Build comprehensive player profiles
        players = {}
        
        for _, r in std.iterrows():
            p = r[PLAYER_COL]
            players[p] = {
                "pos": r["Unnamed: 2_level_0_Pos"],
                "goals": float(r["Performance_Gls"] or 0),
                "assists": float(r["Performance_Ast"] or 0),
                "xg": float(r["Expected_xG"] or 0),
                "xag": float(r.get("Expected_xAG", 0) or 0),
            }
        
        for _, r in play.iterrows():
            p = r[PLAYER_COL]
            if p in players:
                players[p]["minutes"] = float(r["Playing Time_Min"] or 0)
                players[p]["starts"] = float(r["Starts_Starts"] or 0)
        
        for _, r in poss.iterrows():
            p = r[PLAYER_COL]
            if p in players:
                players[p]["prog_carries"] = float(r["Carries_PrgC"] or 0)
                players[p]["touches"] = float(r["Touches_Touches"] or 0)
        
        for _, r in dfn.iterrows():
            p = r[PLAYER_COL]
            if p in players:
                players[p]["tackles"] = float(r["Tackles_Tkl"] or 0)
                players[p]["interceptions"] = float(r["Unnamed: 17_level_0_Int"] or 0)
        
        # Filter match-ready players
        players = {p: v for p, v in players.items() if v.get("minutes", 0) >= 300}
        max_minutes = max(v["minutes"] for v in players.values()) if players else 1
        
        # Scoring function
        def score_player(p, weights):
            workload_penalty = (p["minutes"] / max_minutes) * 1.5
            return sum(p.get(k, 0) * w for k, w in weights.items()) - workload_penalty
        
        # Role definitions for 4-2-3-1
        roles = {
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
        
        used = set()
        xi = {}
        decisions = {}
        
        for role, (pos_tag, weights) in roles.items():
            pool = {
                p: score_player(v, weights)
                for p, v in players.items()
                if pos_tag in v["pos"] and p not in used
            }
            
            if not pool:
                continue
            
            ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)
            chosen, score = ranked[0]
            used.add(chosen)
            xi[role] = chosen
            
            decisions[role] = {
                "selected": chosen,
                "score": round(score, 2),
                "minutes": players[chosen]["minutes"],
                "workload_ratio": round(players[chosen]["minutes"] / max_minutes, 2),
                "key_stats": {k: players[chosen].get(k, 0) for k in weights},
                "alternatives": [(p, round(s, 2)) for p, s in ranked[1:3]]
            }
        
        return json.dumps({
            "team": team,
            "opponent": opponent,
            "formation": "4-2-3-1",
            "starting_xi": xi,
            "detailed_decisions": decisions
        }, indent=2)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# ---------------------------------------------------------------------
# SYSTEM PROMPT & AGENT
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a professional football performance and tactics analyst.

INSTRUCTIONS:
- Use the specialized tools to gather data and analysis
- Base all conclusions strictly on tool outputs
- Never invent players or stats not in the data
- Consider workload_ratio for fatigue/injury risk assessment
- Explain reasoning clearly, referencing specific data points
- When suggesting XI, explain why each player was chosen over alternatives

AVAILABLE TOOLS:
1. load_team_data - Check available data files
2. player_performance - Analyze attacking/creative metrics
3. workload_analysis - Assess player fatigue and load
4. defensive_analysis - Review defensive contributions
5. formation_suggestion - Get tactical formation recommendations
6. starting_xi_selector - Build complete starting lineup

Always break down complex queries into multiple tool calls for comprehensive analysis.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    system_message=SYSTEM_PROMPT,
)

tools = [
    LoadTeamDataTool(),
    PlayerPerformanceTool(),
    WorkloadAnalysisTool(),
    DefensiveAnalysisTool(),
    FormationSuggestionTool(),
    StartingXISelectorTool(),
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
    query = """As a football performance analyst, suggest the best starting XI for Spurs 
    against Brentford. Specify the formation, justify each player's selection using 
    performance and workload data, and explain why other viable options were not selected."""
    
    print("=" * 80)
    print("QUESTION:", query)
    print("=" * 80)
    answer = agent.run(query)
    print("\n" + "=" * 80)
    print("ANSWER:\n", answer)