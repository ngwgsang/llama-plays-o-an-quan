from typing import Dict, List, Any
from utils import log_action_events


class Environment:
    """
    Game environment for the traditional Vietnamese board game Ô ăn quan.

    This environment maintains:
    - Board state
    - Player scores
    - Turn progression
    - Rules for scattering, capturing, and restoring peasants

    The design is stateful and suitable for:
    - Rule-based agents
    - LLM-based agents (reasoning + acting)
    - Reinforcement learning experiments
    """

    def __init__(self):
        # Mapping each player to their five peasant positions
        self.players_map = {
            "A": [f"A{i}" for i in range(1, 6)],
            "B": [f"B{i}" for i in range(1, 6)],
        }
        self.reset()

    def reset(self):
        """
        Reset the game to the initial state.

        Initial setup:
        - Each peasant cell contains 5 peasants
        - Each mandarin cell contains 1 mandarin
        - Both players start with 0 score
        """
        self.game_state = {
            "board": {
                "QA": ["mandarin_a"],
                "A1": ["peasant_a"] * 5,
                "A2": ["peasant_a"] * 5,
                "A3": ["peasant_a"] * 5,
                "A4": ["peasant_a"] * 5,
                "A5": ["peasant_a"] * 5,
                "QB": ["mandarin_b"],
                "B1": ["peasant_b"] * 5,
                "B2": ["peasant_b"] * 5,
                "B3": ["peasant_b"] * 5,
                "B4": ["peasant_b"] * 5,
                "B5": ["peasant_b"] * 5,
            },
            "score": {"A": 0, "B": 0},
            "round": 0,
        }

    def get_game_state(self) -> Dict[str, Any]:
        """
        Return the full current game state.

        This snapshot is typically used for:
        - Agent reasoning
        - UI rendering
        - Debugging
        """
        return self.game_state

    def get_available_pos(self, player_team: str) -> List[str]:
        """
        Return all valid positions that the given player can play.

        A valid position:
        - Belongs to the player
        - Is not a mandarin cell
        - Contains at least one piece
        """
        board = self.game_state["board"]
        return [
            pos
            for pos in self.players_map[player_team]
            if board.get(pos) and not pos.startswith("Q")
        ]

    def restore_peasants(self, player_team: str) -> tuple[bool, str]:
        """
        Restore peasants when a player has no movable pieces.

        Rule:
        - If all five peasant cells are empty:
            - If score >= 5: subtract 5 points and add 1 peasant to each cell
            - Otherwise: the player loses immediately

        Returns:
        - can_continue: whether the game can continue
        - message: descriptive game log
        """
        score = self.game_state["score"]
        board = self.game_state["board"]
        message = ""
        can_continue = True

        if all(not board.get(pos) for pos in self.players_map[player_team]):
            if score[player_team] >= 10:
                score[player_team] -= 10
                for pos in self.players_map[player_team]:
                    board[pos].append(f"peasant_{player_team.lower()}")
                message = f"[RESTORE] Player {player_team} restored 5 peasants."
            else:
                message = (
                    f"[END] Player {player_team} does not have enough score to continue. LOSS."
                )
                can_continue = False

        return can_continue, message

    def is_end(self) -> tuple[bool, str]:
        """
        Check whether the current game has ended.

        End conditions:
        1. Both mandarin pits are empty
        2. Any player reaches the score threshold (>= 25)
        3. A player has no movable peasants and cannot restore

        Returns:
        - is_end (bool): whether the game ends
        - reason (str): textual description of the termination condition
        """
        board = self.game_state["board"]
        score = self.game_state["score"]

        # --- Condition 1: both mandarins are gone ---
        no_mandarin_A = not any(t.startswith("mandarin") for t in board["QA"])
        no_mandarin_B = not any(t.startswith("mandarin") for t in board["QB"])

        if no_mandarin_A and no_mandarin_B:
            return True, "END_BY_NO_MANDARINS"

        # --- Condition 2: score threshold reached ---
        for player in ["A", "B"]:
            if score[player] >= 25:
                return True, f"END_BY_SCORE_THRESHOLD_{player}"

        # --- Condition 3: cannot restore peasants ---
        for player in ["A", "B"]:
            no_peasants = all(
                not board.get(pos)
                for pos in self.players_map[player]
            )

            cannot_restore = score[player] < 10  # restore cost

            if no_peasants and cannot_restore:
                return True, f"END_BY_NO_RESTORE_{player}"

        return False, ""


    def commit_action(self, action: Dict[str, Any]) -> tuple[list, list, bool]:
        """
        Execute a single move in the game.

        Action format:
        {
            "pos": <starting position>,
            "way": "clockwise" or "counter_clockwise"
        }

        Returns:
        - steps: textual description of logical steps
        - animation_events: structured events for UI animation
        - is_end: whether the game has ended
        """
        pos, way = action.get("pos"), action.get("way")
        board_data = self.game_state["board"]
        score_data = self.game_state["score"]

        steps = []
        animation_events = []

        # Circular order of positions on the board
        order = [
            "QA", "A1", "A2", "A3", "A4", "A5",
            "QB", "B5", "B4", "B3", "B2", "B1",
        ]

        # Validate the action
        if not pos or not way or pos not in order or not board_data.get(pos):
            return [f"[error] Invalid move: {pos}"], [], False

        # Work on copies to avoid partial state corruption
        board = {k: v.copy() for k, v in board_data.items()}
        score = score_data.copy()

        # Only peasants can be scattered, mandarins stay in place
        tokens = [t for t in board[pos] if not t.startswith("mandarin")]
        if not tokens:
            return [f"[error] No peasants to scatter from {pos}."], [], False

        # Pick up peasants from the starting cell
        animation_events.append({"type": "pickup", "pos": pos, "pieces": tokens})
        board[pos] = [t for t in board[pos] if t.startswith("mandarin")]

        index = order.index(pos)
        direction = 1 if way == "clockwise" else -1
        steps.append(f"[scatter] {pos} - {way.replace('_', ' ')}")

        # Scatter peasants one by one
        current_pos_for_animation = pos
        for i, token in enumerate(tokens):
            target_index = (index + direction * (i + 1)) % len(order)
            target_pos = order[target_index]
            board[target_pos].append(token)

            animation_events.append({
                "type": "drop",
                "from_pos": current_pos_for_animation,
                "to_pos": target_pos,
                "piece": token,
            })
            current_pos_for_animation = target_pos

        current_index = (index + direction * len(tokens)) % len(order)

        # Repeated capture / scatter loop
        loop_count = 0
        while loop_count < 999:
            loop_count += 1
            next_index = (current_index + direction) % len(order)
            next_pos = order[next_index]

            # Case 1: next cell is empty -> attempt capture
            if not board[next_pos]:
                next_next_index = (next_index + direction) % len(order)
                next_next_pos = order[next_next_index]

                if not board.get(next_next_pos) or not board[next_next_pos]:
                    break

                captured_pieces = board[next_next_pos]

                # --- Rule: forbid capturing immature mandarin pit ---
                if next_next_pos in ["QA", "QB"] and len(captured_pieces) <= 5:
                    animation_events.append({
                        "type": "forbidden_capture",
                        "pos": next_next_pos,
                        "reason": "MANDARIN_TOTAL_LE_5",
                        "pieces": captured_pieces.copy(),
                    })
                    break

                captured_team = "A" if captured_pieces[0].endswith("_a") else "B"

                animation_events.append({
                    "type": "capture",
                    "pos": next_next_pos,
                    "team": captured_team,
                    "pieces": captured_pieces,
                })


                # Update score: peasant = 1, mandarin = 10
                for token in captured_pieces:
                    value = 10 if token.startswith("mandarin") else 1
                    score["A" if token.endswith("_a") else "B"] += value

                board[next_next_pos] = []
                animation_events.append({
                    "type": "score_update",
                    "score": score.copy(),
                })

                current_index = next_next_index

            # Case 2: next cell has peasants -> scatter again
            else:
                tokens_to_scatter = [
                    t for t in board[next_pos] if not t.startswith("mandarin")
                ]
                if not tokens_to_scatter:
                    break

                animation_events.append({
                    "type": "pickup",
                    "pos": next_pos,
                    "pieces": tokens_to_scatter,
                })

                board[next_pos] = [
                    t for t in board[next_pos] if t.startswith("mandarin")
                ]

                index = next_index
                current_pos_for_animation = next_pos
                for i, token in enumerate(tokens_to_scatter):
                    target_index = (index + direction * (i + 1)) % len(order)
                    target_pos = order[target_index]
                    board[target_pos].append(token)

                    animation_events.append({
                        "type": "drop",
                        "from_pos": current_pos_for_animation,
                        "to_pos": target_pos,
                        "piece": token,
                    })
                    current_pos_for_animation = target_pos

                current_index = (index + direction * len(tokens_to_scatter)) % len(order)

        # Commit updated state
        self.game_state["board"] = board
        self.game_state["score"] = score

        # Game ends when both mandarins have been captured
        is_end, end_reason = self.is_end()

        if is_end:
            animation_events.append({
                "type": "end_game",
                "reason": end_reason,
                "final_score": self.game_state["score"].copy(),
            })


        if animation_events:
            log_action_events(
                title=f"Env :: Commit Action ({action['pos']} | {action['way']})",
                animation_events=animation_events,
                border_style="orange1",
            )

        return steps, animation_events, is_end

