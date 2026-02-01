from typing import Dict, List, Any

class Enviroment:
    def __init__(self):
        self.players_map = {"A": [f"A{i}" for i in range(1, 6)], "B": [f"B{i}" for i in range(1, 6)]}
        self.reset()

    def reset(self):
        self.game_state = {
            "board": {
                "QA": ["mandarin_a"], "A1": ["peasant_a"] * 5, "A2": ["peasant_a"] * 5,
                "A3": ["peasant_a"] * 5, "A4": ["peasant_a"] * 5, "A5": ["peasant_a"] * 5,
                "QB": ["mandarin_b"], "B1": ["peasant_b"] * 5, "B2": ["peasant_b"] * 5,
                "B3": ["peasant_b"] * 5, "B4": ["peasant_b"] * 5, "B5": ["peasant_b"] * 5,
            },
            "score": {"A": 0, "B": 0},
            "round": 0
        }
        # print(self.game_state)

    def get_game_state(self) -> Dict[str, Any]:
        return self.game_state

    def get_available_pos(self, player_team: str) -> List[str]:
        board = self.game_state["board"]
        return [pos for pos in self.players_map[player_team] if board.get(pos) and not pos.startswith("Q")]

    def restore_peasants(self, player_team: str) -> tuple[bool, str]:
        score = self.game_state["score"]
        board = self.game_state["board"]
        message = ""
        can_continue = True
        
        if all(not board.get(pos) for pos in self.players_map[player_team]):
            if score[player_team] >= 5:
                score[player_team] -= 5
                for pos in self.players_map[player_team]:
                    board[pos].append(f"peasant_{player_team.lower()}")
                message = f"[RESTORE] Player {player_team} restored 5 peasants."
            else:
                message = f"[END] Player {player_team} does not have enough score to continue. LOSS."
                can_continue = False
        return can_continue, message

    def commit_action(self, action: Dict[str, Any], extended_rules: List[str] | None = None) -> tuple[list, list, bool]:
        extended_rules = extended_rules or ["E1", "E2", "E3", "E4", "E5"]
        apply_e1 = "E1" in extended_rules
        apply_e2 = "E2" in extended_rules

        pos, way = action.get("pos"), action.get("way")
        board_data, score_data, round_idx = self.game_state["board"], self.game_state["score"], self.game_state["round"]
        
        steps = []
        animation_events = []
        order = ["QA", "A1", "A2", "A3", "A4", "A5", "QB", "B5", "B4", "B3", "B2", "B1"]
        
        if not pos or not way or pos not in order or not board_data.get(pos):
            return [f"[error] Invalid move: {pos}"], [], False

        board, score = {k: v.copy() for k, v in board_data.items()}, score_data.copy()
        tokens = [t for t in board[pos] if not t.startswith("mandarin")]
        
        if not tokens: return [f"[error] No peasants to scatter from {pos}."], [], False

        animation_events.append({'type': 'pickup', 'pos': pos, 'pieces': tokens})
        board[pos] = [t for t in board[pos] if t.startswith("mandarin")]

        index, direction = order.index(pos), 1 if way == "clockwise" else -1
        steps.append(f"[scatter] {pos} - {way.replace('_', ' ')}")

        current_pos_for_animation = pos
        for i, token in enumerate(tokens):
            target_index = (index + direction * (i + 1)) % len(order)
            target_pos = order[target_index]
            board[target_pos].append(token)
            animation_events.append({'type': 'drop', 'from_pos': current_pos_for_animation, 'to_pos': target_pos, 'piece': token})
            current_pos_for_animation = target_pos

        current_index = (index + direction * len(tokens)) % len(order)
        
        loop_count = 0
        while loop_count < 100:
            loop_count += 1
            next_index = (current_index + direction) % len(order)
            next_pos = order[next_index]

            if not board[next_pos]:
                next_next_index = (next_index + direction) % len(order)
                next_next_pos = order[next_next_index]
                
                if not board.get(next_next_pos): break
                
                if next_next_pos.startswith("Q"):
                    if apply_e1:
                        if len(board[next_next_pos]) < 5: break
                        if round_idx < 3: break
                
                if not board[next_next_pos]: break
                
                captured_pieces = board[next_next_pos]
                captured_team = "A" if captured_pieces[0].endswith("_a") else "B"
                animation_events.append({'type': 'capture', 'pos': next_next_pos, 'team': captured_team, 'pieces': captured_pieces})
                
                for token in captured_pieces:
                    value = 10 if token.startswith("mandarin") else 1
                    if token.endswith("_a"): score["A"] += value
                    else: score["B"] += value
                
                board[next_next_pos] = []
                animation_events.append({'type': 'score_update', 'score': score.copy()})
                current_index = next_next_index
            else:
                tokens_to_scatter = [t for t in board[next_pos] if not t.startswith("mandarin")]
                if not tokens_to_scatter: break
                
                animation_events.append({'type': 'pickup', 'pos': next_pos, 'pieces': tokens_to_scatter})
                board[next_pos] = [t for t in board[next_pos] if t.startswith("mandarin")]
                
                index = next_index
                current_pos_for_animation = next_pos
                for i, token in enumerate(tokens_to_scatter):
                    target_index = (index + direction * (i + 1)) % len(order)
                    target_pos = order[target_index]
                    board[target_pos].append(token)
                    animation_events.append({'type': 'drop', 'from_pos': current_pos_for_animation, 'to_pos': target_pos, 'piece': token})
                    current_pos_for_animation = target_pos
                current_index = (index + direction * len(tokens_to_scatter)) % len(order)
                if not apply_e2:
                    break

        self.game_state["board"], self.game_state["score"] = board, score
        
        is_end = not any(t.startswith("mandarin") for t in board["QA"]) and not any(t.startswith("mandarin") for t in board["QB"])
        if is_end:
            pass

        return steps, animation_events, is_end
