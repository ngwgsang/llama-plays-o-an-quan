from environment import Enviroment
from agent import Agent

# Initialize game environment and agents
env = Enviroment()

agent_a = Agent(
    team="A",
    persona="an intelligent and strategic player",
    model_endpoint="openai/gpt-4o-mini-2024-07-18",
)

agent_b = Agent(
    team="B",
    persona="a risk-taking and aggressive player",
    model_endpoint="openai/gpt-4o-mini-2024-07-18",
)

round_idx = 0

while True:
    round_idx += 1
    print(f"\n--- Start of Round {round_idx} ---\n")

    # ----- Player A turn -----
    output_a = agent_a.play_turn(
        env.get_game_state(),
        env.get_available_pos("A")
    )

    _, _, _ = env.commit_action({
        "pos": output_a.action.pos,
        "way": output_a.action.way,
    })

    is_end, reason = env.is_end()
    if is_end:
        print(f"\n[GAME END] {reason}")
        break

    # ----- Player B turn -----
    output_b = agent_b.play_turn(
        env.get_game_state(),
        env.get_available_pos("B")
    )

    _, _, _ = env.commit_action({
        "pos": output_b.action.pos,
        "way": output_b.action.way,
    })

    is_end, reason = env.is_end()
    if is_end:
        print(f"\n[GAME END] {reason}")
        break


print("\nFinal Game State:")
print(env.get_game_state())
