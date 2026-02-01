from enviroment import Enviroment
from agent import Agent




env = Enviroment()
agent_a = Agent(
    team="A", 
    persona="an intelligent and strategic player", 
    model_endpoint="openai/gpt-4o-mini-2024-07-18"
)


agent_a.play_turn(env.get_game_state(), env.get_available_pos("A"))

