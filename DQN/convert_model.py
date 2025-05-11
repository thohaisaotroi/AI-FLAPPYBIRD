import torch
from src.deep_q_network import DeepQNetwork
import torch.serialization

torch.serialization.add_safe_globals([DeepQNetwork])

model = torch.load("trained_models/flappy_bird_2000000.pt", map_location="cpu", weights_only=False)

torch.save(model.state_dict(), "trained_models/flappy_bird_2000000_state_dict.pt")
print("Đã chuyển sang state_dict thành công.")
