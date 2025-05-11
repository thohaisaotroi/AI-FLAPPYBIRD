# AI-FlappyBird (NEAT + DQN)

Đây là đồ án trí tuệ nhân tạo mô phỏng AI chơi game Flappy Bird bằng hai phương pháp:
-  NEAT (NeuroEvolution of Augmenting Topologies)
-  DQN (Deep Q-Learning)

## Cách chạy

### NEAT
- Huấn luyện: `python main.py`
- Test AI tốt nhất: `python test_best_genome.py`

### DQN
- Huấn luyện: `python train.py`
- Kiểm tra mô hình: `python test.py`

## Yêu cầu thư viện
Cài các thư viện bằng pip nếu cần:

```bash
pip install pygame neat-python torch tensorboardX matplotlib opencv-python numpy
