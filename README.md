# Flappy Bird with deep reinforcement learning
## Setup
### 1. Clone the repository
```bash
git clone https://github.com/a-mx/flappybird-rl
cd flappybird-rl
```
### 2. Create virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate #Windows
source .venv/bin/activate #Linux
```
### 3. Install required packages
```bash
pip install uv
uv pip install -r requirements.txt
```
#### Training
```bash
python -m train
python -m train --path model/yourmodel.pth #Continue training
```
#### Evaluation
```bash
python -m test --path model/yourmodel.pth
```