# 🧠 GNN과 Federated Learning을 활용한 이상거래 탐지

이 프로젝트는 Graph Neural Networks (GNN) 기반의 모델을 Federated Learning 환경에서 학습시켜, 금융 거래 데이터 내 이상거래(사기) 를 탐지하는 데 목적이 있습니다.

## 🧪 Key Features
🕸 거래 데이터를 위한 그래프 신경망(GNN) 기반 모델

🔐 서버-클라이언트 구조의 Federated Learning 학습 방식

⚙ 사용자 정의 클라이언트 노드 설정

🧹 그래프 데이터 분할 및 전처리를 위한 유틸리티 스크립트 포함

## 📦 Project Structure

```bash
fraud_gnn_federated-main/
│
├── federated/                      # Federated Learning 서버/클라이언트 로직
│   ├── client.py                   # 클라이언트 로직 정의
│   ├── client_main.py              # 클라이언트 실행 진입점
│   ├── server.py                   # 서버 로직 정의
│   ├── server_main.py              # 서버 실행 진입점
│   └── utils.py                    # 공통 유틸 함수
│
├── graph/                          # GNN 모델 및 그래프 데이터 유틸
│   ├── gnn_model.py                # GNN 모델 정의
│   ├── dataset.py                  # 그래프 데이터셋 로더
│   └── graph_utils.py              # 그래프 전처리/유틸 함수
│
├── experiments/                    # 학습 및 평가용 스크립트
│   ├── train_local.py              # 로컬 학습 실행 스크립트
│   └── evaluate.py                 # 모델 평가 스크립트
│
├── scripts/
│   └── split_clients.py            # 클라이언트별 데이터 분할 스크립트
│
├── config/
│   └── settings.yaml               # 학습 설정 파일
│
├── client_node_config.json         # 클라이언트 노드 설정 (연합 학습용)
├── requirements.txt                # Python 의존성 목록
└── .gitignore                      # Git 무시 파일 목록

```

## 🚀 Getting Started

### 1. 의존성 설치
아래 명령어를 통해 필요한 Python 패키지를 설치하세요:

```bash
pip install -r requirements.txt
```

### 2. Configure the settings
연합 학습 구조 및 GNN 학습 파라미터를 설정하려면 다음 파일을 수정하세요:

config/settings.yaml – 에폭 수, 배치 크기, 학습률, 모델 구조 등의 학습 파라미터 설정

client_node_config.json – 클라이언트 수, 각 클라이언트에 할당된 데이터 경로, 연합 토폴로지 등 설정

### 3. 서버 및 클라이언트 실행
각각의 터미널에서 아래 명령어를 실행하세요:
```bash
# Terminal 1: Start the server
python federated/server_main.py

# Terminal 2-N: Start clients
python federated/client_main.py --client_id 0
python federated/client_main.py --client_id 1
...

```

### 4. Evaluate (Optional)
```bash
python experiments/evaluate.py
```
