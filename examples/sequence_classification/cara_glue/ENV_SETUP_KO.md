# 환경 설정 가이드 (한국어)

CaRA GLUE 벤치마크를 실행하기 위한 Python 환경 구성 가이드입니다.
이 환경은 **RTX 5090 (sm_120) + CUDA 12.8** 기준으로 검증되었으며,
다른 GPU에서도 PyTorch 빌드만 바꾸면 동작합니다.

## 검증된 버전 (재현 보장)

| 패키지 | 버전 | 비고 |
|---|---|---|
| Python | 3.11 | 3.10 이상이면 가능 |
| PyTorch | **2.11.0+cu128** | RTX 50xx (sm_120) 지원하는 빌드 |
| CUDA toolkit | 12.8 (torch 번들) | 시스템 CUDA 설치 불필요 |
| transformers | 5.5.4 | `processing_class` API (5.x) 사용 |
| datasets | 4.8.4 | |
| evaluate | 0.4.6 | |
| accelerate | 1.13.0 | |
| peft | 0.18.2.dev0 | 이 레포 editable 설치 |
| sentencepiece | 0.2.1 | DeBERTaV3 토크나이저 의존성 |
| scikit-learn | 1.8.0 | Matthews correlation 등 |
| numpy | 2.4.4 | |

---

## ⚡ 빠른 설정 (3-step, uv 사용)

`uv`는 빠른 Python 패키지 매니저입니다. 없으면 [공식 사이트](https://docs.astral.sh/uv/) 참고.

### Step 1. uv 설치 확인 / 설치

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2. 가상환경 생성 + PyTorch (CUDA 12.8)

```bash
cd /path/to/peft     # 이 레포 루트

# 1) Python 3.11 venv 생성
uv venv .venv --python 3.11

# 2) RTX 50xx / sm_120 지원하는 PyTorch (CUDA 12.8)
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu128
```

> **⚠️ 중요**: 일반 `pip install torch`는 CUDA 12.4 build를 받아오는데, 이건 **RTX 50xx에서 동작하지 않습니다**.
> `--index-url`에서 `cu128`로 받아야 sm_120 (RTX 5090) 가속 가능.
>
> 다른 GPU 사용 시:
> - RTX 30xx/40xx → `cu124` 또는 `cu128` 모두 OK
> - 더 옛날 GPU (T4, V100, RTX 20xx) → `cu121`
> - 또는 CPU-only: `--index-url https://download.pytorch.org/whl/cpu`

### Step 3. 나머지 의존성 + 이 레포 editable 설치

```bash
# requirements.txt에 명시된 transformers / datasets / etc.
uv pip install --python .venv/bin/python -r examples/sequence_classification/cara_glue/requirements.txt

# peft 자체를 editable 설치 (CaraConfig, PsoftConfig 등을 import 가능하게)
uv pip install --python .venv/bin/python -e .
```

---

## 🔍 설치 검증 (smoke test)

```bash
.venv/bin/python -c "
import torch, transformers, peft
print(f'torch  {torch.__version__}, cuda {torch.cuda.is_available()}, GPU = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'transformers {transformers.__version__}')
print(f'peft         {peft.__version__}')

# CaRA import 확인
from peft import CaraConfig, get_peft_model
print('CaraConfig importable ✓')

# 간단한 GPU matmul로 sm_120 동작 확인
x = torch.randn(64, 64, device='cuda')
y = x @ x.T
print(f'cuda matmul OK, dtype={y.dtype}, shape={tuple(y.shape)}')
"
```

**기대 출력**:
```
torch  2.11.0+cu128, cuda True, GPU = NVIDIA GeForce RTX 5090
transformers 5.5.4
peft         0.18.2.dev0
CaraConfig importable ✓
cuda matmul OK, dtype=torch.float32, shape=(64, 64)
```

---

## 🚀 1-epoch 동작 테스트 (~ 2분)

```bash
.venv/bin/python examples/sequence_classification/cara_glue/run_glue_cara.py \
    --task mrpc --seed 42 --num_train_epochs 1 \
    --output_dir /tmp/cara_smoke
```

다음과 비슷한 출력이 나오면 정상:
```
trainable params: 887,042 || all params: 185,310,724 || trainable%: 0.4787
[cara_glue] cara-only params: 294,912 (0.29M)
...
[cara_glue] wrote /tmp/cara_smoke/result.json
[cara_glue] mrpc seed=42 f1=0.86~0.87
```

---

## 💡 자주 만나는 문제

### 1. `torch.cuda.is_available() == False`

```bash
# 확인
nvidia-smi
# → GPU가 인식되는지 확인. 안 되면 NVIDIA 드라이버 설치 필요.

# 그리고 torch가 어떤 cuda 빌드인지
.venv/bin/python -c "import torch; print(torch.version.cuda)"
# → '12.8'이 출력되어야 함. 'None'이면 CPU 빌드를 받은 것.
```

### 2. `RuntimeError: ... sm_120 ... not compatible`

→ PyTorch가 **cu124** 빌드입니다. **cu128**로 재설치:
```bash
uv pip install --python .venv/bin/python --reinstall torch \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3. `RuntimeError: Found dtype Float but expected Half` (STS-B)

→ DeBERTaV3 + bf16의 알려진 버그. 이 레포의 `run_glue_*.py`는 이미 `torch_dtype=torch.float32`를 강제하므로 발생하지 않아야 합니다.
혹시 발생하면 `--bf16` / `--fp16` 플래그를 빼고 fp32로 학습.

### 4. `ModuleNotFoundError: No module named 'sentencepiece'`

→ DeBERTaV3 토크나이저가 필요로 합니다:
```bash
uv pip install --python .venv/bin/python sentencepiece
```

### 5. `TypeError: TrainingArguments got unexpected keyword argument 'overwrite_output_dir'`

→ transformers 5.x에서 일부 인자가 제거됨. `requirements.txt`의 `transformers==5.5.4` 사용 시 이 레포 코드와 호환됩니다.

### 6. `TypeError: ... 'tokenizer' is unexpected ...`

→ transformers 5.x에서 `tokenizer=` → `processing_class=`로 변경. 이 레포는 이미 새 API 사용중.

---

## 🛠 conda를 선호하시면

```bash
conda create -n cara python=3.11 -y
conda activate cara

# PyTorch는 conda나 pip 둘 다 OK. cu128은 보통 pip가 빠릅니다.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 나머지
pip install -r examples/sequence_classification/cara_glue/requirements.txt
pip install -e .
```

---

## 📁 디스크 사용량 참고

- venv (PyTorch + 패키지): **~7-8 GB**
- 사전학습 모델 캐시 (`microsoft/deberta-v3-base`): **~450 MB**
- 30-run sweep 1회 분량 (checkpoint 포함): **~5-15 GB**
  - `save_total_limit=1`이라 epoch마다 best+latest만 유지
  - sweep 종료 후 `hf_trainer/` 폴더 삭제 가능

```bash
# sweep 끝난 뒤 디스크 정리
rm -rf results/*/*/seed_*/hf_trainer
```

---

## 🌐 인터넷 없는 환경에서 실행

처음 실행 시 HuggingFace에서 모델/데이터셋을 다운로드합니다. 오프라인 환경에서는:

1. **인터넷 되는 곳에서 한 번 캐시**:
   ```bash
   .venv/bin/python -c "
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   from datasets import load_dataset
   for task in ['cola', 'stsb', 'rte', 'mrpc', 'sst2', 'qnli']:
       load_dataset('glue', task)
   AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
   AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)
   "
   ```

2. **`~/.cache/huggingface/` 폴더를 오프라인 머신으로 복사**

3. **오프라인 머신에서 실행**:
   ```bash
   export HF_DATASETS_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   .venv/bin/python examples/sequence_classification/cara_glue/run_glue_cara.py ...
   ```

---

## 다음 단계

환경 구성이 끝났다면 [README.md](./README.md)의 sweep 명령어를 참고하세요.
빠른 시작:

```bash
# 전체 30-run sweep (5 seeds × 6 GLUE tasks)
.venv/bin/python examples/sequence_classification/cara_glue/sweep.py \
    --script run_glue_cara.py \
    --seeds 42 1234 24 7 99 \
    --base_output_dir results/cara_r8_psoft \
    --extra --eval_mode psoft_val_test

# 결과 집계 (results/cara_r8_psoft/RESULTS.md 생성)
.venv/bin/python examples/sequence_classification/cara_glue/aggregate.py \
    --results_dir results/cara_r8_psoft
```
