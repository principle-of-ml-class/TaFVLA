# 네 상황 분류와 N별 학습·평가 파이프라인

TaFVLA 안에서 준비, 학습, 평가, 시각화와 oracle routing을 연결한다. 기존 데이터·checkpoint를 수정하지 않는다.
새 schema는 `phase4-manifest-v1`이며 이전 centered/dynamics5 manifest와 별도다.
가져온 함수와 원본 blob SHA는 [출처 기록](phase4_sources.json)에 있다. 외부 저장소의 설치나 개인 경로는 필요 없다.

## 짧은 실행 명령

프로젝트 루트에서 실행한다. 아래 준비와 학습은 전체 데이터를 처리하는 사용자 실행 명령이다.

```bash
# 원본 episode → split → Train calibration → 네 phase → 공통 manifest/normalization/gallery
bash scripts/phase4.sh prepare

# Streamlit 네 화면
bash scripts/phase4.sh ui

# 특정 N: 학습 후 retrieval + endpoint Ridge/VQ 평가
bash scripts/phase4.sh run --n 5

# N=1,3,5,7,10,20 순차 실행, 마지막에 집계와 oracle routing
bash scripts/phase4.sh run

# 실행 없이 N별 명령/출력 경로 확인 (준비 후 사용)
bash scripts/phase4.sh run --dry-run

# 저장된 예측으로 집계·routing만 다시 실행
bash scripts/phase4.sh aggregate
# 기존 분석을 보존한 별도 분석 출력
bash scripts/phase4.sh aggregate --output outputs/phase4/analysis_review
```

선택 UI 의존성은 `requirements-ui.txt`로 분리했다. 새 환경에서만 다음 명령이 필요하다.
기존 PyTorch/CUDA를 설치하거나 바꾸지 않으며, 학습 CLI는 Streamlit/Plotly를 import하지 않는다.

```bash
uv pip install --python .venv/bin/python -r requirements-ui.txt
```

## 설정과 저장 위치

`config/phase4_pipeline.yaml`은 별도의 structured config다. 알 수 없는 key는 오류를 낸다.
다른 설정 파일은 `--config 경로`, 일부 값 변경은 `--set key=value`로 전달한다.

```bash
# 현재 실험 중인 stable 설정을 선택하는 경우에도 그 YAML 자체는 변경하지 않는다.
bash scripts/phase4.sh run --n 5 --set training_config=taf_adapter_sparsh_episode_stable
# 새 준비 설정을 비교할 때는 준비/결과 경로도 새로 지정한다.
bash scripts/phase4.sh prepare --set prepared_dir=metadata/phase4/another_protocol
```

기본 학습 설정은 `taf_adapter_sparsh_episode`다. 원하는 기존 Hydra 설정을 `training_config`에 지정한다.
K-means와 scheduler 옵션은 입력 설정의 값을 그대로 가져오며, 새 구현이나 override를 넣지 않았다.
모든 N에서 seed는 42 하나, batch size·accumulation·update budget·sampler·optimizer 조건은 동일하다.
InfoNCE negative 수는 실제 forward batch에 의해 정해지며 accumulation으로 늘어나지 않는다.

기본 준비 경로는 `metadata/phase4/v1/`이다.

```text
episode_split.json           # 라벨보다 먼저 확정한 raw episode 배정/원본 identity
calibration.json             # sensor별 Train 기준값
labels/episode_*.parquet     # 모든 원본 frame의 0.2/0.3/fused 판정, 신호/지원 범위
labels/episode_*.json        # episode 부호/baseline, threshold와 시간창 진단
episodes.json               # UI 탐색 목록, raw 파일 provenance, frame/anchor support
manifest.parquet            # 모든 N에서 공통인 endpoint, phase, split, identity
manifest_identity.json
normalization_<ID>.pt       # 공통 학습 support의 고유 Train frame 통계
gallery_plan.json           # seed42, 최대256개, episode 내 gap20
summary.json                # 설정/출처/개수/제외 사유/산출물 checksum
```

학습 결과는 `outputs/phase4/v1/<manifest 앞8자>/<experiment 앞12자>/` 아래에 저장한다.

```text
state.json                          # 단계 완료/실패 기록; 재실행 시 호환성 검사
N5_seed42/resolved_training.yaml     # 시작 시 고정한 기존 Hydra 설정의 snapshot
N5_seed42/train/<ID>_N5_seed42/       # 원래 training CLI의 checkpoint/metrics/W&B
N5_seed42/evaluation/
  retrieval_validation.json / retrieval_test.json
  retrieval_*.parquet
  endpoint/
    cache/{train,validation,test}/   # embedding/endpoint target/reconstruction memmap
    code_pair_bank.json
    ridge_probe.npz
    predictions/<split>/<method>/part_*.parquet
    prediction_index.parquet        # episode별 UI 읽기용 index
    metrics.parquet / summary.json
analysis/
  policy.json                       # Validation에서 선택한 정책, Test routing 전에 저장
  comparison.csv / comparison.parquet / retrieval.parquet
  predictions/<split>/<base_method>/
  *_phase_comparison.png / summary.json
```

같은 출력 경로에 다른 config/manifest/checkpoint 결과를 덮어쓰지 않는다.
실패한 준비·평가는 임시 staging 폴더의 `failure.json`에 이유를 남긴다. 정상 Unknown으로 바꾸지 않는다.
Runner는 실패한 N을 기록한 뒤 다음 N으로 진행하고, 완료된 단계는 재사용한다.
중단된 학습은 같은 manifest/config의 최신 checkpoint가 있을 때 기존 training CLI로 resume한다.
checkpoint 없는 미완료 학습 폴더는 덮어쓰지 않고 오류를 낸다.
원본 YAML 변경과 무관하게 고정 snapshot을 사용하며, gallery·batch/sampling·checkpoint identity도 검사한다.
GPU 0을 UUID로 선택하고 내부에서는 cuda:0을 사용한다. Pipeline 간 GPU lock과 N별 순차 실행을 사용한다.

## 분류와 분할을 구분하는 이유

`split_reference=auto`는 현재 `metadata/episode_split/manifest.json`을 우선 사용한다.
Episode split이 없고 예전 sequence manifest만 있으면 Test sequence 배정만 가져오고 나머지를 episode로 배정한다.
참고 자료가 없으면 설정한 비율/seed로 Test sequence와 sequence 내부 Train/Validation episode를 결정한다.
명시적으로 지정한 파일이 없으면 오류를 내며, 새 split을 원하면 `split_reference=null`을 사용한다.
기존 episode 배정은 재사용하지만 과거 No Contact로 제외된 anchor 목록은 재사용하지 않는다.

모든 raw episode에서 position 19부터 끝까지 공통 anchor를 만든다. Phase 변화로 자르지 않는다.
Unknown도 포함하며 짧은 history만 제외한다. Label 생성에는 raw Fz의 episode별 부호/baseline 처리를 쓰고,
모델 학습 target은 원래 6축 wrench와 pressure를 그대로 쓴다.

Sensor별 Train의 고유 raw frame으로 noise/IQR을 계산한다. Noise는 각 episode 안에서 2차 차분한 MAD/sqrt(6)의 중앙값이다.
이 noise는 운동이 섞일 수 있는 대용치이며 물리적인 센서 노이즈 보정값은 아니다.
서로 다른 episode 사이의 차분은 만들지 않는다. 해당 센서에 Train이 없으면 명시적으로 실패한다.
Validation/Test에는 저장한 기준값을 적용하며, episode별 부호/하위 10% pressure baseline 처리는 별도 진단으로 저장한다.

0.2/0.3초 창은 모델 N과 독립적이다. 30Hz에서는 각각 7표본(좌3/우3), 10표본(좌4/우5)이다.
선형 회귀 기울기와 끝점 차이, range_fraction=.02, noise multiplier=3, Holding 절반 threshold,
단일 창 run 최소2프레임을 팀원 구현과 같이 적용한다. Fusion 뒤에는 추가 필터를 넣지 않는다.
Timestamp 단조성·finite·sampling 간격을 검증하고 실제 support/span을 기록한다.
창이 부족한 경계/짧은 episode는 이유와 함께 Unknown으로 기록하지만 파일 오류·NaN·매칭 오류는 실패 처리한다.
이 라벨은 미래 force와 episode 전체 부호/baseline을 사용하는 offline 휴리스틱이며 사람 정답 검증은 아니다.

## 두 예측 방법과 지표

같은 checkpoint의 동일한 tactile embedding을 재사용한다.

- Ridge: Train endpoint wrench6+pressure144에 fit한다. Feature/target 통계는 Train에서만 계산하고,
  alpha는 Validation의 endpoint standardized wrench 오차로 선택한다. Pressure도 같은 probe/alpha를 사용한다.
- VQ: Train에서 관찰한 code pair만 bank에 넣는다. **전체 concat latent의 cosine**으로 한 pair를 고르고,
  정규화하지 않은 원래 codebook vector를 decoder에 넣는다. 결과 이름은 **train-observed code-pair decoding**이다.
  Tactile-only API는 평가용 정답 wrench/pressure를 인자로 받지 않는다. 동점은 lexicographic pair 순서로 해결한다.
- Training mean: Train endpoint 평균값이다.
- GT VQ reconstruction: 정답 N프레임 센서를 VQ/decoder에 넣은 별도 진단이며 tactile-only 예측과 구별한다.

주 지표는 공통 endpoint의 standardized wrench RMSE이고, 모든 방법에서 같은 Train normalization의 wrench std를 쓴다.
Ridge 내부 feature/target 표준화는 Train endpoint 누적 통계에서 얻는다.
축별 raw MAE/RMSE, force/torque 구분 오차와 pressure RMSE도 저장한다. 물리 단위는 가정하지 않는다.
RMSE는 squared error 합/원소 수의 제곱근이며 개별 RMSE를 평균하지 않는다.
Embedding과 150개 target은 memmap, Ridge는 작은 충분통계, 예측은 chunk Parquet을 사용한다.

## 상황별 N 선택과 UI

N=1,3,5,7,10,20, seed42 결과만 사용한다. Method/split/manifest/checkpoint를 구분해 집계한다.
Ridge와 VQ 각각 Validation 최소 오차의 5% 이내인 가장 작은 N을 global 및 phase별로 선택한다.
정확한 최소 오차 N도 별도로 기록한다. Phase별 최소 support는 30 anchors·2 episodes다.
Unknown과 부족한 상황은 해당 method의 global N으로 fallback한다. 팀원의 N 숫자를 하드코딩하지 않는다.
정책을 먼저 저장하고 같은 sample ID의 저장된 Test 예측을 선택한다. 추론을 다시 하지 않는다.
평균 선택 N은 입력량 대용치이며 latency가 아니다. 성능 악화도 그대로 보고한다.

Streamlit은 준비 결과에 연결된 episode만 탐색하며 분류를 다시 실행하지 않는다.
Raw episode·label cache 수를 제한하고 선택한 episode만 읽는다. Model Analysis도 저장된 예측을 index로 읽는다.
Loading 주황 / Holding 초록 / Unloading 보라 / Unknown 회색을 사용한다.

## 가벼운 검증

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m unittest tests.test_phase4 -v
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m script.check_phase4
```

두 번째 명령은 합성 CPU fixture로 준비→여섯 N 평가→routing→Streamlit 네 화면을 검증한다.
학습 update·GPU·운영 W&B·Discord 호출은 없다. 실제 여섯 N 학습과 전체 데이터 전처리는 자동으로 시작하지 않는다.
