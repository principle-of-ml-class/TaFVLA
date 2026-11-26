# TaF-Adapter 학습 안정성 설정

`taf_adapter_sparsh_episode_stable`은 기존 Episode Split 설정을 상속하고,
codebook 초기화와 learning rate 변화만 추가합니다. 배치 128, weight decay 0.001,
dropout 0.2, frozen Sparsh DINO Small, 512×128 codebook 두 개, loss와 manifest,
gallery 및 `epoch_shuffled` sampler/seed는 기존 값입니다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m script.train_adapter \
  --config-name=taf_adapter_sparsh_episode_stable

# 물리 GPU 0의 UUID를 사용하고 Rich dashboard와 trainctl을 함께 실행
bash scripts/train_sparsh_episode_stable.sh
# 학습 없이 실행 명령만 출력
bash scripts/train_sparsh_episode_stable.sh --dry-run
```

기존 trainctl Discord webhook 설정을 사용합니다. 결과와 로그는
`outputs/comparison/sparsh_episode_stable_runs/<manifest_id 앞 8자>_N5_seed42/`에
저장하며 기존 실험 폴더가 차 있으면 fresh run은 오류로 중단합니다.
trainctl 프로세스 로그 경로는 실행 시 출력되고, Rich 화면은 해당 로그와
`metrics.jsonl`을 읽습니다.

K-means는 Train window를 sequence → episode 순서로 균등 배분한 뒤,
episode 내부 시간 구간을 나누어 중복 없이 샘플링합니다. 샘플 수는 최대 16,384개이고,
Train 후보가 적으면 실제 후보 수를 사용합니다. 어느 branch든 유효 latent가
codebook 크기보다 적으면 명시적으로 실패합니다. Train-only normalization을 적용한
초기 MLP 출력에 CPU FP32 K-means++와 Lloyd 반복 20회를 수행합니다.
센서 로더만 사용하며 이미지 decoding이나 Sparsh forward는 없습니다.
거리 지표는 기존 VQ와 같은 제곱 Euclidean 거리입니다.
빈 cluster는 Train latent로 다시 채우고, 마지막 assignment의 점유율과 남은 빈 cluster도 기록합니다.
동일한 초기 모델·데이터·seed에서 재현되며, training RNG를 소비하지 않습니다.

두 기능은 독립적으로 끌 수 있습니다.

```bash
# Uniform 초기화 + warmup/cosine
CUDA_VISIBLE_DEVICES=0 uv run python -m script.train_adapter \
  --config-name=taf_adapter_sparsh_episode_stable vq.initialization=uniform \
  training.output_dir=outputs/comparison/sparsh_episode_uniform_cosine_runs

# K-means + constant LR
CUDA_VISIBLE_DEVICES=0 uv run python -m script.train_adapter \
  --config-name=taf_adapter_sparsh_episode_stable training.scheduler.type=constant \
  training.output_dir=outputs/comparison/sparsh_episode_kmeans_constant_runs
```

스케줄러 설정이 없거나 `training.scheduler=null`이면 constant LR입니다.
Cosine 설정의 1/500/10,000번째 성공한 optimizer update에는 각각
`1e-6 / 1e-4 / 1e-5`를 사용합니다. 첫 update부터 LR을 적용하며,
checkpoint의 optimizer에는 다음 update용 LR이 저장됩니다.
AMP overflow로 update가 생략되면 global step과 scheduler step 모두 그대로이고
skip count만 증가합니다. Microbatch 수와 epoch 경계는 스케줄러를 진행시키지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m script.train_adapter \
  --config-name=taf_adapter_sparsh_episode_stable \
  training.resume=outputs/comparison/sparsh_episode_stable_runs/<실제_run_폴더>/step_00001000.pt
```

Resume은 모델·optimizer·GradScaler·scheduler·모니터링 누적 상태를 복원하고 K-means를 건너뜁니다.
LR 경로를 유지하기 위해 저장된 스케줄러 종류, 전체 update 수, peak/min LR,
warmup 설정을 바꾸면 오류를 냅니다. Scheduler state가 없는 옛 checkpoint는 constant 설정으로만
복원할 수 있습니다.

`metrics.jsonl`과 W&B는 기존 codebook utilization/perplexity에 더해
두 branch의 latent/vector norm, quantization distance, assignment entropy(nats)를 기록합니다.
이는 각 minibatch의 통계를 logging 구간에서 평균한 값입니다. 초기화 결과는
`codebook_initialization` record 및 W&B summary에 sample 수·episode/sequence별 support,
초기화 전후 거리/norm, cluster occupancy, 빈 cluster 수, 소요 시간을 남깁니다.

Gradient norm은 AMP unscale 후 측정합니다. `train/gradient_norm`은 clipping 전 값의
기존 이름이고, `gradient_norm_before_clipping`/`gradient_norm_after_clipping`을 함께 기록합니다.
`gradient_clipping_coefficient`는 해당 update의 전체 gradient에 곱한 비율이며,
`gradient_clipping_fraction`은 모니터링한 성공 update 중 clipping된 비율입니다.
이 누적 상태가 없는 옛 checkpoint를 복원하면 복원 이후부터 계산합니다.
`gradient_norm_module/<모듈 경로>`는 parameter를 직접 소유한 각 모듈의 clipping 전 norm으로,
logging/checkpoint 시점에만 수집합니다. 추가 backward는 없습니다.
LR은 마지막 성공한 update에서 사용한 값이며 `scheduler_step`은 성공한 update 수입니다.

검증 명령:

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m unittest tests.test_training_stability -v
# 물리 GPU 0만 노출한 28초 제한의 실제 데이터 smoke test
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" \
  uv run python -m script.check_training_stability
```

Smoke test만 Train sequence 2개에서 sample 1,024개, K-means 3회, minibatch 2개×batch 2, accumulation 2를
사용합니다. Codebook 크기와 latent dimension은 유지합니다. 검증 결과는 매번 고유한
`outputs/stability_validation/smoke_*/`에 저장합니다. 기본 16,384개/20회 초기화와
10,000-update 학습은 사용자가 학습 명령으로 실행합니다.
