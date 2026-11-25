# DataLoader 분석과 실행 방법

대기 시간의 원인을 구분할 계측과 비교용 옵션을 추가했다. 실제 데이터 benchmark와
GPU 학습은 실행하지 않았다. 실행 중인 학습, checkpoint, 기존 metrics/cache와 Git은
조작하지 않았다. `scripts/train_server.sh`의 연구 설정과 메시지 생성 코드는 변경하지 않았다.

## 확인한 사실과 남은 가설

- 사용자 측정: 0.0197 steps/s, data wait 50,605ms/batch, wait ratio 0.9977.
  데이터 공급이 주된 대기 구간이다. 어느 내부 작업이 이를 지배하는지는 아직 미확정이다.
- 기존 Parquet 지표는 읽기만 포함하고 `_column_matrix`를 제외했다.
  기존 Video 지표는 open/seek/decode/RGB를 합친 호출 시간이며 close는 제외했다.
  서로 다른 호출 수와 worker 중첩 때문에 두 평균을 더해 batch 대기를 설명할 수 없다.
- 중첩 Pressure 열은 기존 구현에서 `to_pylist()` 후 행마다 Python 객체를 만들었다.
  합성 float32 1,024×12×12, 5회 중앙값: 기존 16.4951ms → Arrow 경로 0.4728ms,
  약 34.9배, 배열 값 정확히 일치. **변환 함수만의 CPU 측정이며 학습 speedup이 아니다.**
- Sensor miss마다 파일 전체의 세 열을 다시 읽는다. 256MiB에는 index/force/pressure
  약 44만 행만 들어간다(행당 608byte). 실제 파일별 크기, 접근 빈도, eviction 측정 전에는
  cache thrashing을 주원인으로 단정할 수 없다. 한 파일이 한도를 넘으면 매번 읽는다.
- Decoder close는 기존 지표의 사각지대였다. open/close 반복, keyframe부터 불필요하게
  복호화한 프레임 수, seek 비용을 각각 측정한다. 재사용 효과는 실데이터에서 미측정이다.
- comparison 학습은 이미 고정된 sampler 계획을 갖는다. epoch마다 다시 세는 부분을
  정확한 `len()` 조회로 바꿨다. 일반 sampler의 길이는 실제 yield 수와 다를 수 있어 유지했다.

## 구현과 의미

`dataset/taf_dataset.py`: Arrow 중첩 list를 native flatten하는 경로, 구간별 timer,
센서 eviction/oversize/loaded-byte 통계, worker별 최대 개수가 정해진 decoder LRU.
Hydra 학습 기본값은 worker당 decoder 2개이며, 0으로 설정하면 매 호출 닫는다. 재사용 시에도 매번 같은 PTS로
backward keyframe seek하고 모든 요청 PTS를 검증한다. PID가 다른 handle은 재사용하지
않고, spawn 시 handle을 제거한다. 예외 시 전부 닫고 worker 종료 finalizer를 등록한다.

`utils/pipeline.py`: 선택적으로 배치 내부 읽기를 파일별로 묶고 sampler의 순서로 되돌린다.
샘플 추가·누락·배치 재구성·RNG 사용은 없다. worker 시간은 배치 끝에서 공유해
프레임마다 공유 lock을 잡지 않는다. 상세 계측의 custom batch가 pin-memory 시간을 잰다.

`utils/performance.py`, `script/train_adapter.py`: 기존 metrics.jsonl/W&B 경로에 `perf/*`를
추가한다. Frozen ViT와 Temporal Transformer는 hook으로 따로 측정하고 모델·loss는
수정하지 않는다. CUDA event는 보고 시에만 동기화한다. Rich에는 wait, Parquet/video,
cache hit, throughput, RAM/VRAM을 표시한다. GPU utilization은 API가 가능할 때 보고
시점의 표본이며, 구간 평균이 아니다.

주요 지표:

| 지표 | 단위·해석 |
|---|---|
| `perf/data_wait_ms`, `perf/data_wait_ratio` | 메인 `next(iterator)`의 wall time |
| `perf/worker_<stage>_seconds`, `_calls` | 구간 내 모든 worker 누적 시간·호출 수 |
| `perf/<stage>_worker_ms_per_call` | 누적 시간을 해당 stage의 호출 수로 나눈 값 |
| `perf/<stage>_worker_ms_per_batch` | 구간 누적 시간 / 소비한 batch 수; 실제 batch latency가 아님 |
| `perf/worker_process_<pid>_sample_prepare_seconds` 등 | 프로세스별 sample/batch/collate 누적 시간 |
| `perf/sensor_cache_hits`, `_cache_misses`, `_evictions`, `_oversize`, `_loaded_bytes` | 구간 cache 통계 |
| `perf/video_cache_hit_rate`, `perf/worker_decoded_frames` | decoder 재사용률, seek 이후 버린 프레임도 포함한 decode 수 |
| `perf/delivery_queue_ms_per_batch` | collate 완료→수신, pin 시간을 제외; IPC·직렬화·큐 체류·스케줄링을 포함 |
| `perf/pin_memory_ms_per_batch`, `perf/h2d_transfer_ms` | CPU pinning wall time, CUDA event로 잰 H2D 시간 |
| `perf/vit_forward_ms`, `perf/temporal_forward_ms`, `perf/backward_ms`, `perf/optimizer_ms` | CUDA event 구간 평균; CPU에서는 `_cpu_ms` |
| `perf/worker_<pid>_cpu_pct`, `_rss_gib`, `_peak_rss_gib` | 직접 자식 프로세스의 /proc 통계; resource tracker도 포함 가능 |
| `perf/total_rss_gib` | 메인과 직접 자식의 RSS 합; 공유 페이지가 중복 집계될 수 있음 |

`parquet_open`은 파일/footer 접근, `parquet_read`는 열 읽기, `column_matrix`는 변환, `sensor_gather`는 선택한 행 복사·reshape이다.
`video_decode`는 기존처럼 open/seek/decode/RGB를 포함하므로 하위 stage와 더하면 중복이다.
기존 `perf/parquet_read_ms`는 상세 모드에서 open을 제외한 read 평균이다.
legacy 경로는 PyArrow의 통합 `read_table` 내부 open을 따로 분리할 수 없다.
`av1_decode`는 decode iterator의 다음 프레임 반환 비용이며 iterator 종료 확인도 호출에 포함된다.
worker 작업은 겹치고, prefetch가 구간 경계를 넘으므로 누적 시간을 wall time과 합산하지 않는다.
진행 중 batch의 로컬 timer는 다음 collate에서 반영된다.
별도의 순수 IPC kernel 시간은 측정하지 않는다.

상세 계측을 끄면 frame timer, collate/pin wrapper, model hook이 없어지고 기존의 가벼운
공유 read/decode 통계만 남는다. `monitor_cache=true`는 선택적 공유 cache 통계를 켠다.
모든 계측을 끄려면 `trainctl.performance.enabled=false`를 사용한다.

## Feature Cache 비용과 재사용

기존 `FrozenFrameCache`를 유지했다. 기본 build batch=32를 시간축 T=32로 취급해
max_window_size=20 검사에 걸리던 문제를, 독립 frame B=32,T=1로 전달하도록 수정했다.
ViT에 들어가는 펼친 frame 순서와 개수는 같다. read-only mmap의 동시 열린 파일 수를 제한하고,
feature를 복사한 후에만 이전 mapping을 닫는다. 동일 영상의 서로 다른 subset/정밀도
cache가 metadata를 덮어쓰지 않도록 identity별 sidecar를 사용한다.
기존 유효한 sidecar는 읽기 호환한다. 로더는 생성 시 고정한 ordinal 목록을 사용한다.
전체 frame cache의 identity는 N과 무관하므로 동일 backbone·전처리·정밀도라면
N=1,3,5,7,10,20에서 재사용한다. benchmark의 선택된 frame만 저장한 subset cache는
N에 따라 frame 집합이 달라 전체 cache와 재사용 범위가 다르다.

원본 전체 스캔 없이 기존 `metadata/manifest.json`과 Parquet footer만 읽었다.
anchor는 **1,831,108개**다. manifest 생성 코드에서 각 frame은 채택 또는 한 가지 제외
사유에 속한다. rows+exclusions로 원본 센서 프레임 **10,053,265개**를 추정할 수 있다.
기존 cache는 `all_episodes` 전체를 대상으로 하므로 anchor 수만으로 용량을 계산하면 작게 잡힌다.

| 대상 | FP16, 384차원 | FP32, 384차원 |
|---|---:|---:|
| anchor 개수만 계산한 참고치 | 1.31GiB | 2.62GiB |
| 원본 frame 전체, 영상 ordinal 중복이 없다는 가정 | **7.19GiB** | **14.38GiB** |

위 수치는 feature payload만이다. JSON ordinal/timestamp, NPY header, 잠금 파일,
생성 중 한 영상의 임시 파일과 다른 backbone/정밀도 버전은 추가 공간을 차지한다.
정확한 unique-video-frame 수는 사용자 실행 `estimate`가 계산한다.

생성 시간은 아직 측정하지 않았다. 전체 소요 시간은 `unique_frames / 생성 frames/s`다.
**가정 예시**로 전체 pipeline이 100fps면 약 27.9시간, 500fps면 약 5.59시간이다.
실측 예측치가 아니다. `cache` benchmark는 일부 frame의 decode+ViT+쓰기 시간을 재고
전체로 외삽한 `full_build_seconds_extrapolated`를 출력한다. 무작위 subset의 seek 비용과
storage 상태 때문에 오차가 생길 수 있다. cache hit가 섞이면 외삽하지 않는다.
`estimate --cache-build-fps RATE`에 이 생성률을 전달해 재계산할 수 있다.

손익분기 step 수는 `전체 생성 시간 / (raw 초/step - cached 초/step)`이다.
분모가 0 이하이면 손익분기가 없다. 6개 N을 모두 실행하면 각 N에서 절약한 누적
시간을 합쳐 생성 비용과 비교한다. benchmark의 `subset_build_break_even_steps`는
subset 생성 비용만 사용하므로 전체 cache 손익분기로 해석하면 안 된다.
캐시 read 시간은 `feature_lookup`, cache 생성 hit rate는 영상 단위 재사용률이다.
후자는 학습 중 frame lookup hit rate와 다르다.

Cache ON은 GPU batch shape와 FP16 저장 때문에 raw와 bitwise 동일하다고 보장하지 않는다.
사용자 benchmark는 같은 256개 샘플의 InfoNCE batch에서 feature/embedding/loss 차이를 FP32 `atol=rtol=1e-5`, AMP
`1e-3` 기준으로 검사하고 실패하면 중단한다. 현재 실제 ViT/GPU의 허용 오차 충족은
미검증이며, 이 허용 오차는 선택한 검사 기준이지 이미 확인한 성질이 아니다.
엄격한 bitwise 연구 비교에는 cache OFF를 유지한다.

센서 mmap은 별도 선택 기능으로 구현했다. 원본 기준 payload 약 5.69GiB(+metadata)가
필요할 것으로 추정되며, 사용자 N=1 실측 후 Hydra 학습 기본값을 ON으로 바꿨다.
아래 Persistent Sensor mmap 절에서 생성·비교 명령과 안전성·메모리 해석을 설명한다.

## RAM과 worker 수

센서 cache는 worker당 256MiB 상한을 유지한다. worker 2/4/8의 cache 상한 합은
train loader만 0.5/1/2GiB다. 평가 worker는 이제 `training.eval_num_workers=0`이 기본이다.
이를 늘려 persistent evaluation loader를 사용하면 별도 worker/cache RAM이 든다.
파일 읽기/변환의 임시 배열, 데이터셋 metadata,
spawn 복사, decoder 내부 buffer, pinned batch는 이 상한 밖이다.

B=256, 224×224 RGB uint8 batch는 N=1 약 36.75MiB, N=20 약 735MiB다.
workers×prefetch만큼 대기 batch가 생길 수 있고 소비 중/pinning 중 batch는 별도다.
worker를 늘릴수록 storage 경합·CPU oversubscription·spawn 메모리 중복을 확인한다.
8 workers를 기본 학습 설정으로 적용하지 않았다. VRAM 개선 수치도 아직 측정하지 않았다.

## Hydra 옵션

| 옵션 | 기본값 |
|---|---|
| `data.sensor_arrow_conversion` | `true`; false는 기존 `read_table`+변환 경로 |
| `data.video_cache_max_entries` | `2`; 0은 매 호출 닫기 |
| `data.batch_group_reads` | `true` |
| `data.frame_cache_max_open_files` | `8` |
| `data.sensor_cache_max_mb` | 기존 256; mmap OFF일 때 RAM LRU 상한 |
| `data.sensor_mmap_enabled` | `true` |
| `data.sensor_mmap_dir` | `outputs/sensor_mmap` |
| `data.sensor_mmap_rebuild` | `false` |
| `data.sensor_mmap_max_open_files` | `16`; source당 3개 mapping |
| `data.sensor_mmap_batch_rows` | `8192` |
| `data.frame_cache_enabled` | 기존 false |
| `training.num_workers` | `4` |
| `training.eval_num_workers` | `0`; Validation/Alignment 전용 |
| `training.prefetch_factor` | `1` |
| `trainctl.performance.enabled` | 기존 true |
| `trainctl.performance.detailed_profiling` | 기존 false; 상세 측정 시 true |
| `trainctl.performance.monitor_cache` | 기존 false; cache 측정 시 true |

일반 학습에서는 세부 timer를 끈 채 사용하고, 원인 조사 때만 detailed를 켠다.
validation/checkpoint/40,000 steps/LR/optimizer/negative set/normalization은 기존 설정을 쓴다.

후속 N=1, batch 256, 20-step 실측에서 256MB/worker의 2 workers는 2.736초/step,
4 workers는 1.469초/step이었다. 512MB로 늘려도 적중률은 약 78.5%로 같았다.
따라서 학습 기본값은 4 workers, 256MB, prefetch 1로 맞췄다.
4 workers benchmark의 합산 RSS 표본 최대는 약 21.47GiB였다.
평가 기본값은 이제 0 workers여서 추가 worker pool을 만들지 않는다. 평가 worker를
늘리면 별도 RAM을 사용한다. 큰 N에서의 속도·메모리는 미측정이다.

## 사용자가 실행할 명령

**실행 중인 학습이 끝난 뒤 같은 GPU에서 순차 실행**한다. benchmark는 자동으로 실행하지 않았다.
아래는 repository 루트 기준이며 실제 데이터 경로는 train_server.sh와 같다.
각 실행은 고유한 새 출력 디렉터리를 사용하며 checkpoints를 만들지 않는다.
trainctl/W&B/Discord는 benchmark에서 호출하지 않는다.

```bash
cd /ssdg/spl_yeongyoo/TaFVLA

# 1. 기존 경로 baseline
bash scripts/benchmark_dataloader.sh baseline

# 2. Arrow + worker decoder LRU(2) + 배치 내부 파일 묶기
bash scripts/benchmark_dataloader.sh optimized

# 같은 실행에서 고정된 한 batch plan으로 before/after 비교
bash scripts/benchmark_dataloader.sh compare

# 3. 전체 cache 용량 추정. feature 생성은 하지 않음.
# dataset 초기화는 기존 timestamp cache 상태에 따라 길어질 수 있음.
bash scripts/benchmark_dataloader.sh estimate

# 3. Cache OFF/ON. benchmark frame만 생성하고 수치 오차도 검사.
bash scripts/benchmark_dataloader.sh cache

# 4. workers 2/4/8을 순차 비교
bash scripts/benchmark_dataloader.sh workers

# 다른 N. 높은 N은 raw 경로 VRAM/RAM 요구량을 먼저 확인.
bash scripts/benchmark_dataloader.sh compare --window-sizes 1 3 5 7 10 20

# 생성률 실측 후 전체 생성 시간 재계산 예시; RATE를 실측값으로 바꿀 것.
# bash scripts/benchmark_dataloader.sh estimate --cache-build-fps RATE
```

`TAF_DATA_ROOT`, `TAF_MANIFEST` 환경변수로 경로를 지정할 수 있다.
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`는 unset일 때 1로 정한다.
같은 조건의 baseline/optimized 모두 같은 값을 사용한다.

기본 benchmark는 warmup 2+측정 10 optimizer steps, batch 256, prefetch 1이다.
manifest의 train split과 같은 `FixedValidationBatchSampler`를 사용한다.
정규화는 기존 `normalization_<manifest-id 앞 8자>.pt`를 읽고 identity를 검증한다.
필요하면 `--normalization PATH`를 지정한다. 새 정규화 계산이나 split을 만들지 않는다.
backbone 초기화는 comparison 학습과 같은 `seed+10000`을 사용한다.
각 case에서 모델 상태, optimizer, seed를 초기화하고 같은 계획을 재생한다.

결과 `metrics.jsonl`에는 초기화, sampler 계획 생성, iterator 시작, 첫 batch 대기,
warmup, 정상 상태를 분리해 기록한다. 실제 전체 데이터 초기화와 기존 고정 sampler
계획 생성은 짧지 않을 수 있다. 이 초기 비용도 결과 해석에 포함한다.
측정 중에는 CUDA를 step마다 동기화하지 않는다. 메모리 표본 수집·ID hash 비용은
정상 상태 wall time에 포함된다. 각 case의 계획 SHA와 실제 sample/frame 순서 SHA를 비교한다.
OS page cache를 비우지 않으므로 baseline→optimized 실행 순서의 warm-cache 이점이 있을 수 있다.
분리 명령을 역순으로도 실행해 변동을 확인한다. 10 steps 수치가 전체 40k 성능을 보장하지 않는다.

| 항목 | Before | After | Speedup | Memory | Correctness |
|---|---:|---:|---:|---|---|
| 사용자 N=1 학습 | 0.0197 steps/s | 미측정 | 미측정 | GPU peak 0.887GiB | 실데이터 비교 대기 |
| 합성 중첩열 변환 | 16.4951ms | 0.4728ms | 34.9× (함수 한정) | 별도 peak 미측정 | 배열 값 정확히 동일 |
| Raw DataLoader | 사용자 benchmark 필요 | 사용자 benchmark 필요 | 미측정 | JSON에 RSS/VRAM 기록 | 샘플 tensor/metadata 비교+순서 hash |
| Feature Cache | 사용자 benchmark 필요 | 사용자 benchmark 필요 | 미측정 | FP16 전체 payload 약 7.19GiB | 합성 모델 통과, 실제 GPU 오차 미검증 |

## 짧은 검증

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TERM=xterm-256color timeout 25s uv run --no-sync python -m unittest \
  tests.test_pipeline tests.test_throughput tests.test_performance tests.test_terminal_monitor -q
```

합성 AV1의 무작위·역방향 seek, 동일 영상의 episode 경계, PTS/tensor 일치,
오류 시 정리, FD 회수, cache 한도, disabled 계측, persistent spawn와 순서 보존을 검사한다.
이 실행 환경은 PyTorch FD 전달 소켓을 차단하므로 spawn 테스트에서만 텐서를 값으로
직렬화한다. production 공유 메모리 IPC와 실제 CUDA pinning/transfer는 서버 검증이 필요하다.
이 우회는 학습 코드에 적용하지 않는다. 별도의 CPU 합성 검사에서는 동일한 256개 배치로
baseline/optimized의 warmup 1+측정 1 step을 재생해 sample/frame 순서와 optimizer 이후
모델 상태가 정확히 같은지 확인한다. 실제 데이터/GPU throughput 결과는 아니다.
합성 모델에서는 cached/raw embedding·loss와
projection/temporal gradient, N 간 재사용, mmap eviction와 subset sidecar 독립성을 검사한다.

최종 로컬 검증: 관련 unit/synthetic 검사 30개 통과(12.59초).
Python compileall과 benchmark shell의 `bash -n`도 통과했다.
전체 학습·전체 데이터 benchmark·전체 cache build는 실행하지 않았다.

변경 파일:

- 데이터 경로: `dataset/taf_dataset.py`, `utils/dynamics.py`, `utils/pipeline.py`
- 계측/캐시: `utils/performance.py`, `utils/frame_cache.py`
- 설정: `utils/config.py`, `utils/common.py`, `config/taf_adapter.yaml`
- 학습/표시/비교: `script/train_adapter.py`, `script/terminal_monitor.py`,
  `script/profile_throughput.py`, `scripts/benchmark_dataloader.sh`
- 검증: `tests/test_pipeline.py`, `tests/test_throughput.py`, `tests/test_performance.py`
- 문서: `README.md`, `docs/dataloader_profiling.md`

마지막 변경 후 관련 검사 20개를 12.34초에 재확인했다.

## Persistent Sensor mmap

이 기능은 **센서 Parquet의 반복적인 압축 해제와 NumPy 변환**만 제거한다.
Arrow 변환, batch 내부 파일 묶기, worker 수, Decoder LRU는 그대로 두고 센서 backend만
바꿔 비교한다. 원본·timestamp cache·Frozen ViT cache·dataset 반환 형식은 유지한다.
Hydra 학습은 `data.sensor_mmap_enabled=true`가 기본값이며, 기존 Parquet+RAM LRU 경로는
`data.sensor_mmap_enabled=false`로 선택할 수 있다.

### 저장 구조와 검증

`utils/sensor_cache.py`의 `SensorMmapCache`가 원본 Parquet별로 다음 세 파일을 저장한다.

| 파일 | dtype | shape |
|---|---|---|
| `index.npy` | int64 | `(T,)` |
| `force.npy` | float32 | `(T,6)` |
| `pressure.npy` | float32 | `(T,144)` |

순서는 Parquet의 row order 그대로다. 기존 `_column_matrix`의 float32 변환을 재사용하므로
추가 양자화나 정규화가 없다. 기존 `cache['index'][ep.row_start] + offsets`와
`searchsorted`·일치 검사를 유지한다. 선택한 Force/Pressure는 복사한 writable 배열로
반환하므로 mapping이 eviction되어도 Torch tensor가 닫힌 mapping을 참조하지 않는다.

디스크에는 `cache_root/source_path_hash/identity_hash.json` 포인터와
`gen-<uuid>/{index,force,pressure}.npy`, `metadata.json`이 저장된다.
identity에는 canonical source path, size, mtime_ns, ctime_ns, inode/device,
format/conversion version이 들어간다. Metadata에는 원본 Arrow schema SHA-256,
row 수, 각 NPY의 dtype/shape/정확한 크기, 1,024행 단위 SHA-256을 저장한다.

새 파일은 source별 file lock 안에서 임시 디렉터리에 만들고 fsync한다. 완성된
세대 디렉터리를 rename한 뒤, metadata SHA-256을 포함한 포인터를 atomic replace한다.
Reader는 완성된 포인터가 가리키는 세대만 사용한다. Source가 생성 중 바뀌면 중단한다.
동일 작업에서 spawn worker들이 rebuild를 요청해도 공유된 rebuild token을 사용해
같은 source를 불필요하게 반복 생성하지 않는다.

- 모든 load는 `allow_pickle=False`, `mmap_mode='r'`다.
- 누락·절단·잘못된 헤더·metadata checksum 오류는 lock 안에서 경고 후 재생성한다.
- 유효한 헤더 아래의 데이터 손상은 **접근한 행 블록의 체크섬 검사로 감지해 명확한 오류**를 낸다.
  이 경우 `data.sensor_mmap_rebuild=true` 또는 builder의 `--sensor-mmap-rebuild`로 새 세대를 만든다.
- 블록 검증 결과는 현재 열린 mapping 수명 동안만 재사용한다. 파일 stat이 바뀌면 다시 연다.
  아직 읽지 않은 블록의 손상은 그 블록을 처음 접근할 때 검출한다.
- 디스크 부족·쓰기 실패 시 기존 포인터와 세대는 그대로 남는다. 완성되지 않은 임시 파일은
  사용하지 않는다. 정상 예외의 임시 파일은 정리하지만, 강제 종료의 잔여 파일이나
  이전 세대를 자동 삭제하지 않는다. 따라서 반복 rebuild·source 변경 시 디스크 사용량이 늘 수 있다.
- source dataset 내부 cache 경로는 거부한다. 캐시 무효화는 원본 파일을 수정하지 않는다.

### 메모리와 비용

생성은 `iter_batches(batch_size=data.sensor_mmap_batch_rows)`로 파일을 하나씩 처리하고
NPY에 순차 기록한다. 전체 센서 배열을 RAM에 모으지 않는다. 기본 8,192행 출력 배열은
약 4.75MiB이며, Arrow·압축 해제·일시 변환 버퍼는 별도다. 이 수치는 프로세스 전체 RAM 상한이 아니다.

Reader는 기본 source 16개, 즉 최대 48개 NPY mapping을 유지한다. eviction, 정상 종료,
예외·spawn serialization에서 handle을 정리하고 worker 종료 finalizer도 등록한다.
`max_open_files`는 source 개수이며 세 배열의 FD 수는 그 3배다.

mmap ON에서는 전체 파일 배열을 기존 RAM LRU에 다시 복사하지 않는다.
`sensor_cache_max_mb`는 mmap OFF의 RAM LRU에 적용되고, mmap은 열린 파일 수로 제한한다.
**mmap은 RAM을 전혀 쓰지 않는 기능이 아니다.** 접근한 페이지는 RSS/page cache에 포함되고
OS가 회수할 수 있다. 기존 Python manifest metadata의 worker별 복사는 이 변경의 대상이 아니다.

후속 실측 source frame 수 10,053,265를 기준으로 단일 세대 payload는
`10,053,265 × (8 + 6×4 + 144×4)` = 약 **5.69GiB**다.
NPY header, checksum metadata, 생성 중 한 파일의 임시 세대와 보존된 이전 세대는 추가 공간이다.
정확한 row 수와 용량은 `sensor-build --estimate-only`가 source Parquet footer로 계산한다.
실제 생성 시간과 mmap speedup은 아직 측정하지 않았다.

### 생성과 benchmark 명령

다음은 **사용자 실행용** 명령이다. Codex는 실제 데이터 캐시 생성·학습·benchmark를 실행하지 않았다.
각 실행은 고유한 새 `outputs/dataloader_bench/.../metrics.jsonl`을 사용한다.

```bash
cd /ssdg/spl_yeongyoo/TaFVLA

# 1. 전체 센서 cache 용량만 추정. 센서 cache를 생성하지 않음.
bash scripts/benchmark_dataloader.sh sensor-build \
  --sensor-mmap-dir outputs/sensor_mmap --estimate-only

# 2. A=기존 Parquet+RAM / B=새 mmap cache 생성 후 첫 실행 / C=같은 cache 재사용
# 기존 최적화 조건(N=1, batch256, workers4, prefetch1, RAM LRU256MB)은 고정.
CUDA_VISIBLE_DEVICES=3 bash scripts/benchmark_dataloader.sh sensor-compare \
  --workers 4 --sensor-cache-mb 256 --warmup-steps 2 --steps 20

# 3. 결과가 유리하면 전체 센서 cache를 별도로 생성. GPU forward/학습은 하지 않음.
bash scripts/benchmark_dataloader.sh sensor-build \
  --sensor-mmap-dir outputs/sensor_mmap

# 4. 전체 cache 재사용 성능 측정
CUDA_VISIBLE_DEVICES=3 bash scripts/benchmark_dataloader.sh sensor-warm \
  --sensor-mmap-dir outputs/sensor_mmap \
  --workers 4 --warmup-steps 2 --steps 20

# N별 비교 예시. sensor cache 자체는 N/backbone/seed와 무관하게 재사용됨.
# CUDA_VISIBLE_DEVICES=3 bash scripts/benchmark_dataloader.sh sensor-warm \
#   --sensor-mmap-dir outputs/sensor_mmap --window-sizes 1 3 5 7 10 20 \
#   --workers 4 --warmup-steps 2 --steps 20
```

`TAF_DATA_ROOT`와 `TAF_MANIFEST` 경로 환경변수는 기존 wrapper와 같다.
`--override data.sensor_mmap_max_open_files=32` 등으로 handle/batch 설정을 조정할 수 있다.
캐시 생성 중에도 기존 Dataset 초기화의 metadata/timestamp 검증은 실행된다.
따라서 timestamp cache가 없다면 초기화 자체가 오래 걸릴 수 있다.

`sensor-compare`는 benchmark batch plan이 사용하는 **source 파일 전체**만 새 cache로 만든다.
N/worker 조건별 새 디렉터리를 사용하고 B와 C는 동일 디렉터리를 공유한다.
전체 dataset 생성 비용과 혼동하지 않도록 preparation에 scope·rows·files를 기록한다.
`sensor-warm`은 지정 경로를 재사용하며 누락 파일은 준비 단계에서 생성한다.
이미 생성된 전체 cache의 warm 실험이라면 preparation의 `builds=0`인지 확인한다.
기본값 `--sensor-mmap off`로 기존 benchmark 모드도 그대로 사용할 수 있다.

실제 학습은 추가 옵션 없이 mmap을 사용한다. 기존 `train_server.sh`도 Hydra 기본값을
상속하므로 그대로 실행하면 된다. 기본 설정은 학습 workers 4, 평가 workers 0, prefetch 1,
센서 mmap ON, batch 파일 묶기 ON, worker별 decoder 2개이며 Frozen ViT Cache는 OFF다.
기본 캐시 경로는 `outputs/sensor_mmap`이고, 경로를 변경할 때만 다음 override를 사용한다.

```text
data.sensor_mmap_dir=/path/to/sensor_mmap
```

Batch 256, N 목록, seed, 40k steps, manifest, optimizer, normalization, checkpoint/resume,
validation/checkpoint 주기와 trainctl/W&B/Discord/Rich 흐름은 바꾸지 않는다.
사용 중 문제가 생기면 `data.sensor_mmap_enabled=false`로 기존 RAM LRU 경로로 되돌린다.
생성된 파일을 삭제할 필요는 없다.

### Cold/Warm 해석과 측정값

여기서 cold는 **준비 전 디스크 cache가 없는 상태**다. 생성 시간은 별도 preparation에
기록하고 학습 구간에서 제외한다. 따라서 B도 생성 과정에서 NPY의 OS page cache가
채워질 수 있다. C는 같은 디스크 cache를 재사용하며 B에서 접근한 페이지가 남아 있을 수 있다.
OS page cache를 강제로 비우거나 시스템 설정을 바꾸지 않는다. 어느 실험도 물리 디스크가
cold하다고 보장하지 않으며 `os_page_cache` 필드에 이 조건을 기록한다.

동일 모델 초기 상태·optimizer·seed·batch plan을 재생한다. 전체 측정 sample/frame 순서의
SHA-256을 원래 plan과 대조하고, 시간 측정 후 선택된 샘플의 Force/Pressure byte equality를
검사한다. AV1·collation·이미지 변환·batch 묶기 설정은 A/B/C에서 같다.

| 지표 | 의미 |
|---|---|
| `sensor_mmap_preparation` | 준비 wall time, build/load/conversion 시간, files/rows, 실제 cache 총 크기 |
| `seconds_per_step`, `perf/samples_per_second`, `perf/data_wait_ratio` | 준비·worker 시작·warmup을 제외한 학습 구간 |
| `perf/sensor_parquet_reads`, `perf/worker_parquet_read_calls` | 기존 RAM 경로의 파일 읽기; mmap steady에서는 0이어야 함 |
| `perf/sensor_mmap_parquet_read_calls`, `_parquet_read_seconds`, `_conversion_seconds` | mmap 생성 시의 Arrow batch read/변환; 기존 파일 단위 호출 수와 구분 |
| `perf/sensor_mmap_cache_hits`, `_cache_misses`, `_mapping_hits`, `_mapping_opens` | 디스크 cache 재사용·생성 miss와 process-local handle 재사용을 구분 |
| `perf/sensor_mmap_build_seconds`, `_load_seconds`, `_access_seconds`, `_verify_seconds` | 생성, NPY/meta load, 행 검색·검증·복사, 체크섬 시간 |
| `perf/sensor_mmap_minor_page_faults`, `_major_page_faults` | 상세 계측에서 sensor row access 구간의 process page fault 증가량 |
| `perf/sensor_mmap_initial_build_seconds`, `_initial_build_files` | 학습 시작 전 audit/normalization 또는 benchmark 준비 중 생성 비용 |
| `cpu_peak_sampled_pss_bytes`, `perf/total_pss_gib` | 공유 페이지를 비례 배분한 PSS; 권한이 없으면 누락/null |
| `cpu_peak_sampled_rss_bytes`, `worker_peak_rss_bytes` | 합산 RSS 표본 최대와 process별 HWM; 공유 mapping의 중복 집계 가능 |
| `perf/gpu_utilization_pct` | NVML 사용 가능 시 보고 시점 표본; 불가능하면 available=0과 경고 |

`access_seconds`는 **순수 page fault 시간이나 디스크 I/O 지연이 아니다.** page-in, 블록 체크섬,
index search와 결과 복사를 합친 wall time이다. `verify_seconds`는 그 안에 포함된다.
체크섬은 block당 최초 접근 시 수행하므로 안전성 비용도 benchmark에 포함된다.
PSS 역시 OS의 미매핑 file page cache 전체나 사용자 cgroup 사용량과 같지는 않다.
PSS/RSS 표본은 순차 읽기이며 정확히 동시 측정한 peak가 아니다.
메모리 표본·ID hash 수집 오버헤드는 A/B/C 모두 측정 wall time에 포함된다.
worker 시간은 겹치고 prefetch가 구간 경계를 넘으므로 wall time과 합산하지 않는다.

`sensor_mmap_comparison`에는 실측 steady speedup과 다음 손익분기점을 기록한다.

- subset 손익분기 step = 실제 subset build 초 / (Parquet 초/step − warm mmap 초/step).
- 전체 생성 시간 추정 = subset build 초 × 전체 source rows / subset source rows.
- 전체 손익분기 추정 = 위 전체 생성 시간 추정 / 절약 초/step.

차이가 0 이하이면 손익분기는 null이다. 전체 생성 시간은 row 비례 외삽으로,
source 크기·storage·OS page cache 차이에 따른 오차가 있다. 전체 builder를 실행한 경우에는
그 실제 build 시간을 사용해야 한다. N=1의 이득을 다른 N에 그대로 적용하지 말고,
여섯 N에서 절약하는 누적 시간을 공통 cache 생성 비용과 비교한다.

### 검증 범위

작은 임시 Parquet만 사용해 다중 Arrow batch, N=1/3/5/7/10/20, nonzero global index,
episode row offset, dtype·byte equality, source 변경과 생성 중 변경, 누락/절단 복구,
본문 손상 오류, 쓰기 실패 시 기존 publication 보존, FD eviction, RNG 불변을 검사한다.
2개의 spawn worker가 동시에 같은 source를 요청했을 때 한 세대만 생성됐고,
2개의 persistent DataLoader worker에서도 AV1 영상·모든 metadata·센서 tensor가 일치했다.
이 실행 환경의 PyTorch FD 전달 소켓 제한 때문에 해당 DataLoader 테스트만 기존과 같은
by-value tensor transport를 사용하며 production IPC는 변경하지 않았다.
CPU 합성 batch256에서는 A/B/C의 optimizer 후 모델 상태도 정확히 일치했다.
후속 사용자 N=1, batch256, workers4 실측은 0.409초/step, 626 samples/s,
Data Wait Ratio 16.7%였다. 이전 Parquet+RAM LRU 1.469초/step과 비교하면 약 3.6배다.
이 실행은 준비 단계에서 56개 파일을 새로 생성했고, 학습 구간에서는 Parquet read/변환이 0회였다.
전체 캐시와 다른 N의 속도·메모리·장기 학습 결과까지 검증한 수치는 아니다.

이번 변경의 관련 검사 56개가 통과했다(각 실행을 25초로 제한). 마지막 변경 후
캐시 publication·A/B/C optimizer 동일성·계측 관련 3개 검사도 재확인했다.
Python compileall, 기존/확장 shell의 bash 문법, 최초 구현 시 Hydra 기본 OFF와 override 검증도 통과했다.
전체 데이터에 대한 생성·전처리·실제 training·GPU benchmark는 실행하지 않았다.

이번 mmap 변경 파일은 `utils/sensor_cache.py`, `dataset/taf_dataset.py`,
`utils/performance.py`, `script/train_adapter.py`, `script/profile_throughput.py`,
`utils/config.py`, `utils/common.py`, `config/taf_adapter.yaml`,
`scripts/benchmark_dataloader.sh`, `tests/test_sensor_cache.py`,
`tests/test_profile_throughput.py`, `tests/test_performance.py`, `README.md`, 이 문서다.
기존 `utils/frame_cache.py`, timestamp cache 구현, `scripts/train_server.sh`,
Rich dashboard와 알림 메시지 코드는 이번 작업에서 수정하지 않았다.
