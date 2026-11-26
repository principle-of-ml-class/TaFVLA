# Sparsh DINO Small 통합 결과

ImageNet ViT 경로를 보존하면서, `taf_adapter_sparsh` 설정으로 공식 6채널 Sparsh DINO Small을 선택할 수 있도록 구현했다. VQ-VAE·single-positive InfoNCE·loss 계수·정규화·manifest·gallery는 바꾸지 않았다. 실제 학습은 시작하지 않았고 optimizer update가 없는 검증만 수행했다.

## 공식 모델과 weight

- 모델: [`facebook/sparsh-dino-small`](https://huggingface.co/facebook/sparsh-dino-small)
- Hugging Face revision: `1aa732e94c1dd58294d4553e139c4bbe90e1ba35`
- 파일: `dino_vitsmall.safetensors` — 87,593,104 bytes. Backbone만 있는 공식 파일을 사용하며, 큰 training checkpoint를 가져오거나 일부 key를 임의로 버리지 않는다.
- SHA-256: `4126ba669f0fbe62b50893f46185d021f86d7a3dc671ac6397b2e9e3b9aeb8b8`
- 실제 경로: `outputs/pretrained/sparsh-dino-small/1aa732e94c1dd58294d4553e139c4bbe90e1ba35/dino_vitsmall.safetensors`
- 소스 기준: [`facebookresearch/sparsh@fee6a05a97330eca014b59c71ac41596fe3974a7`](https://github.com/facebookresearch/sparsh/tree/fee6a05a97330eca014b59c71ac41596fe3974a7)
- 로딩 결과: **174 state keys, missing=0, unexpected=0**, 전체 weight/buffer finite. Checksum이나 strict load가 실패하면 실행을 중단한다. `pretrained=false`로 Sparsh를 임의 초기화하는 경로도 차단했다.

WAH의 dependency-light ViT-Base 구현을 참고하되 Small의 실제 width/head 수에 맞추고 공식 소스와 직접 대조했다. GR00T/VLA/ZeRO dependency를 가져오지 않았다. 기존 환경의 safetensors·huggingface-hub를 사용하며 추가 설치나 PyTorch/CUDA 변경은 없었다(`torch 2.14.1+cu126`). 기존 Git 기반 trainctl을 재조회하는 일반 dependency 재해결은 진행하지 않았으며 pyproject/lock의 기존 dependency를 유지했다.

공식 [ViT 구현](https://github.com/facebookresearch/sparsh/blob/fee6a05a97330eca014b59c71ac41596fe3974a7/tactile_ssl/model/vision_transformer.py)을 기준으로 patch=16, channels=6, width=384, blocks=12, heads=6, sinusoidal position, register=1, LayerNorm eps=1e-6이다. Attention은 같은 계산을 PyTorch SDPA로 표현했다. 라이선스 원문은 [여기](licenses/sparsh-CC-BY-NC-4.0.txt)에 보존했다.

## 입력과 출력

```text
원본 RGB                        (B,N,3,H,W)
각 window 내부의 현재/참조 RGB    (B,N,6,320,240)
Sparsh spatial patch tokens     (B,N,300,384)
Sparsh register tokens          (B,N,1,384), spatial pooling에는 제외
학습 가능한 attentive pooling   (B,N,384)
Linear projection              (B,N,256)
기존 causal Temporal Transformer
기존 마지막 summary token       (B,256)
```

300개 patch는 320/16 × 240/16이다. Register token을 CLS로 취급하지 않는다. Register는 backbone 내부 attention에 참여하지만, frame feature를 만드는 pooling에는 정규화된 spatial patch token만 사용한다.

각 local index `i`의 참조는 `max(0,i-5)`이고 채널 순서는 **현재 RGB, 참조 RGB**다. 공식 dataset이 현재부터 과거 순서로 concatenate하는 것과 일치한다. N=7의 마지막 pair는 `(I7,I2)`이고 N=1은 `(I1,I1)`이다. Dataset에 추가 frame을 요청하지 않으며 정확히 N개의 frame-pair feature가 생성된다. 기본 stride=5이고 stride를 바꾸면 checkpoint의 input identity도 달라진다.

## Preprocessing과 공식 구현의 차이

공식 [dataset](https://github.com/facebookresearch/sparsh/blob/fee6a05a97330eca014b59c71ac41596fe3974a7/tactile_ssl/data/vision_tactile.py), [RGB loader/resize](https://github.com/facebookresearch/sparsh/blob/fee6a05a97330eca014b59c71ac41596fe3974a7/tactile_ssl/data/digit/utils.py), [sensor별 설정](https://github.com/facebookresearch/sparsh/blob/fee6a05a97330eca014b59c71ac41596fe3974a7/config/data/vision_based.yaml)을 확인했다.

- RGB uint8은 255로 나누고 float는 finite [0,1]인지 검사한다. **ImageNet mean/std를 적용하지 않는다.**
- Landscape는 공식 loader처럼 시계 방향 90도 회전한다. 이후 공식 loader의 deterministic center-crop 규칙과 320×240 bilinear/antialias resize를 적용한다. 추가 random crop/flip/색상 augmentation은 없다.
- `data.image_size=null`로 원본을 encoder에 전달해, 기존 loader의 224×224 선행 축소를 피한다. RGB 변환·resize도 backbone chunk 단위로 수행해 전체 native window의 FP32 복사를 피한다.
- **custom_6x6_markers의 background subtraction은 기본 false.** 이는 공식 marker GelSight 학습 경로에서 background를 제거하지 않는 선택과 일치한다.
- `tactile_encoder.sparsh.background_subtraction=true`면 해당 **window 첫 이미지**를 배경으로 사용한다. 공식 `compute_diff`처럼 `clip(image-background+0.5)` 후 uint8 수준으로 양자화한다. 이 선택은 공식 외부 no-contact 배경과 다르며, 첫 frame에 접촉이 있을 수도 있다. Window 밖의 배경을 읽지는 않는다.
- 공식 PIL resize 대신 GPU tensor bilinear resize를 사용한다. 비교한 480×640 RGB 표본에서 차이는 최대 **0.0039216(약 1/255)**, 평균 **0.0011352**였다. PIL의 uint8 반올림과 완전히 bitwise 같은 preprocessing이라고 주장하지 않는다.

실제 smoke의 native 이미지는 **380×380**이었다. 이 정사각형 입력에도 공식 crop 수식을 그대로 적용하므로 중앙 285×380 영역을 320×240으로 변환한다. 즉 가장자리 일부가 잘린다. Custom sensor에서 이 시야가 적절한지는 성능 검증이 남아 있다. 임의의 crop 정책을 새로 만들거나 그 영향을 검증 없이 숨기지는 않았다.

## Pooling과 freeze

공식 [AttentivePooler](https://github.com/facebookresearch/sparsh/blob/fee6a05a97330eca014b59c71ac41596fe3974a7/tactile_ssl/downstream_task/attentive_pooler.py)의 `depth=1, complete_block=true`를 따른다. Learned query 하나가 spatial tokens를 cross-attention으로 집약하고, query residual 및 MLP residual을 적용한다. Pooling heads=12, width=384, MLP ratio=4, query/linear 초기화 std=0.02와 output weight rescaling도 반영했다. 별도 pretrained downstream pooling weight를 쓰는 것이 아니라 새로 학습하는 module이다. Module 이름 배치는 다르지만 공식 pooler에 같은 weight를 넣은 출력 차이는 **0**이었다.

Temporal hidden=256/layers=2/heads=4/dropout=0.2, causal mask, 마지막 summary token, 기존 positional embedding 방식을 유지했다.

| 부분 | N=5 parameter 수 | 기본 학습 여부 |
| --- | ---: | --- |
| Sparsh backbone | 21,894,144 | Frozen |
| Spatial attentive pooling | 1,774,848 | 학습 |
| Projection | 98,560 | 학습 |
| Temporal Transformer | 1,579,520 | 학습 |
| Positional embedding + summary token | 5,632 | 학습 |
| Force/Pressure encoder·decoder·VQ 전체 | 1,164,270 | 학습 |
| 전체 trainable | **4,622,830** | 학습 |

`train()` 이후에도 frozen backbone만 eval을 유지한다. Backbone forward에만 no_grad를 적용하고 pooling/projection은 범위 밖에 둔다. `tactile_encoder.freeze_backbone=false`를 주면 전체 backbone을 fine-tuning할 수 있다. 이 경우 VRAM 요구량이 크게 달라지므로 기본 batch128의 fine-tuning 메모리 적합성은 별도 확인이 필요하다.

## Cache·checkpoint·monitoring

같은 frame도 window 시작점에 따라 reference가 달라진다. 따라서 Sparsh에서는 기존 FrozenFrameCache를 사용할 수 없다. Config validation, model factory, cache constructor, cached forward 경로에서 명시적으로 차단한다. Sensor mmap·timestamp·video decoder cache는 그대로 유지한다. Pair cache를 추후 만든다면 current와 reference ID를 **둘 다** key에 넣어야 한다.

새 checkpoint에는 source revision/checksum/경로/strict load 결과와 pairing stride, channel order, pooling, preprocessing version·geometry를 저장한다. ImageNet↔Sparsh resume와 다른 Sparsh input protocol로의 resume는 dataset 초기화 전에 거부한다. 기존 ImageNet checkpoint의 예전 config에는 `backbone_type`이 없으므로 ImageNet으로 해석해 호환성을 유지한다.

Training 시작 시 실제 batch, backbone chunk, N, frozen/trainable counts와 전체 identity를 콘솔·`metrics.jsonl`의 `tactile_backbone` record·W&B summary·trainctl 시작 메시지에 기록한다. Rich dashboard/기존 validation/alignment/UMAP·similarity 코드의 인터페이스는 유지했다.

## 검증 결과와 한계

- Pairing: N=1,3,5,7,10,20의 모든 current/reference index·6채널 shape·입력 불변성·N=1 동일 RGB·N=7 마지막 pair를 검증했다. Chunk가 batch 경계를 걸쳐도 explicit pairing 결과와 일치한다.
- 공식 checkpoint의 174개 key가 strict load됐다. 공식 reference code와 CPU FP32 token 최대 차이는 patch/register 모두 **6.91414e-6**였다. Reference의 einops grid flatten만 torch.flatten으로 제공했고 upstream training dependencies는 설치하지 않았다.
- 실제 GPU 0(RTX 3090, `GPU-d22c4326-fe30-9ba2-8cdc-37601e9587a7`)에서 N=1/5/20 forward가 finite였고 embedding shape는 `(B,256)`이었다.
- 서로 다른 sequence의 실제 Train window 4개에서 forward/backward, frozen backbone gradient=0, pooling/projection/Temporal/position/Force·Pressure MLP/codebook/decoder의 nonzero gradient를 확인했다. NCE→codebook 직접 gradient가 없는 기존 STE routing도 확인했다.
- Unfrozen 실제 window 2개에서 backbone gradient norm **0.838574**를 확인했다. AMP forward도 finite였다.
- Checkpoint save/strict reload 후 embedding 차이는 **0**이었다. 실제 기존 ImageNet checkpoint를 Sparsh로 resume하는 것도 거부했다. 기존 alignment retrieval은 실제 작은 batch에서 작동했다.
- GPU smoke는 약 **11.9초**, peak allocated 약 **1.24 GiB**, **optimizer update 0회**다. 이는 본학습 메모리/속도나 정확도 검증이 아니다.
- 최종 **77개 unit/공용 회귀 테스트가 모두 통과**했다. 결과는 `outputs/sparsh_validation/tests.json`과 `tests.log`에 기록한다. ImageNet·sampler 속도·timestamp/sensor cache·평가 worker·offline W&B·dashboard·기존 visualization 테스트를 포함한다.
- 초기 W&B 검증은 샌드박스 IPC 제한으로 실패했고 로컬 IPC를 허용한 offline 실행에서 통과했다. 통합 테스트를 stdin에서 시작했을 때 spawn이 `<stdin>`을 찾지 못한 실행 방식 오류도 표준 `python -m unittest` 실행으로 수정했다. 이 과정에서 공용 테스트를 우회하거나 fallback을 넣지 않았다.
- 미검증: 실제 장기 학습 및 일반화, 실제 optimizer update 후 trainer resume 전체 루프, full batch128/N=20 메모리·throughput, 다른 native 해상도의 혼합 sensor batch, upstream xFormers/AMP의 bitwise 동등성, 실제 Sparsh embedding의 대규모 UMAP, 원격 W&B/Discord 수신. 성능 향상을 이미 달성했다고 주장하지 않는다.

## 실행 명령

Repository 루트에서 실행한다. 다음 schema 명령은 실제로 실행해 확인했다.

```bash
uv run python -m script.train_adapter \
  --config-name=taf_adapter_sparsh --cfg job
```

N=5 기본 학습(물리 GPU 0의 확인된 UUID 사용):

```bash
CUDA_VISIBLE_DEVICES=GPU-d22c4326-fe30-9ba2-8cdc-37601e9587a7 \
uv run python -m script.train_adapter \
  --config-name=taf_adapter_sparsh
```

GPU 번호 방식의 `CUDA_VISIBLE_DEVICES=0`보다 UUID가 물리 GPU 0을 확실하게 지정한다. 기본 batch128/AdamW lr1e-4/dropout0.2/weight decay0.001/epoch-shuffled sampler/10,000 updates/AMP=true는 기존 alignment 설정을 유지했다. 온전한 모든 N에서 메모리가 부족하면 공통 batch size를 함께 조정해야 한다.

N=5 Rich dashboard + trainctl/Discord + W&B:

```bash
bash scripts/train_sparsh_windows.sh 5
```

N=1,3,5,7,10,20을 **하나씩 순차** 실행:

```bash
bash scripts/train_sparsh_windows.sh
```

실행하지 않고 명령만 확인:

```bash
bash scripts/train_sparsh_windows.sh --dry-run
```

스크립트가 nvidia-smi의 GPU 0 UUID를 읽어 고정하고, 모든 Python과 trainctl 실행을 uv로 시작한다. 여섯 명령을 실제 dry-run으로 검증했으며 **학습은 자동 시작하지 않았다**. 각 N에서 10,000 updates를 실행하는 명령이므로 현재 작업의 smoke와는 별개의 사용자 실행 작업이다.

현재 manifest prefix는 **06dc93e7**이며 N=5의 경로는 다음과 같다.

```text
outputs/comparison/sparsh_runs/06dc93e7_N5_seed42/
  metrics.jsonl
  step_00001000.pt ...
  last.pt
  alignment/
  wandb/
```

다른 N은 suffix의 N만 달라진다. trainctl supervisor의 실제 로그 위치는 기존 monitor가 run log에서 읽어 표시한다. 기존 comparison run 경로는 덮어쓰지 않으며, 새 경로에 결과가 이미 있으면 기존 학습 코드가 명시적인 resume를 요구한다.

검증을 재실행할 때는 다음 명령을 쓴다(optimizer update 없음).

```bash
CUDA_VISIBLE_DEVICES=GPU-d22c4326-fe30-9ba2-8cdc-37601e9587a7 \
uv run python -m script.check_sparsh
```

## 이전 전용 진단 정리

삭제 전 실제 `/proc` 명령 인자를 조사했고 실행 중인 전용 진단 프로세스는 **0개**였다. Repository 내부의 명시된 전용 경로인지 확인한 뒤 삭제했다.

- 전용 Python: `script/diagnose_alignment.py`, `script/diagnose_alignment_phase2.py`, `script/summarize_alignment_diagnostics.py`, `script/summarize_alignment_phase2.py`.
- 전용 runner: `script/run_alignment_diagnostics.sh`, `script/run_alignment_diagnostics_phase2.sh`.
- 전용 config: `config/alignment_diagnostics.yaml`, `config/alignment_diagnostics_phase2.yaml`.
- 전용 test: `tests/test_alignment_diagnostics.py`, `tests/test_alignment_phase2.py`.
- 전용 출력 두 디렉터리 및 해당 bytecode: 기존 `outputs/alignment_diagnostics/`, `outputs/alignment_diagnostics_phase2/` 전체.

요청에 있던 `summarize_alignment_diagnostics_phase2.py`는 존재하지 않았으며, 실제 파일명인 `summarize_alignment_phase2.py`를 삭제했다. 총 **18개 경로/514개 파일**, 논리 크기 **8,741,277,376 bytes**, 파일 할당량 기준 **8,742,596,608 bytes(약 8.74 GB)**를 삭제했다. 새 공식 weight와 검증 checkpoint를 포함한 순감소는 약 8.55 GB다. 상세 목록과 크기는 `outputs/sparsh_validation/cleanup.json`에 있다.

원본 dataset/manifest/gallery/정규화 파일/정식 TaF checkpoint, ImageNet 구현, epoch-shuffled 및 sampler 속도 개선, Rich/trainctl/W&B, sensor mmap/timestamp/video cache, 공용 검증·visualization 코드는 보존했다. 사용 package들은 기존 공용 경로에서도 필요하므로 제거하지 않았다. 원격 W&B나 WAH 프로젝트는 수정하지 않았다. 삭제된 전용 모듈을 import하는 공용 소스는 남아 있지 않다.

## 변경 파일

| 파일 | 변경 목적 |
| --- | --- |
| `model/sparsh_backbone.py`, `model/sparsh.py` | 공식 Small 구조·strict loader·bounded preprocessing·spatial pooling |
| `model/tactile_encoder.py`, `model/taf_adapter.py` | backbone 선택과 기존 Temporal head 재사용, cache 경로 차단 |
| `utils/config.py`, `utils/common.py`, `utils/model_factory.py` | structured Sparsh schema와 config validation |
| `utils/checkpoint.py`, `utils/frame_cache.py`, `utils/trainctl.py` | identity/호환성 guard·cache 거부·시작 메시지 |
| `script/train_adapter.py` | native 이미지 전달, 초기 resume guard와 metadata logging |
| `script/check_model.py`, `script/dynamics_experiment.py`, `script/evaluate_validation_sampling.py`, `script/profile_throughput.py` | native image_size=null 호환 |
| `config/taf_adapter_sparsh.yaml` | 기존 설정을 상속하는 Sparsh 설정 |
| `tests/test_sparsh.py`, `script/check_sparsh.py` | 단위 및 실제 공식-weight/데이터 smoke |
| `scripts/train_sparsh_windows.sh` | GPU 0의 순차 Rich/trainctl 실행과 dry-run |
| `README.md`, 본 문서, `docs/licenses/` | 사용법·차이·검증·cleanup·원문 라이선스 |
