# CLAUDE.md

이 파일은 이 저장소에서 작업할 때 Claude Code(claude.ai/code)에게 지침을 제공합니다.

## 프로젝트 개요

BBSplat(BillBoard Splatting)은 3D Gaussian을 텍스처가 있는 빌보드 프리미티브(학습 가능한 RGB 텍스처와 알파맵을 가진 평면 쿼드)로 대체하는 새로운 시점 합성 방법입니다. 3DGS/2DGS 코드베이스를 기반으로 하며, Mip-NeRF-360, Tanks&Temples, DTU 데이터셋을 대상으로 합니다. 이 포크는 **PlanarSnap** 프로젝트(빌보드 노멀 추출을 활용한 평면 분할)를 위해 여러 버그 수정과 학습 품질 개선이 적용되었습니다.

## 빌드 / 설치

```bash
# CUDA 서브모듈 설치 (클론 후 또는 CUDA 코드 수정 시 실행)
bash bbsplat_install.sh
# 아래와 동일:
pip install ./submodules/diff-bbsplat-rasterization
pip install ./submodules/simple-knn
```

`submodules/diff-bbsplat-rasterization/` 내 `.cu` 또는 `.h` 파일 수정 후에는 서브모듈을 재설치해야 합니다.

## 주요 명령어

**학습:**
```bash
python train.py -s <colmap_데이터셋_경로> --model_path=<출력_경로> \
  --cap_max=160_000 --max_read_points=150_000 --add_sky_box --eval \
  --densify_until_iter 10000
```

**렌더링 / 평가:**
```bash
python render.py -m <모델_경로> -s <데이터셋_경로> --skip_mesh
python metrics.py -m <모델_경로>
```

**인터랙티브 시각화:**
```bash
python visualize.py -m <모델_경로> -s <데이터셋_경로>
```

## 데이터셋별 학습 파라미터

`scripts/train_all.sh` 참조 (정식 레퍼런스):

| 데이터셋 유형 | `--cap_max` | `--max_read_points` | 추가 플래그 |
|---|---|---|---|
| Mip-NeRF-360 실내 (room, bonsai, counter, kitchen) | 160,000 | 150,000 | `--add_sky_box --eval` |
| Mip-NeRF-360 실외 (bicycle, stump, garden) | 300,000 | 290,000 | `--add_sky_box --eval` |
| Tanks & Temples | 300,000 | 290,000 | `--add_sky_box --eval` |
| DTU | 60,000 | 60,000 | `--lambda_normal=0.05 --lambda_dist 100 --eval` |

모든 씬에 `--densify_until_iter 10000` 추가 권장 (아래 학습 품질 섹션 참조).

## 아키텍처

### 학습 파이프라인 (`train.py`)
1. `Scene` → `GaussianModel`을 통해 COLMAP 씬 로드
2. 매 iter: 랜덤 카메라 샘플 → `render()` → 포토메트릭 손실 + 정규화 → optimizer step + MCMC noise injection
3. 3단계 텍스처 학습: 지오메트리(xyz, scale, rotation, SH)가 먼저 학습됨; 텍스처 α와 RGB는 `texture_from_iter=500`에 활성화되고 `densify_until_iter`(권장 10000)에 LR 감소 시작, `texture_to_iter=30000`에 완전 비활성화
4. MCMC 기반 densification(`relocate_gs` / `add_new_gs`)이 기존 3DGS 적응형 densification을 대체

### 핵심 모듈

**`scene/gaussian_model.py` — `GaussianModel`**  
모든 학습 가능 파라미터를 `nn.Parameter`로 보유:
- `_xyz`, `_scaling` (2D, log-space), `_rotation` (쿼터니언), `_features_dc/_rest` (SH 계수)
- `_texture_alpha` [N, 1, H, W] — 빌보드별 학습 가능한 불투명도 마스크
- `_texture_color` [N, 3, H, W] — 빌보드별 학습 가능한 RGB 텍스처
- 텍스처는 저장 시 float32 raw 파라미터로 저장되고, 로드 시 자동 감지 (float32 = 새 형식, uint8 = 레거시 형식)
- `activate_texture_training()` / `deactivate_texture_training()`이 텍스처 LR을 0으로 게이팅

**`gaussian_renderer/__init__.py` — `render()`**  
`diff_bbsplat_rasterization.GaussianRasterizer`를 래핑. 손실에 사용되는 `render`, `rend_dist`, `rend_normal`, `surf_normal`, `impact` 텐서를 반환.

**`scene/__init__.py` — `Scene`**  
데이터셋 형식(COLMAP `sparse/`, Blender `transforms_train.json`, NeILF `sfm_scene.json`)을 감지하여 카메라와 포인트 클라우드를 초기화. `add_sky_box`는 원거리 구형 포인트를 추가.

**`arguments/__init__.py`**  
모든 하이퍼파라미터가 `OptimizationParams`, `ModelParams`, `PipelineParams`의 기본값으로 여기에 존재. 기본 iterations: 32,000. 텍스처 학습 윈도우: iter 500~30,000.

### CUDA 레스터라이저 (`submodules/diff-bbsplat-rasterization/cuda_rasterizer/auxiliary.h`)

Python 변경 없이 동작을 바꾸는 컴파일 타임 플래그:
```c
#define FAST_INFERENCE 0   // 1로 설정 시 약 2배 빠른 렌더링 (약간의 품질 저하)
#define TILE_SORTING 0     // 1로 설정 시 Blender 내보내기 (StopThePop 정렬)
#define PIXEL_RESORTING 0  // TILE_SORTING과 함께 1로 설정 시 Blender 내보내기
```
변경 후 `bash bbsplat_install.sh` 실행 필요.

### 손실 구성 (`train.py:100–122`)
```
total_loss = L1 + λ_dssim*(1-SSIM) + λ_dist*dist_loss + λ_normal*normal_loss
           + λ_texture_value*texture_color_reg + λ_alpha_value*alpha_reg
           + opacity_reg*mean(texture_alpha)
```
`lambda_normal`과 `lambda_dist`는 각각 iter 7000과 3000 이후에만 활성화됨 (이전에는 0).

## 출력 구조

학습된 모델은 `--model_path` 아래 저장:
- `point_cloud/iteration_N/point_cloud.ply` — 빌보드 지오메트리 (xyz, scale, rotation, SH)
- `point_cloud/iteration_N/texture_alpha.npz`, `texture_color.npz` — float32 텍스처 (raw 파라미터)
- `cfg_args` — 직렬화된 학습 인자 (`render.py`가 다시 읽음)

## 학습 품질 개선 사항

### Float32 텍스처 저장/불러오기 (핵심 버그 수정)

**문제**: 원본 코드는 텍스처를 uint8 델타 인코딩으로 저장. `(sigmoid(raw) - alpha_init + 1) * 0.5 * 255`로 저장 시 `-inf` 로짓(완전 투명)이 작은 양수값(~0.004 alpha)으로 변환되어, 많은 빌보드에서 미세한 밝기 기여가 누적 → **9~13 dB PSNR 손실** + 갈색으로 뿌연 아티팩트.

**수정**: `save_texture()`가 raw float32 파라미터를 직접 저장. `load_texture()`는 dtype 자동 감지(float32 = 새 형식, uint8 = 레거시).

```python
# scene/gaussian_model.py — save_texture()
def save_texture(self, folder_path):
    ta_f32 = self._texture_alpha.detach().cpu().numpy().astype(np.float32)
    tc_f32 = self._texture_color.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(folder_path, "texture_alpha.npz"), texture_alpha=ta_f32)
    np.savez_compressed(os.path.join(folder_path, "texture_color.npz"), texture_color=tc_f32)
```

### Scaling/Rotation LR 지수 감소

수렴 이후 지오메트리 드리프트를 방지하기 위해 scaling과 rotation에 지수 감소 스케줄러 추가 (`scene/gaussian_model.py` lines 258–263):

```python
self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
                                                lr_final=training_args.scaling_lr * 0.1,
                                                max_steps=training_args.position_lr_max_steps)
self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                  lr_final=training_args.rotation_lr * 0.1,
                                                  max_steps=training_args.position_lr_max_steps)
```

`update_learning_rate()`에서 매 step 업데이트:
```python
elif param_group["name"] == "scaling":
    param_group['lr'] = self.scaling_scheduler_args(iteration)
elif param_group["name"] == "rotation":
    param_group['lr'] = self.rotation_scheduler_args(iteration)
```

### `densify_until_iter` 권장값: 10,000 + 3단계 수정 체인 (핵심 수정)

**관찰 (v9, v10 학습, room scene)**:
- iter 10,000: PSNR ~26 dB ← 최고 성능
- iter 32,000: PSNR ~19-20 dB ← 하락

v9~v13의 체계적 진단을 통해 3단계 수정이 적용됨:

**수정 1 (v11)** — `opacity_reg`를 MCMC 종료 후 0으로 전환:
```python
current_opacity_reg = opt.opacity_reg if iteration < opt.densify_until_iter else 0.0
total_loss += current_opacity_reg * gaussians.get_texture_alpha.mean()
```
`opacity_reg`는 MCMC가 "죽은" 빌보드를 detect/relocate하기 위해서만 필요. MCMC 없이 계속 실행하면 alpha를 파괴적으로 감소시킴. → **PSNR: 여전히 18.5 dB로 하락**

**수정 2 (v12)** — densification 종료 후 geometry(xyz/scaling/rotation) 동결:
```python
if iteration >= opt.densify_until_iter:
    for param_group in gaussians.optimizer.param_groups:
        if param_group["name"] in ("xyz", "scaling", "rotation"):
            param_group['lr'] = 0.0
```
→ **PSNR: 여전히 18.3 dB로 하락. 단, iter 30000(texture 비활성화 시점)에서 하락이 멈춤 → 텍스처가 주범임을 확인**

**수정 3 (v13)** — 텍스처도 `densify_until_iter`에 동결, 이후 SH(f_dc/f_rest)만 학습:
```python
effective_texture_end = min(opt.texture_to_iter, opt.densify_until_iter)
if opt.texture_from_iter <= iteration < effective_texture_end:
    gaussians.activate_texture_training()
if iteration >= effective_texture_end:
    gaussians.deactivate_texture_training()
```
원인: `activate_texture_training()`이 매 iter 텍스처 LR을 초기값으로 **리셋**. MCMC 종료 후에도 고정 LR로 업데이트되면 수렴한 텍스처가 진동하여 품질 저하.

**v13 결과 (room scene)**:
| Iter | Test PSNR |
|------|-----------|
| 10000 | 26.11 dB |
| 15000 | 26.22 dB |
| 20000 | 26.28 dB |
| 32000 | **26.34 dB** |

렌더링 최종 metrics: PSNR **26.22** / SSIM **0.8298** / LPIPS **0.2999**  
PSNR이 iter 10000 이후 지속 상승 — 하락 문제 완전 해결.

**버전별 비교 (room scene, Test PSNR)**:
| Iter | v9 | v11 | v12 | v13 |
|------|-----|------|------|------|
| 10000 | 25.93 | 25.97 | 26.01 | 26.11 |
| 20000 | — | 23.01 | 22.79 | 26.28 |
| 32000 | 19.70 | 18.53 | 18.34 | **26.34** |

## CUDA 12.8 / WSL2 호환성 수정

다음 변경사항들이 `train.py`에 적용되어 있습니다. **되돌리지 마세요 — WSL2에서의 조용한 학습 실패를 방지합니다.**

### 1. CUDA 메모리 할당자 (`train.py` line 15)
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
```
densification `empty_cache()` 호출 후 CUDA 메모리 단편화 방지. 미적용 시 WSL2 WDDM 재할당이 30~145초 backward pass 스파이크를 유발.

### 2. 이미지 크기 16의 배수 정렬 (`train.py` lines 92–93)
```python
viewpoint_cam.image_width  = viewpoint_cam.image_width  - (viewpoint_cam.image_width  % 16)
viewpoint_cam.image_height = viewpoint_cam.image_height - (viewpoint_cam.image_height % 16)
```
CUDA 12.8 타일 기반 레스터라이저에 필요. 미적용 시 16의 배수가 아닌 크기에서 렌더링 크래시 또는 엣지 아티팩트("칼자국 시임") 발생. GT 이미지는 line 105에서 크롭하여 맞춤.

### 3. XYZ 및 scaling 클램핑 (`train.py` lines 88–90)
```python
with torch.no_grad():
    gaussians._xyz.data = torch.clamp(gaussians._xyz.data, min=-10.0, max=10.0)
    if hasattr(gaussians, '_scaling'):
        gaussians._scaling.data = torch.clamp(gaussians._scaling.data, min=-15.0, max=5.0)
```
MCMC noise injection을 통한 NaN/Inf 전파 방지. 미적용 시 파라미터 폭주로 갈색/검은색 blob 아티팩트 및 학습 크래시 발생.

### 4. MCMC noise에 고정 opacity (`train.py` line 212)
```python
opacity = torch.ones([gaussians._xyz.shape[0], 1], dtype=torch.float32, device="cuda")
```
논문의 실제 구현에 맞게 noise injection에 opacity를 1로 고정. 주석 처리된 대안(texture_alpha 기반 opacity)은 낮은 alpha 프리미티브에서 과도하게 큰 noise를 유발.

### 5. 메모리 관리 추가
- 매 평가/저장 후 `gc.collect()` + `torch.cuda.empty_cache()` (line 171–172)
- 매 densification 단계 후 `gc.collect()` + `torch.cuda.empty_cache()` (line 195–196)
- 매 20 iterations마다 `gc.collect()` (line 222)

WSL2에서 Python GC 사이클과 CUDA 메모리 단편화로 인한 점진적 속도 저하 방지.

## WSL2 / WDDM 학습 성능

Windows WSL2(WDDM 모드)에서는 GPU 타임아웃 스파이크를 피하기 위해 다음 OS 수준 수정이 필요합니다:

### Windows 레지스트리 — TDR 설정 (PowerShell 관리자 권한)
```powershell
# GPU 타임아웃 60초로 증가
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
# TDR 완전 비활성화
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrLevel /t REG_DWORD /d 0 /f
```
재부팅 필요. 미설정 시 기본 2초 TDR이 긴 CUDA 커널 실행 중 GPU 드라이버 리셋을 유발 (iter ~1100–1200에서 9~153초 스파이크로 나타남).

### 하드웨어 가속 GPU 스케줄링 (HAGS)
비활성화: 설정 → 시스템 → 디스플레이 → 그래픽 → 기본 그래픽 설정 변경 → 하드웨어 가속 GPU 스케줄링 → **끄기**

재부팅 필요. HAGS 활성화 시 WSL2 CUDA가 Windows 디스플레이 스케줄러와 경쟁하여 간헐적 GPU 선점 발생.
