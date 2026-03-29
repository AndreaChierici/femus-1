# FEMuS Nonlocal Ex10 — OpenMP Offloading on MI300A: Technical Report

**Date:** 2026-03-27
**System:** AMD MI300A APU (gfx942)
**Working repo:** `~/repos/femus-1`

---

## 1. Automatic Sparsity Pattern Estimate

**File:** `ex10.cpp` (inside `main()`, after mesh setup)

### 1.1 Problem

The nonlocal operator couples every DOF with all other DOFs within an
interaction ball of radius `R = delta1 + eps`.  PETSc sparse matrices must be
preallocated with enough nonzeros per row to hold these couplings.  A value
that is too small causes HIPSPARSE (the GPU sparse-matrix backend) to fail with
a memory access fault, since it cannot dynamically grow the matrix.  A value
that is too large wastes memory — which is critical on MI300A's 128 GB HBM,
where overallocation can trigger out-of-memory kills or catastrophic swap
behavior.

Previously, the sparsity pattern size was hardcoded (e.g., `40000u` or
`50000u`).  This breaks when mesh refinement levels change, because the actual
number of DOFs inside the interaction ball depends on the DOF density (which
grows with refinement) and the interaction radius (which shrinks with
refinement).

### 1.2 Geometric Estimate

The number of nonzeros per row equals the expected number of DOFs inside a ball
of radius `R`:

```
nnz_per_row ≈ BallVolume(R) × (totalDOFs / domainVolume)
```

where:

| Symbol                  | Value / Formula                                | Source                                 |
|-------------------------|------------------------------------------------|----------------------------------------|
| `delta1`                | 0.2 (hardcoded)                                | `nonlocal_assembly_adaptive.hpp:367`   |
| `eps`                   | `0.0125 × (2/3)^numberOfUniformLevels`         | `nonlocal_assembly_adaptive.hpp:494–495` |
| `numberOfUniformLevels` | `argv[1]` (runtime)                            | `ex10.cpp`                             |
| `totalDOFs`             | queried from mesh at finest level              | `Mesh::GetTotalNumberOfDofs(1)`        |
| `domainVolume`          | product of coordinate ranges at finest level   | `Mesh::_topology->_Sol[k]->min/max()`  |
| `dim`                   | mesh dimension (2 or 3)                        | `MultiLevelMesh::GetDimension()`       |

Ball volume formula:

- 2D: `BallVolume = π × R²`
- 3D: `BallVolume = (4/3) × π × R³`

The raw estimate is then scaled with a 1.5× safety factor:

```
sparsitySize = ceil(1.5 × raw_estimate)
```

### 1.3 Safety Factor Justification

The 1.5× factor accounts for:

- **Bounding-box overcount**: The coarse intersection test in the assembly uses
  axis-aligned bounding boxes, which include ~27% more area than the actual ball
  (ratio 4/π ≈ 1.27 in 2D).
- **DOF distribution non-uniformity**: Serendipity elements have corner and
  edge-midpoint nodes at slightly irregular spacing.
- **Boundary effects**: The estimate must cover the worst case (fully interior
  DOFs that see the complete ball).

### 1.4 PetscInt Overflow Guard

When PETSc is compiled with 32-bit `PetscInt` (the default), the product
`localRows × nnz_per_row` must stay below 2^31.  The estimate is clamped:

```
localRows = ceil(totalDOFs / nprocs)
maxSafe   = floor(2^31 / localRows)

if sparsitySize > maxSafe:
    sparsitySize = maxSafe    // prints a warning
```

If clamping occurs, the matrix is under-allocated and the solve will likely
fail.  The remedy is either to build PETSc with `--with-64-bit-indices` or to
use more MPI ranks (which reduces `localRows`).

### 1.5 Example Values

| Case (argv) | totalDOFs | R      | raw_nnz | with_safety | maxSafe (4 ranks) |
|-------------|-----------|--------|---------|-------------|-------------------|
| `3 2 0 1`   | ~37,665   | 0.2033 | 4,888   | 7,332       | 57,902            |
| `4 3 0 1`   | ~148,353  | 0.2025 | 19,102  | 28,654      | 57,902            |
| `5 3 0 1`   | ~590,337  | 0.2017 | 75,624  | 113,437     | 14,558 (clamped!) |

### 1.6 Performance Impact

PETSc preallocates `localRows × nnz_per_row` entries per matrix block.  Each
entry costs 12 bytes (8-byte `double` value + 4-byte `int` column index).  For
the "4 3 0 1" case (148,353 DOFs, 4 MPI ranks, localRows ≈ 37,088):

| Estimate               | d_nnz  | o_nnz  | Memory / rank | Total (4 ranks) |
|------------------------|--------|--------|---------------|-----------------|
| Old hardcoded (50,000) | 37,088 | 50,000 | 38.8 GB       | **155 GB**      |
| New 1.5× (28,654)     | 28,654 | 28,654 | 25.5 GB       | **102 GB**      |
| Hypothetical exact     | 19,100 | 19,100 | 17.0 GB       | **68 GB**       |

**Time cost**: The setup phase zeroes the preallocated memory.  At ~100 GB/s
effective bandwidth per NUMA node:

| Estimate               | Alloc + zero time | Wasted vs. exact |
|------------------------|-------------------|------------------|
| Old hardcoded (50,000) | ~1.5 s            | +0.9 s           |
| New 1.5× (28,654)     | ~1.0 s            | +0.3 s           |
| Hypothetical exact     | ~0.7 s            | —                |

The solve phase is **unaffected**: after `MatAssemblyEnd()`, PETSc compresses
the matrix to actual nonzeros.  SpMV and the Krylov iteration only touch real
entries.

The timing difference is negligible (well under 1% of total wall-clock time,
which is dominated by the assembly loop — typically several minutes).  But the
**memory difference is critical**: the old hardcoded value preallocated 155 GB
across 4 ranks, exceeding MI300A's 128 GB HBM, which forces swap or OOM-kill.
The new geometric estimate fits within HBM.

### 1.7 Runtime Diagnostics

The code prints the computed values at startup:

```
>>> Sparsity estimate: totalDOFs=148353  R=0.2025  ballVol=0.1289
    domainVol=1.0  raw_nnz=19102  with_safety=28654
```

---

## 2. Adaptive GPU/CPU Threshold (`MIN_GPU_WORK`)

**File:** `include/NonLocal.hpp`, inside `ProcessTasks_GPU()`

### 2.1 Problem

For each source element, the code decides whether to offload work to the GPU or
run on the CPU.  The original decision used a hardcoded threshold:

```cpp
const unsigned MIN_GPU_WORK = 10000;
if (totalJelWork < MIN_GPU_WORK) {
    ProcessTasks_CPU(...);
    return;
}
```

`totalJelWork` is the total number of (task, element) pairs for the current
source element.  The problem is that the **work per item** varies enormously
with the element type, so a single threshold is only correct for one element
type (it happened to be roughly right for 2D serendipity quads).

### 2.2 Work Per Item Analysis

Each work item in the GPU kernel executes a loop over `nDof2` shape functions
and `nGauss2` Gauss points.  Each Gauss-point iteration involves a distance
computation (~5×dim FLOPs), a smoothstep evaluation (~12 FLOPs), and DOF
accumulations (~2×nDof2 FLOPs).  Estimated FLOPs per work item:

```
flopsPerItem = nDof2 × nGauss2 × (5 × dim + 12 + 2 × nDof2)
```

| Element type       | dim | nDof2 | nGauss2 | FLOPs/item | Old threshold OK?            |
|--------------------|-----|-------|---------|------------|------------------------------|
| P1 triangle        | 2   | 3     | 3       | 252        | Too low (needs ~120K)        |
| Q1 quad            | 2   | 4     | 4       | 512        | Too low (needs ~59K)         |
| P2 triangle        | 2   | 6     | 7       | 1,428      | Too low (needs ~21K)         |
| **Q8 serendipity** | 2   | 8     | 9       | **2,736**  | **About right (~11K)**       |
| Q9 biquadratic     | 2   | 9     | 9       | 3,078      | Slightly high                |
| Hex27 (3D)         | 3   | 27    | 27      | 59,049     | Way too high (needs ~500)    |

### 2.3 Adaptive Threshold

GPU kernel launch overhead on MI300A is approximately 30 μs.  At ~1 TFLOP/s
effective throughput (double precision, accounting for memory bandwidth), this
equals ~30 million FLOPs.  The breakeven number of work items:

```
minGPUWork = 30,000,000 / flopsPerItem
```

with a floor of 128 (one thread team — the minimum meaningful GPU dispatch).

Implementation:

```cpp
unsigned nDof2_est   = (nElem > 0) ? region2.GetDofNumber(0) : 8;
unsigned nGauss2_est = (nElem > 0)
    ? region2.GetFem(0)->GetGaussPointNumber() : 9;
unsigned flopsPerItem = nDof2_est * nGauss2_est
    * (5 * dimSpace + 12 + 2 * nDof2_est);
unsigned minGPUWork = 30000000u / std::max(flopsPerItem, 1u);
minGPUWork = std::max(minGPUWork, 128u);
```

### 2.4 Resulting Thresholds

| Element type        | Adaptive threshold | Old threshold | Effect                           |
|---------------------|--------------------|---------------|----------------------------------|
| P1 triangle (2D)    | 119,048            | 10,000        | Avoids wasteful GPU launches     |
| Q1 quad (2D)        | 58,594             | 10,000        | Avoids wasteful GPU launches     |
| P2 triangle (2D)    | 21,008             | 10,000        | Modest increase, correct         |
| Q8 serendipity (2D) | 10,965             | 10,000        | Nearly identical                 |
| Q9 biquadratic (2D) | 9,747              | 10,000        | Nearly identical                 |
| Hex27 (3D)          | 508                | 10,000        | **20× lower — enables GPU for 3D** |

### 2.5 Dependencies on Runtime Parameters

The threshold adapts automatically through the element type, which is
determined by the input mesh and the FE order set in `ex10.cpp`:

- **`dim`** (2 or 3): from the mesh → heavier 3D items lower the threshold.
- **`nDof2`**: from the FE order (`SERENDIPITY` = 8, `FIRST` = 3–4,
  `SECOND` = 9–27).
- **`nGauss2`**: from the quadrature rule for the element type.

No explicit runtime argument is needed — the threshold self-tunes from the
problem's finite element discretization.  The runtime arguments like `3 2 0 1`
or `4 3 0 1` do not directly change the threshold.  They indirectly affect
whether the GPU path is taken by changing `totalJelWork` (finer meshes produce
more work per source element, making GPU offloading more likely).

### 2.6 Tuning the Launch Overhead Constant

The numerator `30,000,000` (30 M FLOPs) encapsulates the hardware-specific GPU
kernel launch cost.  If porting to a different GPU:

- **Higher launch overhead** (e.g., discrete GPU with PCIe data transfer):
  increase the constant → more conservative GPU offloading.
- **Lower launch overhead** (e.g., tightly-coupled APU with warm caches):
  decrease the constant → more aggressive GPU offloading.

30 M is appropriate for MI300A with `HSA_XNACK=1` (unified memory, no explicit
data transfer, but page-fault warmup cost).

### 2.7 Runtime Diagnostics

The code prints the computed threshold once on the first call:

```
>>> GPU/CPU threshold: minGPUWork=10965 (nDof2=8 nGauss2=9 dim=2 flops/item=2736)
```

It also prints running CPU/GPU call counts every 500 source elements:

```
>>> Offload stats so far: GPU=480 CPU=20
```

---

## 3. GPU Affinity via `omp_set_default_device()` and the `ROCR_VISIBLE_DEVICES` Problem

**File:** `ex10.cpp` (after `FemusInit`, before mesh setup)

### 3.1 The Problem with `ROCR_VISIBLE_DEVICES`

MI300A is an APU: the CPU cores and all 4 GPU Compute Dies (GCDs) share the
same physical HBM.  When `HSA_XNACK=1` is set, the GPU can access any
host-allocated memory through page faults — the entire address space is
unified.

The affinity scripts (`set_gpu_device_mi300a.sh` and `set_cpu_gpu_mi300a.sh`)
set:

```bash
export ROCR_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
```

This hides all GCDs except one from the HSA runtime.  On MI300A, this causes
problems:

1. **Breaks XNACK page-table initialization**: Only the single visible GCD
   gets proper page-table setup.  The unified address space that XNACK relies
   on is compromised.
2. **Crashes the OpenMP AMDGPU runtime plugin**: The plugin discovers and
   initializes devices at startup.  Restricting visibility mid-launch (via the
   wrapper script) causes it to fail with a segmentation fault during offload.
3. **Was designed for MI300X** (discrete GPU with separate HBM per device).  On
   MI300A's unified memory architecture, hiding devices from the HSA runtime
   undermines the shared address space.

The companion script `set_cpu_gpu_mi300a.sh` has the same
`ROCR_VISIBLE_DEVICES` issue, plus it sets `GOMP_CPU_AFFINITY` (a GCC-specific
variable that is not recognized by amdclang's LLVM OpenMP runtime).

### 3.2 In-Code GPU Affinity

Instead of hiding devices, the code selects the target GCD using the OpenMP
device API:

```cpp
#ifdef _OPENMP
{
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    omp_set_default_device(localRank);
}
#endif
```

How it works:

- `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` creates a communicator of all
  ranks on the same node.  The rank within this communicator (0, 1, 2, or 3) is
  the node-local rank.
- `omp_set_default_device(localRank)` tells the OpenMP runtime to direct all
  subsequent `#pragma omp target` regions to that GCD.
- **All 4 GCDs remain visible** to the HSA runtime, so XNACK page tables are
  properly initialized for the entire unified address space.

### 3.3 Compile-Time Guard

The `#include <omp.h>` and the `omp_set_default_device()` call are wrapped in
`#ifdef _OPENMP` / `#endif`.  The `_OPENMP` macro is defined automatically by
the compiler when `-fopenmp` is passed.  This means the code compiles and runs
correctly both with and without OpenMP:

- **With `-fopenmp`**: GPU affinity is set per rank.
- **Without `-fopenmp`**: The block is skipped; no OpenMP headers are needed.

### 3.4 CPU Affinity

CPU affinity (pinning each MPI rank to the NUMA node closest to its GCD) is
handled by OpenMPI's binding options rather than in-code:

```bash
mpirun -n 4 --map-by numa --bind-to numa ./NonLocal_ex10 3 2 0 1
```

On MI300A with 96 cores and 4 NUMA nodes:

| Rank | NUMA node | Cores  | Nearest GCD |
|------|-----------|--------|-------------|
| 0    | 0         | 0–23   | GCD 0       |
| 1    | 1         | 24–47  | GCD 1       |
| 2    | 2         | 48–71  | GCD 2       |
| 3    | 3         | 72–95  | GCD 3       |

**Do NOT use `set_gpu_device_mi300a.sh` or `set_cpu_gpu_mi300a.sh`** — they
set `ROCR_VISIBLE_DEVICES` which breaks OpenMP offloading on MI300A.

---

## 4. Build and Run

### 4.1 Environment

```bash
source ~/femus_env.sh
```

This loads the ROCm toolchain, sets `HSA_XNACK=1` for unified shared memory,
and points to the correct PETSc/SLEPc installation.

### 4.2 Build

```bash
cd ~/repos/femus-1/build
cmake -DCMAKE_CXX_FLAGS="-fopenmp --offload-arch=gfx942" ..
make -j
```

### 4.3 Run

```bash
cd ~/repos/femus-1/build/applications/NonLocal/ex10
mpirun -n 4 --map-by numa --bind-to numa ./NonLocal_ex10 4 3 0 1
```

### 4.4 Runtime Arguments

```
./NonLocal_ex10 <numberOfUniformLevels> <lmax1> <cutFem> <correctConstant>
```

| Argument                | Description                                     | Typical |
|-------------------------|-------------------------------------------------|---------|
| `numberOfUniformLevels` | Mesh refinement levels (argv[1])                | 3 or 4  |
| `lmax1`                 | Octree adaptive integration depth (argv[2])     | 2 or 3  |
| `cutFem`                | Use cut-FEM assembly, 0 or 1 (argv[3])          | 0       |
| `correctConstant`       | Apply constant correction, 0 or 1 (argv[4])     | 1       |

---

## 5. Summary of Files Changed

| File | Change |
|------|--------|
| `applications/NonLocal/ex10/ex10.cpp` | Automatic sparsity pattern estimate; in-code GPU affinity via `omp_set_default_device()` guarded by `#ifdef _OPENMP`. |
| `applications/NonLocal/ex10/include/NonLocal.hpp` | Adaptive `MIN_GPU_WORK` threshold based on element type; runtime diagnostic messages for GPU/CPU routing decisions. |
