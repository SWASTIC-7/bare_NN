# PTX ISA Quick Reference

> **PTX** (Parallel Thread Execution) is NVIDIA's low-level virtual ISA for GPU programming. This document provides a concise reference.

---

## 1. Directives

| Directive | Description | Example |
|-----------|-------------|---------|
| `.version` | PTX version | `.version 7.0` |
| `.target` | Target GPU architecture | `.target sm_80` |
| `.address_size` | Address size (32/64 bit) | `.address_size 64` |
| `.visible` | Symbol visible to host/other modules | `.visible .entry kernel` |
| `.entry` | Kernel entry point | `.entry myKernel(...)` |
| `.func` | Device function | `.func (.reg .f32 ret) add(...)` |
| `.param` | Parameter declaration | `.param .u64 ptr` |
| `.reg` | Register declaration | `.reg .f32 %f<16>;` |
| `.local` | Thread-local memory | `.local .f32 arr[64];` |
| `.shared` | Block-shared memory | `.shared .f32 sdata[256];` |
| `.global` | Global memory | `.global .f32 data[1024];` |
| `.const` | Constant memory | `.const .f32 weights[128];` |
| `.align` | Memory alignment | `.align 16` |
| `.extern` | External symbol | `.extern .shared .f32 dynshared[];` |

---

## 2. State Spaces (Memory Types)

| Space | Scope | Latency | Description |
|-------|-------|---------|-------------|
| `.reg` | Thread | Fastest | Registers (on-chip, per-thread) |
| `.sreg` | Thread | Fastest | Special registers (read-only) |
| `.local` | Thread | High | Thread-local memory (off-chip, cached) |
| `.shared` | Block | Low | Shared memory (on-chip, per-block) |
| `.global` | Grid | High | Global memory (off-chip, cached) |
| `.const` | Grid | Low | Constant memory (cached, read-only) |
| `.param` | Kernel | — | Kernel parameters |
| `.tex` | Grid | Low | Texture memory (cached, read-only) |

---

## 3. Data Types

| Type | Size | Description |
|------|------|-------------|
| `.pred` | 1 bit | Predicate (boolean) |
| `.b8`, `.b16`, `.b32`, `.b64` | 8/16/32/64 bits | Untyped bits |
| `.u8`, `.u16`, `.u32`, `.u64` | 8/16/32/64 bits | Unsigned integer |
| `.s8`, `.s16`, `.s32`, `.s64` | 8/16/32/64 bits | Signed integer |
| `.f16`, `.f16x2` | 16/32 bits | Half precision float |
| `.f32` | 32 bits | Single precision float |
| `.f64` | 64 bits | Double precision float |
| `.bf16`, `.bf16x2` | 16/32 bits | Brain float (ML) |
| `.tf32` | 32 bits | TensorFloat-32 (Tensor Cores) |

---

## 4. Special Registers

| Register | Description | Example |
|----------|-------------|---------|
| `%tid.x`, `%tid.y`, `%tid.z` | Thread ID within block | `mov.u32 %r1, %tid.x;` |
| `%ntid.x`, `%ntid.y`, `%ntid.z` | Block dimensions (blockDim) | `mov.u32 %r2, %ntid.x;` |
| `%ctaid.x`, `%ctaid.y`, `%ctaid.z` | Block ID within grid (blockIdx) | `mov.u32 %r3, %ctaid.x;` |
| `%nctaid.x`, `%nctaid.y`, `%nctaid.z` | Grid dimensions (gridDim) | `mov.u32 %r4, %nctaid.x;` |
| `%laneid` | Lane ID within warp (0–31) | `mov.u32 %r5, %laneid;` |
| `%warpid` | Warp ID within block | `mov.u32 %r6, %warpid;` |
| `%smid` | SM (Streaming Multiprocessor) ID | `mov.u32 %r7, %smid;` |
| `%clock`, `%clock64` | Cycle counter | `mov.u64 %rd1, %clock64;` |
| `%globaltimer` | Global nanosecond timer | `mov.u64 %rd2, %globaltimer;` |

---

## 5. Instruction Set

### 5.1 Data Movement

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `mov` | Move/copy value | `mov.u32 %r1, %r2;` |
| `ld` | Load from memory | `ld.global.f32 %f1, [%rd1];` |
| `st` | Store to memory | `st.global.f32 [%rd1], %f1;` |
| `ld.param` | Load kernel parameter | `ld.param.u64 %rd1, [param];` |
| `cvta` | Convert address to generic | `cvta.to.global.u64 %rd1, %rd2;` |
| `cvt` | Convert between types | `cvt.f32.s32 %f1, %r1;` |
| `shfl.sync` | Warp shuffle | `shfl.sync.down.b32 %r1, %r2, 1, 0x1f, 0xffffffff;` |

#### Load/Store Modifiers

| Modifier | Description |
|----------|-------------|
| `.global`, `.shared`, `.local`, `.const` | Memory space |
| `.ca` | Cache at all levels |
| `.cg` | Cache at L2, bypass L1 |
| `.cs` | Cache streaming (evict first) |
| `.lu` | Last use (hint for eviction) |
| `.cv` | Don't cache, volatile |
| `.volatile` | Volatile (no caching) |
| `.v2`, `.v4` | Vector load/store (2 or 4 elements) |

---

### 5.2 Integer Arithmetic

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `add` | Addition | `add.s32 %r1, %r2, %r3;` |
| `sub` | Subtraction | `sub.s32 %r1, %r2, %r3;` |
| `mul` | Multiplication (low bits) | `mul.lo.s32 %r1, %r2, %r3;` |
| `mul.hi` | Multiplication (high bits) | `mul.hi.s32 %r1, %r2, %r3;` |
| `mul.wide` | Widening multiply | `mul.wide.s32 %rd1, %r1, %r2;` |
| `mad` | Multiply-add | `mad.lo.s32 %r1, %r2, %r3, %r4;` |
| `div` | Division | `div.s32 %r1, %r2, %r3;` |
| `rem` | Remainder | `rem.s32 %r1, %r2, %r3;` |
| `abs` | Absolute value | `abs.s32 %r1, %r2;` |
| `neg` | Negate | `neg.s32 %r1, %r2;` |
| `min` | Minimum | `min.s32 %r1, %r2, %r3;` |
| `max` | Maximum | `max.s32 %r1, %r2, %r3;` |
| `clz` | Count leading zeros | `clz.b32 %r1, %r2;` |
| `popc` | Population count (1-bits) | `popc.b32 %r1, %r2;` |
| `bfind` | Find first set bit | `bfind.u32 %r1, %r2;` |
| `brev` | Bit reverse | `brev.b32 %r1, %r2;` |
| `bfe` | Bit field extract | `bfe.u32 %r1, %r2, 8, 4;` |
| `bfi` | Bit field insert | `bfi.b32 %r1, %r2, %r3, 8, 4;` |

---

### 5.3 Floating-Point Arithmetic

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `add` | Addition | `add.f32 %f1, %f2, %f3;` |
| `sub` | Subtraction | `sub.f32 %f1, %f2, %f3;` |
| `mul` | Multiplication | `mul.f32 %f1, %f2, %f3;` |
| `fma` | Fused multiply-add | `fma.rn.f32 %f1, %f2, %f3, %f4;` |
| `mad` | Multiply-add (may split) | `mad.f32 %f1, %f2, %f3, %f4;` |
| `div` | Division | `div.approx.f32 %f1, %f2, %f3;` |
| `rcp` | Reciprocal (1/x) | `rcp.approx.f32 %f1, %f2;` |
| `sqrt` | Square root | `sqrt.approx.f32 %f1, %f2;` |
| `rsqrt` | Reciprocal sqrt (1/√x) | `rsqrt.approx.f32 %f1, %f2;` |
| `abs` | Absolute value | `abs.f32 %f1, %f2;` |
| `neg` | Negate | `neg.f32 %f1, %f2;` |
| `min` | Minimum | `min.f32 %f1, %f2, %f3;` |
| `max` | Maximum | `max.f32 %f1, %f2, %f3;` |
| `ex2` | 2^x (fast) | `ex2.approx.f32 %f1, %f2;` |
| `lg2` | log₂(x) (fast) | `lg2.approx.f32 %f1, %f2;` |
| `sin` | Sine (fast) | `sin.approx.f32 %f1, %f2;` |
| `cos` | Cosine (fast) | `cos.approx.f32 %f1, %f2;` |

#### Rounding Modifiers

| Modifier | Description |
|----------|-------------|
| `.rn` | Round to nearest even |
| `.rz` | Round toward zero |
| `.rm` | Round toward −∞ |
| `.rp` | Round toward +∞ |
| `.approx` | Fast approximation |
| `.ftz` | Flush denormals to zero |
| `.sat` | Saturate result to [0,1] |

---

### 5.4 Comparison & Selection

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `setp` | Set predicate on comparison | `setp.lt.f32 %p1, %f1, %f2;` |
| `set` | Set register on comparison | `set.lt.f32.f32 %f1, %f2, %f3;` |
| `selp` | Select based on predicate | `selp.f32 %f1, %f2, %f3, %p1;` |
| `slct` | Select based on sign | `slct.f32.s32 %f1, %f2, %f3, %r1;` |

#### Comparison Operators

| Operator | Description |
|----------|-------------|
| `.eq` | Equal |
| `.ne` | Not equal |
| `.lt` | Less than |
| `.le` | Less than or equal |
| `.gt` | Greater than |
| `.ge` | Greater than or equal |
| `.lo`, `.ls`, `.hi`, `.hs` | Unsigned comparisons |
| `.equ`, `.neu`, `.ltu`, `.gtu`, etc. | Unordered (NaN-safe) |

---

### 5.5 Logic & Bitwise

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `and` | Bitwise AND | `and.b32 %r1, %r2, %r3;` |
| `or` | Bitwise OR | `or.b32 %r1, %r2, %r3;` |
| `xor` | Bitwise XOR | `xor.b32 %r1, %r2, %r3;` |
| `not` | Bitwise NOT | `not.b32 %r1, %r2;` |
| `shl` | Shift left | `shl.b32 %r1, %r2, 4;` |
| `shr` | Shift right (logical) | `shr.u32 %r1, %r2, 4;` |
| `shr` | Shift right (arithmetic) | `shr.s32 %r1, %r2, 4;` |
| `cnot` | Logical NOT (0→1, else→0) | `cnot.b32 %r1, %r2;` |

#### Predicate Logic

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `and` | Predicate AND | `and.pred %p1, %p2, %p3;` |
| `or` | Predicate OR | `or.pred %p1, %p2, %p3;` |
| `xor` | Predicate XOR | `xor.pred %p1, %p2, %p3;` |
| `not` | Predicate NOT | `not.pred %p1, %p2;` |

---

### 5.6 Control Flow

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `bra` | Unconditional branch | `bra TARGET;` |
| `@%p bra` | Conditional branch | `@%p1 bra LABEL;` |
| `@!%p bra` | Negated conditional | `@!%p1 bra ELSE;` |
| `call` | Function call | `call (%ret), func, (%arg1);` |
| `ret` | Return from function | `ret;` |
| `exit` | Exit thread | `exit;` |

---

### 5.7 Synchronization

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `bar.sync` | Block barrier (all threads) | `bar.sync 0;` |
| `bar.arrive` | Signal arrival at barrier | `bar.arrive 0, %r1;` |
| `bar.red` | Barrier with reduction | `bar.red.and.pred %p1, 0, %p2;` |
| `membar.cta` | Memory fence (block) | `membar.cta;` |
| `membar.gl` | Memory fence (global) | `membar.gl;` |
| `membar.sys` | Memory fence (system) | `membar.sys;` |
| `fence` | Memory fence | `fence.acq_rel.gpu;` |

---

### 5.8 Atomic Operations

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `atom.add` | Atomic add | `atom.global.add.f32 %f1, [%rd1], %f2;` |
| `atom.sub` | Atomic subtract | `atom.global.sub.u32 %r1, [%rd1], %r2;` |
| `atom.exch` | Atomic exchange | `atom.global.exch.b32 %r1, [%rd1], %r2;` |
| `atom.cas` | Compare and swap | `atom.global.cas.b32 %r1, [%rd1], %r2, %r3;` |
| `atom.min` | Atomic minimum | `atom.global.min.s32 %r1, [%rd1], %r2;` |
| `atom.max` | Atomic maximum | `atom.global.max.s32 %r1, [%rd1], %r2;` |
| `atom.and` | Atomic AND | `atom.global.and.b32 %r1, [%rd1], %r2;` |
| `atom.or` | Atomic OR | `atom.global.or.b32 %r1, [%rd1], %r2;` |
| `atom.xor` | Atomic XOR | `atom.global.xor.b32 %r1, [%rd1], %r2;` |
| `atom.inc` | Atomic increment (wrap) | `atom.global.inc.u32 %r1, [%rd1], %r2;` |
| `atom.dec` | Atomic decrement (wrap) | `atom.global.dec.u32 %r1, [%rd1], %r2;` |
| `red` | Reduction (no return) | `red.global.add.f32 [%rd1], %f1;` |

---

### 5.9 Warp-Level Operations

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `shfl.sync.up` | Shuffle up (lower lane → higher) | `shfl.sync.up.b32 %r1, %r2, 1, 0, 0xffffffff;` |
| `shfl.sync.down` | Shuffle down (higher → lower) | `shfl.sync.down.b32 %r1, %r2, 1, 0x1f, 0xffffffff;` |
| `shfl.sync.bfly` | Shuffle butterfly (XOR) | `shfl.sync.bfly.b32 %r1, %r2, 1, 0x1f, 0xffffffff;` |
| `shfl.sync.idx` | Shuffle indexed | `shfl.sync.idx.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;` |
| `vote.sync.all` | All threads predicate true | `vote.sync.all.pred %p1, %p2, 0xffffffff;` |
| `vote.sync.any` | Any thread predicate true | `vote.sync.any.pred %p1, %p2, 0xffffffff;` |
| `vote.sync.uni` | All threads same predicate | `vote.sync.uni.pred %p1, %p2, 0xffffffff;` |
| `vote.sync.ballot` | Ballot (bitmask of preds) | `vote.sync.ballot.b32 %r1, %p1, 0xffffffff;` |
| `match.sync.any` | Find matching values | `match.sync.any.b32 %r1, %r2, 0xffffffff;` |
| `redux.sync` | Warp reduce | `redux.sync.add.s32 %r1, %r2, 0xffffffff;` |

---

### 5.10 Type Conversion

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `cvt` | Convert types | `cvt.rn.f32.s32 %f1, %r1;` |
| `cvt.rni` | Round to nearest int | `cvt.rni.f32.f32 %f1, %f2;` |
| `cvt.rzi` | Truncate to int | `cvt.rzi.f32.f32 %f1, %f2;` |
| `cvt.sat` | Saturating convert | `cvt.sat.s8.s32 %r1, %r2;` |
| `cvta` | Address space conversion | `cvta.to.global.u64 %rd1, %rd2;` |

---

### 5.11 Texture & Surface

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `tex` | Texture fetch | `tex.1d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [tex1, {%f5}];` |
| `tld4` | Texture gather | `tld4.r.2d.v4.f32.f32 {...}, [tex1, {%f1,%f2}];` |
| `txq` | Texture query | `txq.width.b32 %r1, [tex1];` |
| `suld` | Surface load | `suld.b.1d.v4.b32 {...}, [surf1, {%r1}];` |
| `sust` | Surface store | `sust.b.1d.v4.b32 [surf1, {%r1}], {...};` |
| `suq` | Surface query | `suq.width.b32 %r1, [surf1];` |

---

### 5.12 Tensor Core (Matrix) Operations

| Instruction | Description | Syntax |
|-------------|-------------|--------|
| `wmma.load` | Load matrix fragment | `wmma.load.a.sync.aligned.m16n16k16.row.f16 {...}, [%rd1], %r1;` |
| `wmma.store` | Store matrix fragment | `wmma.store.d.sync.aligned.m16n16k16.row.f32 [%rd1], {...}, %r1;` |
| `wmma.mma` | Matrix multiply-accumulate | `wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32 {...}, {...}, {...}, {...};` |
| `mma` | Generic MMA | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...}, {...}, {...}, {...};` |
| `ldmatrix` | Load matrix for MMA | `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {...}, [%rd1];` |

---

## 6. Predicated Execution

Any instruction can be predicated (conditionally executed):

```ptx
setp.lt.f32 %p1, %f1, 0.0;      // Set predicate: p1 = (f1 < 0)
@%p1 neg.f32 %f1, %f1;          // Execute only if p1 is true
@!%p1 mov.f32 %f2, %f1;         // Execute only if p1 is false
```

---

## 7. Register Naming Convention

| Pattern | Description | Example |
|---------|-------------|---------|
| `%r<N>` | 32-bit registers | `%r0`, `%r1`, `%r15` |
| `%rd<N>` | 64-bit registers | `%rd0`, `%rd1` |
| `%f<N>` | 32-bit float registers | `%f0`, `%f1` |
| `%fd<N>` | 64-bit float registers | `%fd0`, `%fd1` |
| `%p<N>` | Predicate registers | `%p0`, `%p1` |
| `%h<N>` | 16-bit registers | `%h0`, `%h1` |

Declare with: `.reg .f32 %f<32>;` (creates `%f0` through `%f31`)

---

## 8. Common Patterns

### Global Index Calculation
```ptx
mov.u32         %r1, %tid.x;          // threadIdx.x
mov.u32         %r2, %ctaid.x;        // blockIdx.x
mov.u32         %r3, %ntid.x;         // blockDim.x
mad.lo.u32      %r4, %r2, %r3, %r1;   // idx = blockIdx.x * blockDim.x + threadIdx.x
```

### Bounds Check
```ptx
setp.ge.u32     %p1, %r4, %r5;        // p1 = (idx >= n)
@%p1 bra        EXIT;                  // Skip if out of bounds
```

### Shared Memory Reduction (sum)
```ptx
bar.sync        0;                     // Synchronize block
shr.u32         %r6, %r3, 1;          // s = blockDim.x / 2
LOOP:
setp.ge.u32     %p2, %r1, %r6;        // if (tid >= s) skip
@%p2 bra        SKIP;
add.u32         %r7, %r1, %r6;        // tid + s
shl.b32         %r8, %r7, 2;          // byte offset
add.u64         %rd2, %rd1, %r8;      // address
ld.shared.f32   %f2, [%rd2];          // load sdata[tid + s]
shl.b32         %r9, %r1, 2;
add.u64         %rd3, %rd1, %r9;
ld.shared.f32   %f3, [%rd3];          // load sdata[tid]
add.f32         %f3, %f3, %f2;        // sdata[tid] += sdata[tid + s]
st.shared.f32   [%rd3], %f3;
SKIP:
bar.sync        0;
shr.u32         %r6, %r6, 1;          // s >>= 1
setp.ne.u32     %p3, %r6, 0;
@%p3 bra        LOOP;
```

### Atomic Add to Global
```ptx
atom.global.add.f32 %f4, [%rd4], %f3; // atomicAdd(result, value)
```

---

## 9. Quick Reference Card

| Category | Key Instructions |
|----------|------------------|
| **Move** | `mov`, `ld`, `st`, `cvt`, `cvta` |
| **Math (int)** | `add`, `sub`, `mul`, `mad`, `div`, `rem`, `min`, `max` |
| **Math (float)** | `add`, `sub`, `mul`, `fma`, `div`, `rcp`, `sqrt`, `rsqrt` |
| **Transcendental** | `sin`, `cos`, `ex2`, `lg2` |
| **Logic** | `and`, `or`, `xor`, `not`, `shl`, `shr` |
| **Compare** | `setp`, `set`, `selp`, `slct` |
| **Control** | `bra`, `call`, `ret`, `exit`, `@%p` |
| **Sync** | `bar.sync`, `membar`, `fence` |
| **Atomic** | `atom.add`, `atom.cas`, `atom.exch`, `red` |
| **Warp** | `shfl.sync`, `vote.sync`, `redux.sync` |
| **Tensor** | `wmma.*`, `mma`, `ldmatrix` |

---

## 10. Version History

| PTX Version | SM Target | Key Features |
|-------------|-----------|--------------|
| 6.0 | sm_70 | Tensor Cores, independent thread scheduling |
| 6.3 | sm_72 | INT8 tensor ops |
| 7.0 | sm_80 | Ampere, async copy, `redux.sync` |
| 7.1 | sm_86 | GA10x support |
| 7.5 | sm_87 | Orin support |
| 7.8 | sm_89 | Ada Lovelace |
| 8.0 | sm_90 | Hopper, TMA, distributed shared memory |

---

*Reference: [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)*
