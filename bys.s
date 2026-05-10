// ============================================================
//  Bayesian Optimisation  -  ARM64 Assembly for Apple M1/M2
//
//  Build:  clang -arch arm64 -o bayesian_opt bayesian_opt.s -lm
//  Run:    ./bayesian_opt
//
//  Algorithm: Gaussian Process + Lower Confidence Bound (LCB)
//    surrogate : GP with RBF kernel  l=0.5, sigma^2=0.01
//    acquisition: LCB(x) = mu(x) - kappa*sigma(x),  kappa=2.0   (minimise)
//    grid search: 200 points per iteration over [-2, 2]
//    seeds: 3 evaluations at -2, 0, +2
//    iterations: 15
//
//  Target (black-box):
//    f(x) = sin(3x) + x^2 - 0.7*cos(5x),  x in [-2, 2]
// ============================================================
//  Constants
.equ MAX_OBS,  20
.equ GRID_N,   200
.equ MAX_ITER, 15

//  BSS – zero-initialised globals
.section __DATA,__bss
.align  3
obs_x:   .space 160
obs_y:   .space 160
K_mat:   .space 3200
L_mat:   .space 3200
alph:    .space 160
n_obs:   .space 8

//  Read-only data
.section __TEXT,__const
.align  3
C_LEN2:   .double 0.5
C_NOISE:  .double 0.01
C_KAPPA:  .double 2.0
C_XLO:    .double -2.0
C_XHI:    .double 2.0
C_STEP:   .double 0.02010050251256281
C_ONE:    .double 1.0
C_ZERO:   .double 0.0
C_THREE:  .double 3.0
C_FIVE:   .double 5.0
C_07:     .double 0.7
C_POSINF: .double 1.0e300
FMT_HDR:  .asciz "Bayesian Optimisation \nf(x) = sin(3x) + x^2 - 0.7*cos(5x),  x in [-2, 2]\n\n"
FMT_ITER: .asciz "[iter %2lld]  x = %8.4f   f(x) = %8.4f\n"
FMT_BEST: .asciz "\n=== Best after %lld iterations ===\n  x    = %10.6f\n  f(x) = %10.6f\n"

//  CODE
.section __TEXT,__text,regular,pure_instructions
.align  4

//  rbf_kernel  —  d0=xi, d1=xj  →  d0 = exp(-(xi-xj)²/(2ℓ²))
rbf_kernel:
    stp  x29, x30, [sp, #-16]!
    mov  x29, sp
    fsub  d0, d0, d1
    fmul  d0, d0, d0
    adrp  x0, C_LEN2@PAGE
    add   x0, x0, C_LEN2@PAGEOFF
    ldr   d1, [x0]
    fdiv  d0, d0, d1
    fneg  d0, d0
    bl    _exp
    ldp  x29, x30, [sp], #16
    ret

//  target_f  —  d0=x  →  d0 = sin(3x)+x²-0.7cos(5x)
target_f:
    stp  x29, x30, [sp, #-32]!
    mov  x29, sp
    str   d0, [sp, #16]
    adrp  x0, C_THREE@PAGE
    add   x0, x0, C_THREE@PAGEOFF
    ldr   d1, [x0]
    ldr   d0, [sp, #16]
    fmul  d0, d0, d1
    bl    _sin
    str   d0, [sp, #24]
    ldr   d0, [sp, #16]
    fmul  d0, d0, d0
    ldr   d1, [sp, #24]
    fadd  d0, d0, d1
    str   d0, [sp, #24]
    adrp  x0, C_FIVE@PAGE
    add   x0, x0, C_FIVE@PAGEOFF
    ldr   d1, [x0]
    ldr   d0, [sp, #16]
    fmul  d0, d0, d1
    bl    _cos
    adrp  x0, C_07@PAGE
    add   x0, x0, C_07@PAGEOFF
    ldr   d1, [x0]
    fmul  d0, d0, d1
    ldr   d1, [sp, #24]
    fsub  d0, d1, d0
    ldp  x29, x30, [sp], #32
    ret

//  build_K  —  fills K_mat[n×n] with RBF + σ²·I
build_K:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8, d9, [sp, #48]
    adrp  x0, n_obs@PAGE
    add   x0, x0, n_obs@PAGEOFF
    ldr   x19, [x0]
    adrp  x20, obs_x@PAGE
    add   x20, x20, obs_x@PAGEOFF
    adrp  x21, K_mat@PAGE
    add   x21, x21, K_mat@PAGEOFF
    adrp  x0, C_NOISE@PAGE
    add   x0, x0, C_NOISE@PAGEOFF
    ldr   d8, [x0]
    mov   x8, #0
.bk_i:
    cmp   x8, x19
    b.ge  .bk_done
    ldr   d9, [x20, x8, lsl #3]
    mov   x9, #0
.bk_j:
    cmp   x9, x19
    b.ge  .bk_inext
    ldr   d1, [x20, x9, lsl #3]
    fmov  d0, d9
    bl    rbf_kernel
    cmp   x8, x9
    b.ne  .bk_wr
    fadd  d0, d0, d8
.bk_wr:
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x9
    str   d0, [x21, x10, lsl #3]
    add   x9, x9, #1
    b     .bk_j
.bk_inext:
    add   x8, x8, #1
    b     .bk_i
.bk_done:
    ldp   d8, d9, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

//  cholesky  —  Cholesky decomposition of K_mat into L_mat
cholesky:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8, d9, [sp, #48]
    adrp  x0, n_obs@PAGE
    add   x0, x0, n_obs@PAGEOFF
    ldr   x19, [x0]
    adrp  x20, K_mat@PAGE
    add   x20, x20, K_mat@PAGEOFF
    adrp  x21, L_mat@PAGE
    add   x21, x21, L_mat@PAGEOFF
    mov   x8, x19
    mov   x9, #MAX_OBS
    mul   x8, x8, x9
    mov   x9, #0
.ch_cp:
    cmp   x9, x8
    b.ge  .ch_fac
    ldr   d0, [x20, x9, lsl #3]
    str   d0, [x21, x9, lsl #3]
    add   x9, x9, #1
    b     .ch_cp
.ch_fac:
    mov   x8, #0
.ch_j:
    cmp   x8, x19
    b.ge  .ch_ret
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x8
    ldr   d8, [x21, x10, lsl #3]
    mov   x9, #0
.ch_ds:
    cmp   x9, x8
    b.ge  .ch_sq
    mov   x11, #MAX_OBS
    mul   x11, x8, x11
    add   x11, x11, x9
    ldr   d0, [x21, x11, lsl #3]
    fmul  d0, d0, d0
    fsub  d8, d8, d0
    add   x9, x9, #1
    b     .ch_ds
.ch_sq:
    fsqrt  d8, d8
    str    d8, [x21, x10, lsl #3]
    add   x12, x8, #1
.ch_i:
    cmp   x12, x19
    b.ge  .ch_jnx
    mov   x10, #MAX_OBS
    mul   x10, x12, x10
    add   x10, x10, x8
    ldr   d9, [x21, x10, lsl #3]
    mov   x9, #0
.ch_os:
    cmp   x9, x8
    b.ge  .ch_od
    mov   x11, #MAX_OBS
    mul   x11, x12, x11
    add   x11, x11, x9
    ldr   d0, [x21, x11, lsl #3]
    mov   x11, #MAX_OBS
    mul   x11, x8, x11
    add   x11, x11, x9
    ldr   d1, [x21, x11, lsl #3]
    fmul  d0, d0, d1
    fsub  d9, d9, d0
    add   x9, x9, #1
    b     .ch_os
.ch_od:
    fdiv  d9, d9, d8
    mov   x10, #MAX_OBS
    mul   x10, x12, x10
    add   x10, x10, x8
    str   d9, [x21, x10, lsl #3]
    add   x12, x12, #1
    b     .ch_i
.ch_jnx:
    add   x8, x8, #1
    b     .ch_j
.ch_ret:
    ldp   d8, d9, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

//  solve_alpha  —  solves (L Lᵀ)α = obs_y
solve_alpha:
    stp  x29, x30, [sp, #-64]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8, d9, [sp, #48]
    adrp  x0, n_obs@PAGE
    add   x0, x0, n_obs@PAGEOFF
    ldr   x19, [x0]
    adrp  x20, L_mat@PAGE
    add   x20, x20, L_mat@PAGEOFF
    adrp  x21, obs_y@PAGE
    add   x21, x21, obs_y@PAGEOFF
    adrp  x22, alph@PAGE
    add   x22, x22, alph@PAGEOFF
    mov   x8, #0
.sa_fi:
    cmp   x8, x19
    b.ge  .sa_bi
    ldr   d0, [x21, x8, lsl #3]
    mov   x9, #0
.sa_fj:
    cmp   x9, x8
    b.ge  .sa_fd
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x9
    ldr   d1, [x20, x10, lsl #3]
    ldr   d2, [x22, x9, lsl #3]
    fmul  d1, d1, d2
    fsub  d0, d0, d1
    add   x9, x9, #1
    b     .sa_fj
.sa_fd:
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x8
    ldr   d1, [x20, x10, lsl #3]
    fdiv  d0, d0, d1
    str   d0, [x22, x8, lsl #3]
    add   x8, x8, #1
    b     .sa_fi
.sa_bi:
    sub   x8, x19, #1
.sa_bloop:
    cmp   x8, #0
    b.lt  .sa_ret
    ldr   d0, [x22, x8, lsl #3]
    add   x9, x8, #1
.sa_bj:
    cmp   x9, x19
    b.ge  .sa_bd
    mov   x10, #MAX_OBS
    mul   x10, x9, x10
    add   x10, x10, x8
    ldr   d1, [x20, x10, lsl #3]
    ldr   d2, [x22, x9, lsl #3]
    fmul  d1, d1, d2
    fsub  d0, d0, d1
    add   x9, x9, #1
    b     .sa_bj
.sa_bd:
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x8
    ldr   d1, [x20, x10, lsl #3]
    fdiv  d0, d0, d1
    str   d0, [x22, x8, lsl #3]
    sub   x8, x8, #1
    b     .sa_bloop
.sa_ret:
    ldp   d8, d9, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

//  gp_predict  —  d0=x★  →  d0=μ, d1=σ
gp_predict:
    stp  x29, x30, [sp, #-256]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8, d9, [sp, #48]
    str  d10, [sp, #64]
    str  d0, [sp, #72]
    adrp  x0, n_obs@PAGE
    add   x0, x0, n_obs@PAGEOFF
    ldr   x19, [x0]
    adrp  x20, obs_x@PAGE
    add   x20, x20, obs_x@PAGEOFF
    adrp  x21, alph@PAGE
    add   x21, x21, alph@PAGEOFF
    adrp  x22, L_mat@PAGE
    add   x22, x22, L_mat@PAGEOFF
    mov   x8, #0
.gp_kv:
    cmp   x8, x19
    b.ge  .gp_mu
    ldr   d0, [sp, #72]
    ldr   d1, [x20, x8, lsl #3]
    bl    rbf_kernel
    add   x0, sp, #80
    str   d0, [x0, x8, lsl #3]
    add   x8, x8, #1
    b     .gp_kv
.gp_mu:
    fmov  d8, #0.0
    mov   x8, #0
.gp_mu2:
    cmp   x8, x19
    b.ge  .gp_fwd
    add   x0, sp, #80
    ldr   d0, [x0, x8, lsl #3]
    ldr   d1, [x21, x8, lsl #3]
    fmul  d0, d0, d1
    fadd  d8, d8, d0
    add   x8, x8, #1
    b     .gp_mu2
.gp_fwd:
    mov   x8, #0
.gp_fi:
    cmp   x8, x19
    b.ge  .gp_vtv
    add   x0, sp, #80
    ldr   d0, [x0, x8, lsl #3]
    mov   x9, #0
.gp_fj:
    cmp   x9, x8
    b.ge  .gp_fd
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x9
    ldr   d1, [x22, x10, lsl #3]
    add   x10, sp, #80
    ldr   d2, [x10, x9, lsl #3]
    fmul  d1, d1, d2
    fsub  d0, d0, d1
    add   x9, x9, #1
    b     .gp_fj
.gp_fd:
    mov   x10, #MAX_OBS
    mul   x10, x8, x10
    add   x10, x10, x8
    ldr   d1, [x22, x10, lsl #3]
    fdiv  d0, d0, d1
    add   x10, sp, #80
    str   d0, [x10, x8, lsl #3]
    add   x8, x8, #1
    b     .gp_fi
.gp_vtv:
    adrp  x0, C_ONE@PAGE
    add   x0, x0, C_ONE@PAGEOFF
    ldr   d9, [x0]
    mov   x8, #0
.gp_vv:
    cmp   x8, x19
    b.ge  .gp_sigma
    add   x0, sp, #80
    ldr   d0, [x0, x8, lsl #3]
    fmul  d0, d0, d0
    fsub  d9, d9, d0
    add   x8, x8, #1
    b     .gp_vv
.gp_sigma:
    fmov  d0, #0.0
    fmaxnm d9, d9, d0
    fsqrt  d1, d9
    fmov   d0, d8
    ldr   d10, [sp, #64]
    ldp   d8, d9, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #256
    ret

//  add_obs  —  d0=x, d1=y  →  append and increment n_obs
add_obs:
    stp  x29, x30, [sp, #-16]!
    mov  x29, sp
    adrp  x0, n_obs@PAGE
    add   x0, x0, n_obs@PAGEOFF
    ldr   x1, [x0]
    adrp  x2, obs_x@PAGE
    add   x2, x2, obs_x@PAGEOFF
    str   d0, [x2, x1, lsl #3]
    adrp  x2, obs_y@PAGE
    add   x2, x2, obs_y@PAGEOFF
    str   d1, [x2, x1, lsl #3]
    add   x1, x1, #1
    str   x1, [x0]
    ldp  x29, x30, [sp], #16
    ret

//  update_gp  —  rebuild K, Cholesky, alpha
update_gp:
    stp  x29, x30, [sp, #-16]!
    mov  x29, sp
    bl   build_K
    bl   cholesky
    bl   solve_alpha
    ldp  x29, x30, [sp], #16
    ret

//  grid_argmax  —  returns x that minimises LCB(x) = μ(x) − κ·σ(x)
grid_argmax:
    stp  x29, x30, [sp, #-96]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8, d9, [sp, #48]
    stp  d10, d11, [sp, #64]
    str  d12, [sp, #80]
    adrp  x0, C_KAPPA@PAGE
    add   x0, x0, C_KAPPA@PAGEOFF
    ldr   d8, [x0]
    adrp  x0, C_POSINF@PAGE
    add   x0, x0, C_POSINF@PAGEOFF
    ldr   d9, [x0]
    adrp  x0, C_XLO@PAGE
    add   x0, x0, C_XLO@PAGEOFF
    ldr   d10, [x0]
    adrp  x0, C_STEP@PAGE
    add   x0, x0, C_STEP@PAGEOFF
    ldr   d11, [x0]
    fmov  d12, d10
    mov   x19, #0
.ga_loop:
    cmp   x19, #GRID_N
    b.ge  .ga_done
    scvtf d0, x19
    fmul  d0, d0, d11
    fadd  d0, d0, d10
    str   d0, [sp, #88]
    bl    gp_predict
    ldr   d2, [sp, #88]
    fmul  d1, d1, d8
    fsub  d0, d0, d1
    fcmp  d0, d9
    b.ge  .ga_next
    fmov  d9, d0
    fmov  d12, d2
.ga_next:
    add   x19, x19, #1
    b     .ga_loop
.ga_done:
    fmov  d0, d12
    ldr   d12, [sp, #80]
    ldp   d10, d11, [sp, #64]
    ldp   d8, d9, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #96
    ret

//  _main
.globl _main
_main:
    stp  x29, x30, [sp, #-128]!
    mov  x29, sp
    stp  x19, x20, [sp, #16]
    stp  x21, x22, [sp, #32]
    stp  d8, d9, [sp, #48]
    stp  d10, d11, [sp, #64]
    stp  d12, d13, [sp, #80]
    adrp  x0, FMT_HDR@PAGE
    add   x0, x0, FMT_HDR@PAGEOFF
    bl    _printf
    adrp  x0, C_XLO@PAGE
    add   x0, x0, C_XLO@PAGEOFF
    ldr   d0, [x0]
    str   d0, [sp, #96]
    bl    target_f
    fmov  d1, d0
    ldr   d0, [sp, #96]
    bl    add_obs
    fmov  d0, #0.0
    str   d0, [sp, #96]
    bl    target_f
    fmov  d1, d0
    ldr   d0, [sp, #96]
    bl    add_obs
    adrp  x0, C_XHI@PAGE
    add   x0, x0, C_XHI@PAGEOFF
    ldr   d0, [x0]
    str   d0, [sp, #96]
    bl    target_f
    fmov  d1, d0
    ldr   d0, [sp, #96]
    bl    add_obs
    bl    update_gp
    adrp  x0, obs_x@PAGE
    add   x0, x0, obs_x@PAGEOFF
    ldr   d8, [x0]
    adrp  x0, obs_y@PAGE
    add   x0, x0, obs_y@PAGEOFF
    ldr   d9, [x0]
    mov   x19, #1
.find_best:
    cmp   x19, #3
    b.ge  .main_loop
    adrp  x0, obs_y@PAGE
    add   x0, x0, obs_y@PAGEOFF
    ldr   d0, [x0, x19, lsl #3]
    fcmp  d0, d9
    b.ge  .next_seed
    fmov  d9, d0
    adrp  x0, obs_x@PAGE
    add   x0, x0, obs_x@PAGEOFF
    ldr   d8, [x0, x19, lsl #3]
.next_seed:
    add   x19, x19, #1
    b     .find_best
.main_loop:
    mov   x20, #1
.bo_loop:
    cmp   x20, #MAX_ITER
    b.gt  .print_best
    bl    grid_argmax
    str   d0, [sp, #96]
    bl    target_f
    str   d0, [sp, #104]
    sub   sp, sp, #32
    str   x20, [sp, #0]
    ldr   d0, [sp, #128]
    str   d0, [sp, #8]
    ldr   d1, [sp, #136]
    str   d1, [sp, #16]
    adrp  x0, FMT_ITER@PAGE
    add   x0, x0, FMT_ITER@PAGEOFF
    bl    _printf
    add   sp, sp, #32
    ldr   d0, [sp, #104]
    fcmp  d0, d9
    b.ge  .no_best_update
    fmov  d9, d0
    ldr   d8, [sp, #96]
.no_best_update:
    ldr   d0, [sp, #96]
    ldr   d1, [sp, #104]
    bl    add_obs
    bl    update_gp
    add   x20, x20, #1
    b     .bo_loop
.print_best:
    sub   sp, sp, #32
    mov   x1, #MAX_ITER
    str   x1, [sp, #0]
    str   d8, [sp, #8]
    str   d9, [sp, #16]
    adrp  x0, FMT_BEST@PAGE
    add   x0, x0, FMT_BEST@PAGEOFF
    bl    _printf
    add   sp, sp, #32
    mov   x0, #0
    ldp   d12, d13, [sp, #80]
    ldp   d10, d11, [sp, #64]
    ldp   d8, d9, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret
