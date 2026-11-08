"""Compile actual CUDA bodies as host C++ for address/descriptor regressions.

This checks integer addressing and branch semantics, not GPU concurrency,
NVRTC compilation, streams, device throughput, or complete CUDA games.
Sparse virtual mappings exercise >2^31 element addresses without allocating
multi-gigabyte physical RAM. A real-CUDA category test is separately gated.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.features import FEATURE_DIM, KINDS, PHASES
from stella.gpu_game_tables import LAYOUT, MAX_ACTIONS, STATE_SIZE, header, pack_state, action_array
from stella.hard_examples import SPECIFIC_KINDS, classify_decision
from stella.environment import Episode, heuristic_action
from stella.preset_catalogue import catalogue_presets

CUDA_SOURCE = Path(__file__).resolve().parents[1] / "stella/gpu_games.cu"

HOST_PREFIX = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sys/mman.h>
#include <stdexcept>
#define __device__
#define __global__
#define __constant__
#define __shared__ static
struct Dim {unsigned int x=0;};
Dim blockIdx, blockDim, threadIdx;
inline void __syncthreads() {}
inline int __clz(unsigned int x) {return __builtin_clz(x);}
inline int atomicMax(int* p,int v) {int old=*p;if(v>old)*p=v;return old;}
inline int atomicAdd(int* p,int v) {int old=*p;*p+=v;return old;}
"""
HOST_WRAPPERS = r"""
extern "C" void host_hard_category(int* states,const int* actions,const int* counts,
                                  const int* choices,int* output,int n) {
    blockDim.x=128;
    for(int i=0;i<n;i++) {
        blockIdx.x=i/128;threadIdx.x=i%128;
        game_hard_category(states,actions,counts,choices,output,n);
    }
}
extern "C" int host_action_capacity() {
    int actions[MAX_ACTIONS*4+4],count=0;
    for(int& value:actions)value=-123;
    for(int i=0;i<MAX_ACTIONS+3;i++)push_action(actions,count,AC_LEAVE);
    if(count!=MAX_ACTIONS+1)return 1;
    for(int i=MAX_ACTIONS*4;i<MAX_ACTIONS*4+4;i++)if(actions[i]!=-123)return 2;
    return 0;
}
template<class T> struct Mapping {
    T* p;size_t bytes;
    Mapping(size_t count):bytes(count*sizeof(T)) {
        void* raw=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        if(raw==MAP_FAILED)throw std::runtime_error("sparse virtual mapping unavailable");
        p=static_cast<T*>(raw);
    }
    ~Mapping(){munmap(p,bytes);}
};
extern "C" int host_wide_addresses() {
    try {
        // Real feature encoder at action-row address > INT32_MAX / 958.
        const int row=480000,n=row+1,start=row*MAX_ACTIONS;
        Mapping<int> states(size_t(n)*STATE_SIZE), actions(size_t(n)*MAX_ACTIONS*4),
                     counts(n),offsets(n);
        Mapping<double> beliefs(size_t(n)*72);
        Mapping<float> features((size_t(start)+MAX_ACTIONS)*FEATURE_DIM);
        int* s=states.p+size_t(row)*STATE_SIZE;
        s[S_phase]=PH_SHOP;s[S_node]=46;s[S_floor]=20;s[S_cap]=90;
        counts.p[row]=MAX_ACTIONS;offsets.p[row-1]=start;offsets.p[row]=start+MAX_ACTIONS;
        for(int j=0;j<MAX_ACTIONS;j++)actions.p[(size_t(row)*MAX_ACTIONS+j)*4]=AC_LEAVE;
        blockIdx.x=row;threadIdx.x=0;blockDim.x=1;
        game_features(states.p,beliefs.p,actions.p,counts.p,offsets.p,features.p,n);
        if(features.p[size_t(start)*FEATURE_DIM+F_action_kind+AC_LEAVE]!=1.0f)return 1;
        if(features.p[(size_t(start)+1)*FEATURE_DIM+F_inventory]!=0.0f)return 2;
        // Real MT stride arithmetic crossing INT32_MAX elements.
        const int stride=3500000;
        Mapping<u32> randoms(size_t(stride)*625);
        RNG rng{randoms.p,stride};rng.at(624)=601;
        if(randoms.p[size_t(stride)*624]!=601)return 3;
        // Real finish kernel crossing INT32_MAX packed-state elements.
        const int last=2147483647/STATE_SIZE+1,nn=last+1;
        Mapping<int> finish_states(size_t(nn)*STATE_SIZE),scores(nn),steps(nn);
        int* final_state=finish_states.p+size_t(last)*STATE_SIZE;
        final_state[S_H+6]=2;final_state[S_steps]=73;
        blockDim.x=128;blockIdx.x=last/128;threadIdx.x=last%128;
        game_finish(finish_states.p,scores.p,steps.p,nn);
        if(scores.p[last]!=2 || steps.p[last]!=73)return 4;
        return 0;
    }catch(const std::exception&){return 77;}
}
"""


@pytest.fixture(scope="module")
def native_kernel(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if sys.platform != "linux" or not compiler:
        pytest.skip("Linux C++ compiler is needed for the independent host kernel checks")
    directory = tmp_path_factory.mktemp("cuda-host")
    source = directory / "kernel.cpp"
    source.write_text(HOST_PREFIX + header() + CUDA_SOURCE.read_text() + HOST_WRAPPERS)
    library = directory / "kernel.so"
    process = subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
                              "-ffp-contract=off", str(source), "-o", str(library)],
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr
    module = C.CDLL(str(library))
    pointer = C.POINTER(C.c_int32)
    module.host_hard_category.argtypes = [pointer]*5 + [C.c_int]
    module.host_hard_category.restype = None
    module.host_wide_addresses.argtypes = []
    module.host_wide_addresses.restype = C.c_int
    module.host_action_capacity.argtypes = []
    module.host_action_capacity.restype = C.c_int
    return module


def descriptor_cases():
    states, actions, counts, choices, expected = [], [], [], [], []
    names = (None, *SPECIFIC_KINDS, "shop_general", "page_general")
    presets = catalogue_presets()
    for index in (0, 167, 1300, 3961, 4959):
        for seed in (0, 11, 29):
            episode = Episode(presets[index], seed)
            while not episode.done:
                legal = episode.actions
                if len(legal) > 1:
                    packed, _ = pack_state(episode.state)
                    padded = np.zeros((MAX_ACTIONS, 4), np.int32)
                    padded[:len(legal)] = [action_array(a) for a in legal]
                    descriptors = [(a.kind, a.index, a.price) for a in legal]
                    for choice in range(len(legal)):
                        states.append(packed.copy());actions.append(padded.copy())
                        counts.append(len(legal));choices.append(choice)
                        expected.append(names.index(classify_decision(episode.state.phase, episode.state.offer, descriptors, choice)))
                episode.step(heuristic_action(episode.state, legal))
    # Exercise the priority case where either BUY160 or ENHANCE180 can be skipped.
    state, _ = pack_state(Episode(presets[0], 8).state)
    state[LAYOUT["phase"]] = PHASES.index("SHOP")
    combined = [("ENHANCE", 0, 180), ("BUY", 2, 160), ("RESTOCK", 0, 100), ("LEAVE", 0, 0)]
    padded = np.zeros((MAX_ACTIONS, 4), np.int32)
    for j, (kind, index, price) in enumerate(combined):padded[j] = KINDS.index(kind), index, price, 0
    for choice in range(4):
        states.append(state.copy());actions.append(padded.copy());counts.append(4);choices.append(choice)
        expected.append(names.index(classify_decision("SHOP", (), combined, choice)))
    return tuple(np.ascontiguousarray(a, dtype=np.int32) for a in (states, actions, counts, choices, expected))


def test_actual_kernel_uses_wide_addresses(native_kernel):
    code = native_kernel.host_wide_addresses()
    if code == 77:
        pytest.skip("Host virtual address limit prevents sparse >2^31 address probe")
    assert code == 0, f"Wide address probe failed in stage {code}"


def test_action_capacity_overflow_is_reported_without_writing_the_next_row(native_kernel):
    assert native_kernel.host_action_capacity() == 0


def test_actual_fused_category_matches_public_python_rules(native_kernel):
    states, actions, counts, choices, expected = descriptor_cases()
    output = np.empty(len(states), np.int32)
    ptr = lambda a: a.ctypes.data_as(C.POINTER(C.c_int32))
    native_kernel.host_hard_category(*map(ptr, (states, actions, counts, choices, output)), len(states))
    np.testing.assert_array_equal(output, expected)
    assert len(output) > 1000
    assert not states[:, LAYOUT["error"]].any()


def test_fused_category_rejects_illegal_choice_without_reading_padding(native_kernel):
    states, actions, counts, choices, _ = descriptor_cases()
    states, actions, counts, choices = (a[:2].copy() for a in (states, actions, counts, choices))
    choices[:] = [-1, MAX_ACTIONS]
    output = np.empty(2, np.int32)
    ptr = lambda a: a.ctypes.data_as(C.POINTER(C.c_int32))
    native_kernel.host_hard_category(*map(ptr, (states, actions, counts, choices, output)), 2)
    assert np.all(states[:, LAYOUT["error"]] == 7)
    assert np.all(output == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA runtime unavailable")
def test_real_cuda_fused_category_matches_public_python_rules():
    from stella.cuda_runtime import CudaModule
    states, actions, counts, choices, expected = descriptor_cases()
    module = CudaModule(header() + CUDA_SOURCE.read_text())
    try:
        tensors = tuple(torch.from_numpy(a).to("cuda") for a in (states, actions, counts, choices))
        output = torch.empty(len(states), dtype=torch.int32, device="cuda")
        module.launch("game_hard_category", len(states), (*tensors, output, len(states)))
        np.testing.assert_array_equal(output.cpu().numpy(), expected)
        assert not tensors[0][:, LAYOUT["error"]].any().item()
    finally:
        torch.cuda.synchronize()
        module.close()
