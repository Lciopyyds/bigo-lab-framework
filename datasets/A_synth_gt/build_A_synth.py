import json
import random
import hashlib
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
TASKS_OUT = OUT_DIR / "tasks.jsonl"
LABELS_OUT = OUT_DIR / "labels.jsonl"

random.seed(2026)

# 7类（先不做 n!）
CLASSES = [
    "O(1)",
    "O(log n)",
    "O(n)",
    "O(n log n)",
    "O(n^2)",
    "O(n^3)",
    "O(2^n)",
]

# 先跑通；之后可调到 1000/2000+
NUM_EACH = 300


def make_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:06d}"


# -------------------------
# dataclass_code（保持与你示例一致）
# -------------------------
def dataclass_code_for_n_list_int() -> str:
    # 输入：n + 一行 n 个 int（末尾再有一个空行）
    return r"""import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *
from dataclasses import dataclass

import builtins
import re

def strip(s, characters = None):
    if characters is None:
        characters = [' ', '\t', '\n', '\r', '\v', '\f']
    else:
        characters = list(characters)

    characters = [x for x in characters if len(x) > 0] 

    i = 0
    while i < len(s):
        found_sep_candidate = False

        for sep_candidate in characters:
            if s[i:i + len(sep_candidate)] == sep_candidate:
                found_sep_candidate = True
                i += len(sep_candidate)
                break

        if not found_sep_candidate:
            break

    j = len(s) - 1
    while j >= 0:
        found_sep_candidate = False

        for sep_candidate in characters:
            if s[j + 1 - len(sep_candidate):j+1] == sep_candidate:
                found_sep_candidate = True
                j -= len(sep_candidate)
                break

        if not found_sep_candidate:
            break

    return s[i:j+1]

def split(s, sep=None, maxsplit=-1):
    if sep == '':
        raise builtins.ValueError('empty separator')

    if type(sep) == list and '' in sep:
        raise builtins.ValueError('empty separator')

    if sep is None:
        sep = [' ', '\t', '\n', '\r', '\v', '\f']
        result = []
        word = ''
        count_split = 0
        
        if maxsplit == -1:
            maxsplit = len(s) * 1000

        i = 0
        while i < len(s):
            found_sep_candidate = False

            for sep_candidate in sep:
                if s[i:i + len(sep_candidate)] == sep_candidate:
                    found_sep_candidate = True

                    if word:
                        result.append(word)
                        count_split += 1
                        word = ''

                    i += len(sep_candidate)
                    break

            if not found_sep_candidate and count_split < maxsplit:
                word += s[i]
                i += 1

            elif not found_sep_candidate and count_split >= maxsplit:
                word += s[i:]
                i = len(s)

        if word:
            result.append(word)
        return result
    
    if type(sep) == str:
        sep = [sep]

    if maxsplit == -1:
        maxsplit = 0
    elif maxsplit == 0:
        maxsplit = -1

    return re.split(re.compile("|".join([re.escape(x) for x in sep])), s, maxsplit=maxsplit)

class str_escaped(str):
    def split(self, sep=None, maxsplit=-1):
        return split(self, sep=sep, maxsplit=maxsplit)
    
    def strip(self, chars=None):
        return strip(self, characters = chars)

from dataclasses import dataclass
from typing import List

@dataclass
class Input:
    n: int
    a_list: List[int]

    @classmethod
    def from_str(cls, input_: str):
        n, a_list, _ = input_.split('\n')
        n = int(n)
        a_list = list(map(int, a_list.split()))
        assert n == len(a_list)
        return cls(n, a_list)

    def __repr__(self):
        return str(self.n) + '\n' + ' '.join(map(str, self.a_list)) + '\n'
""".strip()


def inputs_example_for_n_list(n: int) -> str:
    a = [random.randint(-10, 10) for _ in range(n)]
    return str(n) + "\n" + " ".join(map(str, a)) + "\n"


# -------------------------
# 工具：随机变量名/小变体
# -------------------------
def _vnames():
    return (
        random.choice(["x", "u", "p", "t", "z", "w"]),
        random.choice(["arr", "a", "nums", "v", "data"]),
        random.choice(["ans", "s", "res", "out", "acc"]),
    )


# ============================================================
# O(1) templates (>=8)
# ============================================================
def sol_O1_hash():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
x = (n * 1315423911) & 0xffffffff
x ^= (x >> 16)
x = (x * 2654435761) & 0xffffffff
first = {an}[0] if {an} else 0
print((x + first) % 1000000007)
""".strip()


def sol_O1_fixed_loop():
    vn, an, sn = _vnames()
    k = random.choice([32, 64, 128])
    return f"""n = int(input())
{an} = list(map(int, input().split()))
{sn} = 0
x = n & 0xffffffff
for _ in range({k}):
    x = (x * 1103515245 + 12345) & 0x7fffffff
    {sn} ^= x
first = {an}[0] if {an} else 0
print(({sn} + first) & 0xffffffff)
""".strip()


def sol_O1_lookup_table():
    vn, an, sn = _vnames()
    table = [random.randint(0, 1000) for _ in range(16)]
    table_str = ", ".join(map(str, table))
    return f"""n = int(input())
{an} = list(map(int, input().split()))
T = [{table_str}]
idx = (n ^ (n >> 3)) & 15
val = T[idx]
first = {an}[0] if {an} else 0
print((val + first) % 1000000007)
""".strip()


def sol_O1_const_matrix():
    vn, an, sn = _vnames()
    A = [[random.randint(0, 9) for _ in range(3)] for _ in range(3)]
    B = [[random.randint(0, 9) for _ in range(3)] for _ in range(3)]
    Aflat = ", ".join(str(x) for row in A for x in row)
    Bflat = ", ".join(str(x) for row in B for x in row)
    return f"""n = int(input())
arr = list(map(int, input().split()))
A = [{Aflat}]
B = [{Bflat}]
# 3x3 multiply (constant)
C0 = A[0]*B[0] + A[1]*B[3] + A[2]*B[6]
C4 = A[3]*B[1] + A[4]*B[4] + A[5]*B[7]
C8 = A[6]*B[2] + A[7]*B[5] + A[8]*B[8]
first = arr[0] if arr else 0
print((C0 + C4 + C8 + first + n) % 1000000007)
""".strip()


def sol_O1_bitmix():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
x = n & 0xffffffff
x ^= (x << 13) & 0xffffffff
x ^= (x >> 17)
x ^= (x << 5) & 0xffffffff
first = {an}[0] if {an} else 0
print((x + first) & 0xffffffff)
""".strip()


def sol_O1_const_recursion():
    # 固定深度递归（常数）
    depth = random.choice([8, 10, 12])
    return f"""n = int(input())
a = list(map(int, input().split()))
def f(k, x):
    if k == 0:
        return x
    return f(k-1, (x * 131 + k) & 0xffffffff)
x = f({depth}, n & 0xffffffff)
first = a[0] if a else 0
print((x + first) % 1000000007)
""".strip()


def sol_O1_const_heap():
    # 固定次数 heap 操作（常数）
    k = random.choice([16, 24, 32])
    return f"""import heapq
n = int(input())
a = list(map(int, input().split()))
h = []
x = n & 0xffffffff
for i in range({k}):
    x = (x * 1664525 + 1013904223) & 0xffffffff
    heapq.heappush(h, x % 100000)
s = 0
while h:
    s = (s * 131 + heapq.heappop(h)) % 1000000007
first = a[0] if a else 0
print((s + first) % 1000000007)
""".strip()


def sol_O1_branch_tree():
    return r"""n = int(input())
a = list(map(int, input().split()))
x = n & 0xffffffff
if x & 1:
    x ^= 0x9e3779b9
else:
    x = (x + 0x7f4a7c15) & 0xffffffff
if x & 2:
    x = (x * 2654435761) & 0xffffffff
else:
    x ^= (x >> 16)
first = a[0] if a else 0
print((x + first) % 1000000007)
""".strip()


# ============================================================
# O(log n) templates (>=8)
# ============================================================
def sol_Olog_halving():
    return r"""n = int(input())
a = list(map(int, input().split()))
x = max(1, n)
s = 0
while x > 1:
    s ^= x
    x //= 2
print(s)
""".strip()


def sol_Olog_pow2_grow():
    return r"""n = int(input())
a = list(map(int, input().split()))
i = 1
s = 0
while i < max(1, n):
    s = (s * 131 + i) % 1000000007
    i *= 2
print(s)
""".strip()


def sol_Olog_bitlen():
    return r"""n = int(input())
a = list(map(int, input().split()))
x = max(1, n)
s = 0
for k in range(x.bit_length()):
    s ^= (x >> k)
print(s & 0xffffffff)
""".strip()


def sol_Olog_binary_search():
    # 手写二分：在排序数组上找第一个 >= 0（log n），排序不做（否则变 nlogn）
    # 为了不需要排序，我们构造一个单调函数 f(mid) = mid - n//2
    return r"""n = int(input())
_ = input().split()
# find smallest mid such that mid >= n//2
l, r = 0, n
target = n // 2
while l < r:
    m = (l + r) // 2
    if m >= target:
        r = m
    else:
        l = m + 1
print(l)
""".strip()


def sol_Olog_gcd_chain():
    return r"""import math
n = int(input())
a = list(map(int, input().split()))
x = max(1, n)
y = 1
# Euclid-like shrink, O(log n)
while x > 1:
    y = math.gcd(x, y + 1)
    x //= 2
print(y)
""".strip()


def sol_Olog_fast_pow():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
base = 3
exp = max(0, n)
res = 1
while exp > 0:
    if exp & 1:
        res = (res * base) % MOD
    base = (base * base) % MOD
    exp >>= 1
print(res)
""".strip()


def sol_Olog_divide_conquer():
    # 递归每次减半：T(n)=T(n/2)+O(1)
    return r"""n = int(input())
a = list(map(int, input().split()))
def f(x):
    if x <= 1:
        return x
    return (f(x//2) + 1) & 0xffffffff
print(f(max(1, n)))
""".strip()


def sol_Olog_shift_reduce():
    return r"""n = int(input())
a = list(map(int, input().split()))
x = max(1, n)
s = 0
while x:
    s ^= (x & -x)
    x >>= 1
print(s & 0xffffffff)
""".strip()


# ============================================================
# O(n) templates (>=8)
# ============================================================
def sol_On_sum():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
{sn} = 0
for {vn} in {an}:
    {sn} = ({sn} + {vn}) % 1000000007
print({sn})
""".strip()


def sol_On_max():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
{sn} = -10**18
for {vn} in {an}:
    if {vn} > {sn}:
        {sn} = {vn}
print({sn})
""".strip()


def sol_On_setuniq():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
seen = set()
for {vn} in {an}:
    seen.add({vn})
print(len(seen))
""".strip()


def sol_On_prefix_checksum():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
{sn} = 0
p = 0
for {vn} in {an}:
    p += {vn}
    {sn} = ({sn} + (p & 0xffffffff)) % 1000000007
print({sn})
""".strip()


def sol_On_counter_like():
    # 不用 collections.Counter，避免 import 噪声；手写 dict
    return r"""n = int(input())
a = list(map(int, input().split()))
mp = {}
for v in a:
    mp[v] = mp.get(v, 0) + 1
# checksum
s = 0
for k, c in mp.items():
    s = (s + (k * 131 + c)) % 1000000007
print(s)
""".strip()


def sol_On_two_pointers():
    # 单 pass 双指针（仍 O(n)）
    return r"""n = int(input())
a = list(map(int, input().split()))
i, j = 0, n-1
s = 0
while i <= j:
    s = (s + a[i]) % 1000000007
    if i != j:
        s = (s + a[j]) % 1000000007
    i += 1
    j -= 1
print(s)
""".strip()


def sol_On_bucket_small_range():
    # 值域[-10,10]（inputs_example 用的是 -10..10），桶计数 O(n)
    return r"""n = int(input())
a = list(map(int, input().split()))
cnt = [0]*21
for v in a:
    cnt[v+10] += 1
s = 0
for i,c in enumerate(cnt):
    s = (s * 131 + c) % 1000000007
print(s)
""".strip()


def sol_On_linear_transform():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
mul = 911382323
add = 972663749
for v in a:
    s = (s + (v*mul + add)) % MOD
print(s)
""".strip()


# ============================================================
# O(n log n) templates (>=8)
# ============================================================
def sol_Onlogn_sort_scan():
    vn, an, sn = _vnames()
    return f"""n = int(input())
{an} = list(map(int, input().split()))
{an}.sort()
{sn} = 0
for {vn} in {an}:
    {sn} = ({sn} * 131 + {vn}) % 1000000007
print({sn})
""".strip()


def sol_Onlogn_heap_popall():
    vn, an, sn = _vnames()
    return f"""import heapq
n = int(input())
{an} = list(map(int, input().split()))
heapq.heapify({an})
{sn} = 0
while {an}:
    {vn} = heapq.heappop({an})
    {sn} = ({sn} * 131 + {vn}) % 1000000007
print({sn})
""".strip()


def sol_Onlogn_sorted_group():
    return r"""n = int(input())
a = list(map(int, input().split()))
a.sort()
# count runs (linear after sort)
runs = 0
i = 0
while i < n:
    j = i + 1
    while j < n and a[j] == a[i]:
        j += 1
    runs += 1
    i = j
print(runs)
""".strip()


def sol_Onlogn_sort_two_pointer():
    return r"""n = int(input())
a = list(map(int, input().split()))
a.sort()
i, j = 0, n-1
s = 0
while i < j:
    s = (s + (a[j] - a[i])) % 1000000007
    i += 1
    j -= 1
print(s)
""".strip()


def sol_Onlogn_merge_sort_manual():
    # 手写 merge sort，明确 nlogn
    return r"""n = int(input())
a = list(map(int, input().split()))
def msort(x):
    if len(x) <= 1:
        return x
    mid = len(x)//2
    L = msort(x[:mid])
    R = msort(x[mid:])
    i=j=0
    out=[]
    while i<len(L) and j<len(R):
        if L[i] <= R[j]:
            out.append(L[i]); i+=1
        else:
            out.append(R[j]); j+=1
    if i<len(L): out.extend(L[i:])
    if j<len(R): out.extend(R[j:])
    return out
b = msort(a)
s = 0
for v in b:
    s = (s*131 + v) % 1000000007
print(s)
""".strip()


def sol_Onlogn_n_bisect_like():
    # n 次插入到“保持有序”的数组会变成 n^2，不做。
    # 这里做：n 次“二分定位”但不插入（保持 log n），总 nlogn
    return r"""n = int(input())
a = list(map(int, input().split()))
# prepare a monotonic array base
base = list(range(n))
def lower_bound(x):
    l, r = 0, n
    while l < r:
        m = (l+r)//2
        if base[m] >= x:
            r = m
        else:
            l = m+1
    return l
s = 0
for v in a:
    s = (s + lower_bound((v % (n+1)))) % 1000000007
print(s)
""".strip()


def sol_Onlogn_sort_then_binary_checks():
    # sort + 多次二分（仍 nlogn）
    return r"""n = int(input())
a = list(map(int, input().split()))
a.sort()
def lb(x):
    l, r = 0, n
    while l < r:
        m = (l+r)//2
        if a[m] >= x:
            r = m
        else:
            l = m+1
    return l
s = 0
for v in a:
    s = (s + lb(v)) % 1000000007
print(s)
""".strip()


def sol_Onlogn_topk_heap():
    # k ~ n/2，用 heap 维护 top-k，整体 n log n
    return r"""import heapq
n = int(input())
a = list(map(int, input().split()))
k = max(1, n//2)
h = []
for v in a:
    if len(h) < k:
        heapq.heappush(h, v)
    else:
        if v > h[0]:
            heapq.heapreplace(h, v)
s = 0
for v in h:
    s = (s*131 + v) % 1000000007
print(s)
""".strip()


# ============================================================
# O(n^2) templates (>=8)
# ============================================================
def sol_On2_fullpairs_xor():
    return r"""n = int(input())
a = list(map(int, input().split()))
s = 0
for i in range(n):
    ai = a[i]
    for j in range(n):
        s = (s + (ai ^ a[j])) & 0xffffffff
print(s)
""".strip()


def sol_On2_uppertri_sum():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for i in range(n):
    ai = a[i]
    for j in range(i, n):
        s = (s + ai + a[j]) % MOD
print(s)
""".strip()


def sol_On2_break_but_worst():
    # 有早停，但最坏仍 n^2（例如全不满足早停条件）
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for i in range(n):
    for j in range(n):
        s = (s + (a[i]*31 + a[j])) % MOD
        if a[i] == 999999999:  # 几乎不会触发
            break
print(s)
""".strip()


def sol_On2_dp_like():
    # 2D DP 形态（计算所有 i,j）
    return r"""n = int(input())
a = list(map(int, input().split()))
# dp[i][j] = (a[i] + a[j]) mod ...
MOD = 1000000007
dp = [[0]*n for _ in range(n)]
s = 0
for i in range(n):
    ai = a[i]
    row = dp[i]
    for j in range(n):
        row[j] = (ai + a[j]) % MOD
        s = (s + row[j]) % MOD
print(s)
""".strip()


def sol_On2_string_match_like():
    # 朴素匹配 O(n*m)，这里令 m=n（用数字转字符串）
    return r"""n = int(input())
a = list(map(int, input().split()))
s = ''.join('1' if v>=0 else '0' for v in a)
t = s  # same length
cnt = 0
for i in range(n - n + 1):
    ok = True
    for j in range(n):
        if s[i+j] != t[j]:
            ok = False
            break
    if ok:
        cnt += 1
print(cnt)
""".strip()


def sol_On2_pair_hash():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for i in range(n):
    for j in range(n):
        s = (s + ((a[i]*1315423911) ^ (a[j]*2654435761)) ) % MOD
print(s)
""".strip()


def sol_On2_count_inversions_naive():
    return r"""n = int(input())
a = list(map(int, input().split()))
inv = 0
for i in range(n):
    ai = a[i]
    for j in range(i+1, n):
        if ai > a[j]:
            inv += 1
print(inv)
""".strip()


def sol_On2_all_diffs():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for i in range(n):
    for j in range(n):
        d = a[i] - a[j]
        s = (s + (d*d)) % MOD
print(s)
""".strip()


# ============================================================
# O(n^3) templates (>=8)
# ============================================================
def sol_On3_floyd_like():
    return r"""n = int(input())
vals = list(map(int, input().split()))
need = n*n
if len(vals) < need:
    seed = (n * 911382323) & 0xffffffff
    while len(vals) < need:
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        vals.append(seed % 1000)
A = [vals[i*n:(i+1)*n] for i in range(n)]
for k in range(n):
    Ak = A[k]
    for i in range(n):
        Ai = A[i]
        aik = Ai[k]
        for j in range(n):
            v = aik + Ak[j]
            if v < Ai[j]:
                Ai[j] = v
s = 0
for i in range(n):
    for j in range(n):
        s = (s * 131 + A[i][j]) % 1000000007
print(s)
""".strip()


def sol_On3_naive_matmul():
    return r"""n = int(input())
vals = list(map(int, input().split()))
# make two n*n matrices
need = 2*n*n
if len(vals) < need:
    seed = (n * 1234567) & 0xffffffff
    while len(vals) < need:
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        vals.append(seed % 50)
A = [vals[i*n:(i+1)*n] for i in range(n)]
B = [vals[n*n + i*n:n*n + (i+1)*n] for i in range(n)]
MOD = 1000000007
C = [[0]*n for _ in range(n)]
for i in range(n):
    for k in range(n):
        aik = A[i][k]
        for j in range(n):
            C[i][j] = (C[i][j] + aik * B[k][j]) % MOD
s = 0
for i in range(n):
    for j in range(n):
        s = (s*131 + C[i][j]) % MOD
print(s)
""".strip()


def sol_On3_triplet_count():
    return r"""n = int(input())
a = list(map(int, input().split()))
cnt = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            if (a[i] + a[j] + a[k]) % 7 == 0:
                cnt += 1
print(cnt)
""".strip()


def sol_On3_3sum_naive():
    return r"""n = int(input())
a = list(map(int, input().split()))
target = 0
cnt = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            if a[i] + a[j] + a[k] == target:
                cnt += 1
print(cnt)
""".strip()


def sol_On3_cubic_relax():
    return r"""n = int(input())
a = list(map(int, input().split()))
# build n*n from a, repeat if needed
vals = a[:]
need = n*n
if len(vals) < need:
    seed = (n * 998244353) & 0xffffffff
    while len(vals) < need:
        seed = (seed * 1664525 + 1013904223) & 0xffffffff
        vals.append(seed % 1000)
M = [vals[i*n:(i+1)*n] for i in range(n)]
for k in range(n):
    for i in range(n):
        for j in range(n):
            M[i][j] = (M[i][j] + M[i][k] - M[k][j]) % 1000000007
s = 0
for i in range(n):
    for j in range(n):
        s = (s*131 + M[i][j]) % 1000000007
print(s)
""".strip()


def sol_On3_conv_naive():
    return r"""n = int(input())
a = list(map(int, input().split()))
# naive 3-fold convolution-ish: sum_{i,j,k} a[i]*a[j]*a[k] mod
MOD = 1000000007
s = 0
for i in range(n):
    ai = a[i]
    for j in range(n):
        aij = ai * a[j]
        for k in range(n):
            s = (s + aij * a[k]) % MOD
print(s)
""".strip()


def sol_On3_dp3d_shape():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
# 3-level DP-like loops (no huge memory)
s = 0
for i in range(n):
    for j in range(n):
        base = (a[i] + a[j]) % MOD
        for k in range(n):
            s = (s + base + a[k]) % MOD
print(s)
""".strip()


def sol_On3_triangle():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for i in range(n):
    for j in range(i, n):
        for k in range(j, n):
            s = (s + a[i] + a[j] + a[k]) % MOD
print(s)
""".strip()


# ============================================================
# O(2^n) templates (>=8)
# ============================================================
def sol_O2n_mask_enum():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for mask in range(1 << n):
    tot = 0
    m = mask
    i = 0
    while i < n:
        if m & 1:
            tot += a[i]
        m >>= 1
        i += 1
    s = (s + tot) % MOD
print(s)
""".strip()


def sol_O2n_backtrack():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
def dfs(i, acc):
    global s
    if i == n:
        s = (s + acc) % MOD
        return
    dfs(i+1, acc)
    dfs(i+1, acc + a[i])
dfs(0, 0)
print(s)
""".strip()


def sol_O2n_gray_code():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
prev = 0
cur_sum = 0
for t in range(1 << n):
    g = t ^ (t >> 1)
    diff = prev ^ g
    if diff:
        b = (diff & -diff).bit_length() - 1
        if (g >> b) & 1:
            cur_sum += a[b]
        else:
            cur_sum -= a[b]
    s = (s + cur_sum) % MOD
    prev = g
print(s)
""".strip()


def sol_O2n_popcount_weight():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for mask in range(1 << n):
    # compute weighted sum by popcount
    pc = 0
    m = mask
    while m:
        m &= m - 1
        pc += 1
    s = (s + pc) % MOD
print(s)
""".strip()


def sol_O2n_subset_dp():
    # 2^n * n：子集DP形态
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
dp = [0] * (1 << n)
for mask in range(1, 1 << n):
    lsb = mask & -mask
    b = (lsb.bit_length() - 1)
    dp[mask] = (dp[mask ^ lsb] + a[b]) % MOD
s = 0
for v in dp:
    s = (s + v) % MOD
print(s)
""".strip()


def sol_O2n_backtrack_prune_worst():
    # 带剪枝条件但几乎不触发 -> 最坏仍 2^n
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
def dfs(i, acc):
    global s
    if acc == 10**18:  # 几乎不触发
        return
    if i == n:
        s = (s + acc) % MOD
        return
    dfs(i+1, acc)
    dfs(i+1, acc + a[i])
dfs(0, 0)
print(s)
""".strip()


def sol_O2n_enum_bits():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for mask in range(1 << n):
    tot = 0
    for i in range(n):
        if (mask >> i) & 1:
            tot += (a[i] * (i+1))
    s = (s + tot) % MOD
print(s)
""".strip()


def sol_O2n_subset_minmax():
    return r"""n = int(input())
a = list(map(int, input().split()))
MOD = 1000000007
s = 0
for mask in range(1 << n):
    mn = 10**18
    mx = -10**18
    for i in range(n):
        if (mask >> i) & 1:
            v = a[i]
            if v < mn: mn = v
            if v > mx: mx = v
    if mn == 10**18:
        mn = 0; mx = 0
    s = (s + (mx - mn)) % MOD
print(s)
""".strip()


# -------------------------
# 模板池
# -------------------------
SOL_POOL = {
    "O(1)": [
        sol_O1_hash, sol_O1_fixed_loop, sol_O1_lookup_table, sol_O1_const_matrix,
        sol_O1_bitmix, sol_O1_const_recursion, sol_O1_const_heap, sol_O1_branch_tree
    ],
    "O(log n)": [
        sol_Olog_halving, sol_Olog_pow2_grow, sol_Olog_bitlen, sol_Olog_binary_search,
        sol_Olog_gcd_chain, sol_Olog_fast_pow, sol_Olog_divide_conquer, sol_Olog_shift_reduce
    ],
    "O(n)": [
        sol_On_sum, sol_On_max, sol_On_setuniq, sol_On_prefix_checksum,
        sol_On_counter_like, sol_On_two_pointers, sol_On_bucket_small_range, sol_On_linear_transform
    ],
    "O(n log n)": [
        sol_Onlogn_sort_scan, sol_Onlogn_heap_popall, sol_Onlogn_sorted_group, sol_Onlogn_sort_two_pointer,
        sol_Onlogn_merge_sort_manual, sol_Onlogn_n_bisect_like, sol_Onlogn_sort_then_binary_checks, sol_Onlogn_topk_heap
    ],
    "O(n^2)": [
        sol_On2_fullpairs_xor, sol_On2_uppertri_sum, sol_On2_break_but_worst, sol_On2_dp_like,
        sol_On2_string_match_like, sol_On2_pair_hash, sol_On2_count_inversions_naive, sol_On2_all_diffs
    ],
    "O(n^3)": [
        sol_On3_floyd_like, sol_On3_naive_matmul, sol_On3_triplet_count, sol_On3_3sum_naive,
        sol_On3_cubic_relax, sol_On3_conv_naive, sol_On3_dp3d_shape, sol_On3_triangle
    ],
    "O(2^n)": [
        sol_O2n_mask_enum, sol_O2n_backtrack, sol_O2n_gray_code, sol_O2n_popcount_weight,
        sol_O2n_subset_dp, sol_O2n_backtrack_prune_worst, sol_O2n_enum_bits, sol_O2n_subset_minmax
    ],
}

# inputs_example 的 n：保证例子可运行但不太慢
EX_N_MAP = {
    "O(1)": 10,
    "O(log n)": 128,
    "O(n)": 80,
    "O(n log n)": 120,
    "O(n^2)": 25,
    "O(n^3)": 10,
    "O(2^n)": 16,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dc = dataclass_code_for_n_list_int()

    with TASKS_OUT.open("w", encoding="utf-8") as tasks_f, LABELS_OUT.open("w", encoding="utf-8") as labels_f:
        idx = 0
        for label in CLASSES:
            pool = SOL_POOL[label]
            for k in range(NUM_EACH):
                idx += 1
                task_id = make_id("A", idx)

                # 随机抽模板
                sol = random.choice(pool)()

                # 让 problem_id/solution_id 保持你示例的风格
                problem_id = f"synth_{label.replace(' ', '').replace('^','')}"
                solution_id = f"{problem_id}_{k:04d}"

                # inputs_example
                n_ex = EX_N_MAP[label]
                inp_ex = inputs_example_for_n_list(n_ex)

                code_hash = hashlib.md5(sol.encode("utf-8")).hexdigest()[:10]

                task = {
                    "problem_name": f"SYNTH.{label}#{k}",
                    "problem_id": problem_id,
                    "solution_id": solution_id,
                    "solution_code": sol,
                    "dataclass_code": dc,
                    "inputs_example": inp_ex,
                }
                tasks_f.write(json.dumps(task, ensure_ascii=False) + "\n")

                labels_f.write(json.dumps({
                    "task_id": task_id,
                    "problem_id": problem_id,
                    "solution_id": solution_id,
                    "time_label": label,
                    "label_source": "ground_truth",
                    "template_hash": code_hash
                }, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote tasks  -> {TASKS_OUT}")
    print(f"[OK] Wrote labels -> {LABELS_OUT}")
    print(f"[OK] Total tasks: {len(CLASSES)*NUM_EACH}")


if __name__ == "__main__":
    main()