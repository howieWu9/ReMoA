# -*- coding: utf-8 -*-
"""
MOE-DUQGEN++ (Auto-K + RL)
--------------------------
在 `moe_duqgen_klock_v2.py` 基础上加入 REINFORCE/A2C 风格的优势项更新：
- 新增参数：--use_advantage / --baseline_beta / --use_critic / --critic_lr
- 在权重更新时以 advantage = score - baseline 或 score - V(S) 缩放步长，
  对负优势抑制更新，降方差、更稳。
其他逻辑（候选选优、去重、两阶段硬/软配、z-score、熵/地板、r1~r4、Auto-K 等）保持不变。
"""

import os, json, math, random, argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------- 配置 --------------------
@dataclass
class Cfg:
    collection_jsonl: str
    collection_emb_pt: str
    out_dir: str

    # 生成数量/聚类
    target_num_q: int = 1000
    auto_k: bool = False
    align_to_k: bool = False

    # 相似与榜单
    topk: int = 100
    sim_threshold: float = 0.58

    # LLM 生成
    llama_model: str = "meta-llama/Llama-2-7b-chat-hf"
    llama_cache: str = "./cache"
    llama_temperature: float = 0.8
    llama_top_p: float = 0.9
    llama_max_new_tokens: int = 64
    load_in_8bit: bool = False

    # Contriever
    contriever_name: str = "facebook/contriever-msmarco"
    contriever_cache: str = "./cache"
    max_seq_len: int = 256

    # 候选/去重/奖励
    n_cands: int = 5
    dedup_threshold: float = 0.90
    r1_tau: float = 18.0
    k_target: float = 0.7

    # 专家 + 学习
    init_w: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    eta: float = 0.30                  # 学习率
    use_zscore: bool = True
    z_temp: float = 1.3                # zscore 温度放大
    use_floor: bool = True
    w_floor: float = 0.05
    use_entropy: bool = True
    entropy: float = 0.10
    hard_match: bool = True            # 专家-奖励硬配比重
    lambda_hard: float = 0.80
    hard_warmup_ratio: float = 0.30    # 前 30% 步使用硬配

    # 专家项细节
    sqrt_neighbor_score: bool = False

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

# -------------------- 工具 --------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

@torch.no_grad()
def mean_pooling(last_hidden_state, attn_mask):
    mask = attn_mask.unsqueeze(-1).bool()
    last_hidden_state = last_hidden_state.masked_fill(~mask, 0.0)
    return last_hidden_state.sum(1) / attn_mask.sum(1, keepdim=True)

def load_docs_and_emb(cfg: Cfg):
    docs = [json.loads(l) for l in open(cfg.collection_jsonl, "r", encoding="utf-8")]
    emb_obj = torch.load(cfg.collection_emb_pt, map_location="cpu")
    if isinstance(emb_obj, dict):
        mat, order = [], []
        mp = {str(k): (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k,v in emb_obj.items()}
        for d in docs:
            if d["docid"] in mp:
                mat.append(mp[d["docid"]]); order.append(d)
        if len(mat)==0:
            raise ValueError("No embedding matched docids.")
        D = torch.stack(mat,0).float()
        docs = order
    else:
        D = emb_obj if isinstance(emb_obj, torch.Tensor) else torch.tensor(emb_obj).float()
        if D.size(0)!=len(docs):
            raise ValueError("Embeddings count mismatch.")
    D = F.normalize(D, p=2, dim=1)
    return docs, D

def tfidf_stats(texts: List[str]):
    vec = TfidfVectorizer(max_features=120000, ngram_range=(1,2))
    M = vec.fit_transform(texts)
    sums = np.asarray(M.sum(1)).ravel()
    lens = np.array([max(1,len(t.split())) for t in texts])
    x = sums / lens
    return x, float(x.min(initial=0)), float(x.max(initial=1)), vec, M

def NI_proxy(texts: List[str]):
    lens = np.array([max(1,len(t.split())) for t in texts], dtype=float)
    uniq = np.array([len(set(t.lower().split())) for t in texts], dtype=float)
    NI = 1.0 - (uniq / lens)
    NI = (NI - NI.min()) / max(1e-8, NI.max() - NI.min())
    return NI.astype(float)

@torch.no_grad()
def cos_neighbors(D: torch.Tensor, thr: float):
    N = D.size(0); bs = 2048
    cnt = torch.zeros(N, dtype=torch.long)
    for i in tqdm(range(0,N,bs), desc="cos>thr neighbors"):
        q = D[i:i+bs]
        sim = q @ D.T
        if q.size(0) == D.size(0):
            sim.fill_diagonal_(0)
        cnt[i:i+bs] = (sim > thr).sum(1).cpu()
    return cnt.numpy(), int(cnt.max().item())

@torch.no_grad()
def load_contriever(cfg: Cfg, local_only: bool):
    tok = AutoTokenizer.from_pretrained(cfg.contriever_name, cache_dir=cfg.contriever_cache, local_files_only=local_only)
    mdl = AutoModel.from_pretrained(cfg.contriever_name, cache_dir=cfg.contriever_cache, local_files_only=local_only).to(cfg.device)
    return tok, mdl

@torch.no_grad()
def encode_texts(tok, mdl, texts: List[str], cfg: Cfg):
    out=[]; bs=16
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        inp = tok(chunk, truncation=True, padding="max_length", max_length=cfg.max_seq_len, return_tensors="pt").to(cfg.device)
        last = mdl(**inp).last_hidden_state
        emb = mean_pooling(last, inp["attention_mask"])
        out.append(F.normalize(emb,p=2,dim=1).cpu())
    return torch.cat(out,0)

def nearest_cluster(labels: np.ndarray, D: torch.Tensor, c: int):
    cids = np.unique(labels)
    centers=[]
    for cid in cids:
        rows = torch.from_numpy(np.where(labels==cid)[0]).long()
        centers.append(F.normalize(D[rows].mean(0,keepdim=True),p=2,dim=1))
    centers = torch.cat(centers,0)
    sim = (centers[c:c+1] @ centers.T).squeeze(0)
    sim[c] = -1
    return int(torch.argmax(sim).item())

# ---------- 自动选择 K ----------
def auto_select_k(D: torch.Tensor, seed: int = 42) -> int:
    N = D.size(0)
    ks = sorted({max(8,int(np.sqrt(N))), max(8,N//80), max(8,N//60), max(8,N//40), max(8,N//20)})
    ks = [k for k in ks if k<N]
    sample_n = min(5000,N)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, sample_n, replace=False)
    X = D[idx].cpu().numpy()
    best_k, best_score = ks[0], -1.0
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        try:
            sc = silhouette_score(X, labels, metric="euclidean")
        except Exception:
            sc = -1.0
        if sc > best_score:
            best_score, best_k = sc, k
    return best_k

# ---------- 从语料自动构造 few-shot ----------
def auto_few_shots(texts: List[str], tfidf_vec: TfidfVectorizer, k: int = 3) -> List[Tuple[str,str]]:
    kk = min(k, max(1,len(texts)))
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(tfidf_vec.transform(texts))
    shots=[]
    for cid in range(kk):
        rows = np.where(km.labels_==cid)[0]
        ridx = int(np.random.choice(rows)) if len(rows)>0 else cid
        p = texts[ridx]
        row = tfidf_vec.transform([p])
        vocab = np.array(tfidf_vec.get_feature_names_out())
        top = np.argsort(-row.toarray()[0])[:5]
        terms = [t for t in vocab[top] if len(t.split())<=3]
        if len(terms)>=2:
            q = f"How does {terms[0]} relate to {terms[1]} in this passage?"
        elif len(terms)==1:
            q = f"What is {terms[0]} in this passage?"
        else:
            q = "How does the described effect occur and under what conditions?"
        shots.append((p[:600], q))
    while len(shots)<k:
        shots.append((
            "The study investigates the impact of a treatment on a population using a randomized design.",
            "How does the treatment affect the outcome compared to control?"
        ))
    return shots[:k]

# ---------- 可回答片段 A（TF-IDF 滑窗） ----------
def build_answer_snippets(texts: List[str], tfidf_vec: TfidfVectorizer, win_len: int = 70) -> List[str]:
    A=[]
    vocab = tfidf_vec.get_feature_names_out()
    for p in texts:
        toks = p.split()
        if len(toks) <= win_len:
            A.append(p)
            continue
        # 选 10 个高 tf-idf 的词，滑窗命中最多作为可回答片段
        row = tfidf_vec.transform([p]).toarray()[0]
        top_idx = np.argsort(-row)[:10]
        keys = set(vocab[i] for i in top_idx)
        best_s, best_l, best_r = -1, 0, win_len
        for l in range(0, len(toks)-win_len+1, max(10, win_len//2)):
            r = l + win_len
            seg = " ".join(toks[l:r]).lower()
            score = sum(1 for k in keys if k in seg)
            if score>best_s: best_s, best_l, best_r = score, l, r
        A.append(" ".join(toks[best_l:best_r]))
    return A

# -------------------- LLaMA-2 生成器 --------------------
class LlamaGen:
    def __init__(self, cfg: Cfg, auto_shots: List[Tuple[str,str]], local_only: bool = False):
        self.cfg = cfg
        self.tok = None; self.mdl = None
        self._build(local_only=local_only)
        self.auto_shots = auto_shots

    def _build(self, local_only: bool):
        self.tok = AutoTokenizer.from_pretrained(
            self.cfg.llama_model, cache_dir=self.cfg.llama_cache,
            use_fast=True, token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            local_files_only=local_only
        )
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        quant = None
        if self.cfg.device=="cuda" and self.cfg.load_in_8bit:
            try:
                quant = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                print("[WARN] bitsandbytes 不可用，改半精度。")
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.cfg.llama_model, cache_dir=self.cfg.llama_cache,
            device_map="auto" if self.cfg.device=="cuda" else None,
            quantization_config=quant if quant is not None else None,
            dtype=torch.float16 if self.cfg.device=="cuda" else torch.float32,
            low_cpu_mem_usage=True, token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            local_files_only=local_only
        )
        if self.cfg.device!="cuda":
            self.mdl.to(self.cfg.device)

    @torch.no_grad()
    def gen(self, p_text: str) -> str:
        fs = "\n\n".join([f"Example {i+1}:\nPassage: {ep}\nQuery: {eq}" for i,(ep,eq) in enumerate(self.auto_shots)])
        prompt = (
            "You are a search query generator for document retrieval. Given a passage P and three human-written examples <P,Q>, "
            "write ONE specific, answerable, entity-rich query Q for the new passage. "
            "Do not copy phrases. Prefer 'how/why/under what conditions/compare'. 10–20 words. Single sentence.\n\n"
            f"{fs}\n\nNow generate a query for the following passage:\nPassage: {p_text}\nQuery:"
        )
        inp = self.tok(prompt, return_tensors="pt").to(self.cfg.device)
        out = self.mdl.generate(**inp, max_new_tokens=self.cfg.llama_max_new_tokens,
                                temperature=self.cfg.llama_temperature, do_sample=True, top_p=self.cfg.llama_top_p,
                                pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        q = text.split("Query:")[-1].strip().split("\n")[0].strip()
        if not q.endswith("?"): q += "?"
        return q

# -------------------- 专家打分 --------------------
def expert_scores(idx:int, cid:int, cluster_sizes, neighbor_counts, max_neighbors,
                  tfidf_norm, tfidf_min, tfidf_max, NI_all, mu_NI, sd_NI,
                  dist_to_center, dmin, dmax, sqrt_neighbor:bool):
    s1 = neighbor_counts[idx] / max(1, max_neighbors)
    if sqrt_neighbor: s1 = math.sqrt(max(0.0, s1))
    denom = max(1e-8, tfidf_max)
    s2 = 1.0/(1.0 + math.exp(-((tfidf_norm[idx]*denom - tfidf_min)/denom)))
    ni = NI_all[idx]
    s3 = 1.0 if sd_NI<=1e-8 else math.exp(-((ni - mu_NI)**2)/(2*(sd_NI**2)))
    if cluster_sizes[cid] == 1:
        s4 = 0.85
    else:
        s4 = (dmax - dist_to_center[idx]) / max(1e-8, (dmax - dmin))
        s4 = float(np.clip(s4, 0.0, 1.0))
    return [float(s1), float(s2), float(s3), float(s4)]

# -------------------- r1~r4 --------------------
@torch.no_grad()
def rewards(q:str, p_text:str, prow:int, cid:int, labels:np.ndarray,
            D:torch.Tensor, tok, mdl, cfg:Cfg, covered:set,
            cluster_uncovered:Dict[int,set], A_texts:List[str]):
    q_emb = encode_texts(tok, mdl, [q], cfg)[0:1].to(D.device)
    p_emb = D[prow:prow+1].to(D.device)
    sims = (q_emb @ D.T).squeeze(0).cpu().numpy()

    # r1: 指数贴现 rank 奖励
    top_idx = np.argsort(-sims)[:cfg.topk].tolist()
    rnk = cfg.topk + 1
    for r, ridx in enumerate(top_idx, start=1):
        if ridx == prow:
            rnk = r; break
    r1 = math.exp(-(rnk - 1)/cfg.r1_tau)

    # r2: 判别性（强负）——同簇最相近 & 最近簇最相近
    cos_qp = float(F.cosine_similarity(q_emb, p_emb).item())
    same_pool = np.where(labels==cid)[0].tolist()
    same_pool = [x for x in same_pool if x!=prow]
    near_cid = nearest_cluster(labels, D, cid)
    near_pool = np.where(labels==near_cid)[0].tolist()

    def argmax_sim(pool):
        if len(pool)==0: return prow
        arr = np.array(pool, dtype=int)
        vals = sims[arr]
        return int(arr[int(np.argmax(vals))])

    same_hard = argmax_sim(same_pool)
    near_hard = argmax_sim(near_pool)

    cos_same = float((q_emb @ D[same_hard:same_hard+1].T).item())
    cos_near = float((q_emb @ D[near_hard:near_hard+1].T).item())
    r2 = cos_qp * (1 - cos_same) * (1 - cos_near)

    # r3: Q-A 一致性（A 为可回答片段）
    A = A_texts[prow] if A_texts is not None else p_text[:200]
    A_emb = encode_texts(tok, mdl, [A], cfg)[0:1].to(D.device)
    cos_qA = float(F.cosine_similarity(q_emb, A_emb).item())
    cos_pA = float(F.cosine_similarity(p_emb, A_emb).item())
    r3 = cos_qA * cos_pA * (1 - abs(cos_qp - cfg.k_target))

    # r4: 覆盖增益（簇归一化 + 全局）
    neighbors = set(np.where(sims > cfg.sim_threshold)[0].tolist())
    global_new = len(covered.union({prow, *neighbors})) - len(covered)
    global_left = D.size(0) - len(covered)
    g_part = global_new / max(1, global_left)

    cset = set(np.where(labels==cid)[0].tolist())
    c_uncovered = cluster_uncovered[cid]
    add_c = len(c_uncovered.intersection({prow, *neighbors}))
    c_left = max(1, len(c_uncovered))
    c_part = add_c / c_left

    cluster_weight = len(cset)/float(D.size(0))
    r4 = cluster_weight * c_part + (1 - cluster_weight) * g_part

    return float(r1), float(r2), float(r3), float(r4), q_emb.cpu(), neighbors

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection_jsonl", required=True)
    ap.add_argument("--collection_emb_pt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--local_only", action="store_true")

    # 生成数量/聚类
    ap.add_argument("--target_num_q", type=int, default=1000)
    ap.add_argument("--auto_k", action="store_true")
    ap.add_argument("--align_to_k", action="store_true")

    # 相似/榜单
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--sim_threshold", type=float, default=0.58)

    # LLM
    ap.add_argument("--llama_model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    # 候选与去重
    ap.add_argument("--n_cands", type=int, default=5)
    ap.add_argument("--use_candidate_best", action="store_true")
    ap.add_argument("--use_dedup", action="store_true")
    ap.add_argument("--dedup_threshold", type=float, default=0.90)

    # 专家学习与正则
    ap.add_argument("--eta", type=float, default=0.30)
    ap.add_argument("--use_zscore", action="store_true")
    ap.add_argument("--z_temp", type=float, default=1.3)
    ap.add_argument("--use_floor", action="store_true")
    ap.add_argument("--w_floor", type=float, default=0.05)
    ap.add_argument("--use_entropy", action="store_true")
    ap.add_argument("--entropy", type=float, default=0.10)
    ap.add_argument("--hard_match", action="store_true")
    ap.add_argument("--lambda_hard", type=float, default=0.80)
    ap.add_argument("--hard_warmup_ratio", type=float, default=0.30)

    # 其它奖励
    ap.add_argument("--r1_tau", type=float, default=18.0)
    ap.add_argument("--k_target", type=float, default=0.7)

    # RL 扩展
    ap.add_argument("--use_advantage", action="store_true",
                    help="Use REINFORCE-style advantage (score - EMA baseline) for weight updates")
    ap.add_argument("--baseline_beta", type=float, default=0.05,
                    help="EMA baseline momentum for advantage; larger = faster tracking")
    ap.add_argument("--use_critic", action="store_true",
                    help="Enable a tiny linear critic V(S)=w·S+b (A2C) to form advantage")
    ap.add_argument("--critic_lr", type=float, default=0.02,
                    help="Learning rate for the linear critic")

    args = ap.parse_args()

    # 装配 cfg
    cfg = Cfg(
        collection_jsonl=args.collection_jsonl,
        collection_emb_pt=args.collection_emb_pt,
        out_dir=args.out_dir,
        target_num_q=args.target_num_q,
        auto_k=args.auto_k,
        align_to_k=args.align_to_k,
        topk=args.topk, sim_threshold=args.sim_threshold,
        llama_model=args.llama_model,
        llama_temperature=args.temp, llama_top_p=args.top_p,
        llama_max_new_tokens=args.max_new_tokens,
        load_in_8bit=args.load_in_8bit,
        n_cands=args.n_cands,
        dedup_threshold=args.dedup_threshold,
        r1_tau=args.r1_tau, k_target=args.k_target,
        eta=args.eta, use_zscore=args.use_zscore, z_temp=args.z_temp,
        use_floor=args.use_floor, w_floor=args.w_floor,
        use_entropy=args.use_entropy, entropy=args.entropy,
        hard_match=args.hard_match, lambda_hard=args.lambda_hard,
        hard_warmup_ratio=args.hard_warmup_ratio
    )
    cfg.llama_cache = args.cache_dir
    cfg.contriever_cache = args.cache_dir

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    print(f"[CFG] device={cfg.device}  n_cands={cfg.n_cands}  dedup={args.use_dedup}@{cfg.dedup_threshold}  "
          f"zscore={cfg.use_zscore}@{cfg.z_temp}  eta={cfg.eta}  floor={cfg.use_floor}@{cfg.w_floor}  "
          f"entropy={cfg.use_entropy}@{cfg.entropy}  hard_match={cfg.hard_match}@{cfg.lambda_hard}  "
          f"topk={cfg.topk}  sim_thr={cfg.sim_threshold}")

    # 1) 数据
    docs, D = load_docs_and_emb(cfg)
    texts = [d["doctext"] for d in docs]
    N = len(docs)
    print(f"Loaded {N} docs, dim={D.size(1)}")

    # 2) 选 K & 聚类（若未 auto_k，则 K=target_num_q）
    K = auto_select_k(D, seed=cfg.seed) if cfg.auto_k else cfg.target_num_q
    print(f"Auto-selected K = {K}" if cfg.auto_k else f"Fixed K = {K} (== target_num_q)")
    km = KMeans(n_clusters=min(K, N), random_state=cfg.seed, n_init=10)
    labels = km.fit_predict(D.cpu().numpy())
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    dists = torch.cdist(D, centers)
    dist_to_center = dists.min(1).values
    cluster_sizes = np.bincount(labels, minlength=centers.size(0))
    dmin, dmax = float(dist_to_center.min()), float(dist_to_center.max())

    # 3) 预处理
    tfidf_norm, tfidf_min, tfidf_max, tfidf_vec, _ = tfidf_stats(texts)
    NI_all = NI_proxy(texts); mu_NI, sd_NI = float(NI_all.mean()), float(NI_all.std() + 1e-8)
    neighbor_counts, max_neighbors = cos_neighbors(D, cfg.sim_threshold)

    # 3.1 A 片段
    A_texts = build_answer_snippets(texts, tfidf_vec, win_len=70)

    # 4) Contriever
    ctok, cmdl = load_contriever(cfg, local_only=args.local_only)

    # 5) few-shot + 生成器
    shots = auto_few_shots(texts, tfidf_vec, k=3)
    qgen = LlamaGen(cfg, auto_shots=shots, local_only=args.local_only)

    # 6) 输出/日志
    with open(os.path.join(cfg.out_dir, "prompts_used.txt"), "w", encoding="utf-8") as pf:
        pf.write("\n\n".join([f"Example {i+1}:\nPassage: {ep}\nQuery: {eq}" for i,(ep,eq) in enumerate(shots)]))
    weights_hist_path = os.path.join(cfg.out_dir, "weights_history.jsonl")
    open(weights_hist_path, "w").close()

    # 7) 生成规模
    total_to_gen = (centers.size(0) if cfg.align_to_k else cfg.target_num_q)
    if cfg.align_to_k:
        # K 对齐：每簇 1 条
        per_cluster_quota = np.ones(centers.size(0), dtype=int)
    else:
        # 不对齐：按簇大小分配
        per_cluster_quota = np.ceil(total_to_gen * (cluster_sizes / cluster_sizes.sum())).astype(int)
    selected_per_cluster = np.zeros_like(per_cluster_quota)

    # 覆盖集合（全局 + 分簇）
    covered = set()
    cluster_uncovered = {cid: set(np.where(labels==cid)[0].tolist()) for cid in range(centers.size(0))}

    # 语义去重库
    q_bank = torch.empty((0, D.size(1)))

    # 权重
    w = np.array(cfg.init_w, dtype=float)

    # RL 状态
    baseline = 0.0
    critic_w = np.zeros(4, dtype=float)
    critic_b = 0.0

    # 8) 主循环
    out_queries = os.path.join(cfg.out_dir, "generated_queries.jsonl")
    with open(out_queries, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=total_to_gen)
        produced = 0
        history_max_sims = []

        while produced < total_to_gen:
            # 8.1 选择当前进度最落后的簇
            candidates = np.where(selected_per_cluster < per_cluster_quota)[0]
            if len(candidates) == 0: break
            progress = selected_per_cluster[candidates] / np.maximum(1, per_cluster_quota[candidates])
            cid = int(candidates[np.argmin(progress)])

            rows_all = np.where(labels==cid)[0]
            if len(rows_all)==0:
                selected_per_cluster[cid] = per_cluster_quota[cid]; continue

            # 8.2 专家打分（簇内）
            S=[]; idxs=[]
            for ridx in rows_all:
                S.append(expert_scores(
                    idx=int(ridx), cid=cid, cluster_sizes=cluster_sizes,
                    neighbor_counts=neighbor_counts, max_neighbors=max_neighbors,
                    tfidf_norm=tfidf_norm, tfidf_min=tfidf_min, tfidf_max=tfidf_max,
                    NI_all=NI_all, mu_NI=mu_NI, sd_NI=sd_NI,
                    dist_to_center=dist_to_center, dmin=dmin, dmax=dmax,
                    sqrt_neighbor=cfg.sqrt_neighbor_score
                ))
                idxs.append(int(ridx))
            S=np.array(S, dtype=float)

            if cfg.use_zscore:
                S = (S - S.mean(0)) / (S.std(0)+1e-8)
                S = S * cfg.z_temp

            weighted = S @ w.reshape(-1,1)
            cand_idx = np.where(weighted == weighted.max())[0]
            i_local = int(np.random.choice(cand_idx))
            ridx = int(idxs[i_local])
            p_text = docs[ridx]["doctext"]

            # 8.3 候选生成 + 奖励 + 去重（动态阈值）
            best_q, best_score, best_qemb, best_rewards, best_neighbors = None, -1, None, None, None
            for _ in range(max(1, cfg.n_cands if args.use_candidate_best else 1)):
                q = qgen.gen(p_text)
                r1, r2, r3, r4, q_emb, neigh = rewards(q, p_text, ridx, cid, labels, D, ctok, cmdl, cfg, covered, cluster_uncovered, A_texts)

                # 动态 τ_dup: 用历史最大相似度的 75% 分位；簇内略严，跨簇略松
                tau = cfg.dedup_threshold
                if args.use_dedup and q_bank.size(0) > 0:
                    if len(history_max_sims)>4:
                        tau = float(np.quantile(np.array(history_max_sims), 0.75))
                        tau = float(np.clip(tau, 0.88, 0.95))
                # 与库最大相似度
                is_dup = False
                if args.use_dedup and q_bank.size(0) > 0:
                    sim = float((F.normalize(q_emb,p=2,dim=1) @ q_bank.T).max().item())
                    history_max_sims.append(sim)
                    if sim >= tau: is_dup=True

                if not is_dup:
                    score = (r1 + r2 + r3 + r4) / 4.0
                    if score > best_score:
                        best_q, best_score, best_qemb = q, score, q_emb
                        best_rewards, best_neighbors = (r1,r2,r3,r4), neigh

            # 若都判重复，取最高分兜底
            if best_q is None:
                q = qgen.gen(p_text)
                r1, r2, r3, r4, q_emb, neigh = rewards(q, p_text, ridx, cid, labels, D, ctok, cmdl, cfg, covered, cluster_uncovered, A_texts)
                best_q, best_score, best_qemb = q, (r1+r2+r3+r4)/4.0, q_emb
                best_rewards, best_neighbors = (r1,r2,r3,r4), neigh

            q = best_q
            r1, r2, r3, r4 = best_rewards

            # 覆盖更新（全局+簇）
            covered.add(ridx)
            if best_neighbors is not None:
                cluster_uncovered[cid] -= set([ridx]) # 自身
                cluster_uncovered[cid] -= set([i for i in best_neighbors if labels[i]==cid])
            else:
                cluster_uncovered[cid].discard(ridx)

            # 去重库
            if best_qemb is not None:
                q_bank = torch.cat([q_bank, F.normalize(best_qemb, p=2, dim=1).cpu()], dim=0)

            # 8.4 权重更新：两阶段（前 warmup 用硬配，后期软正则） + RL 优势项
            s_i = S[i_local]  # [s1..s4]
            t_ratio = (produced+1) / float(total_to_gen)
            alpha = cfg.lambda_hard if (cfg.hard_match and t_ratio <= cfg.hard_warmup_ratio) else 0.0

            # “硬配”把与四奖励对齐（r1..r4），“软配”按专家分数中心化
            reward_vec = np.array([r1,r2,r3,r4], dtype=float)
            soft_grad = s_i - s_i.mean()
            hard_grad = reward_vec - reward_vec.mean()
            grad = (1-alpha)*soft_grad + alpha*hard_grad

            # ---------- RL: advantage ----------
            if args.use_critic:
                # 线性价值函数 V(S)=w·S + b（A2C）
                v_pred = float(np.dot(critic_w, s_i) + critic_b)
                advantage = best_score - v_pred
                # TD(0) / 在线最小二乘式更新
                err = advantage
                critic_w += args.critic_lr * err * s_i
                critic_b += args.critic_lr * err
            elif args.use_advantage:
                # EMA baseline（REINFORCE）
                advantage = best_score - baseline
                baseline = (1.0 - args.baseline_beta) * baseline + args.baseline_beta * best_score
            else:
                advantage = best_score

            # 按优势项缩放更新幅度（adv<0 抑制当前偏好）
            delta = cfg.eta * advantage * grad

            w = np.clip(w + delta, 0.0, None)
            if cfg.use_floor: w = np.maximum(w, cfg.w_floor)
            w = w / w.sum()
            if cfg.use_entropy:
                uniform = np.ones_like(w)/len(w)
                w = (1-cfg.entropy)*w + cfg.entropy*uniform
                w = w / w.sum()

            with open(weights_hist_path, "a", encoding="utf-8") as wh:
                wh.write(json.dumps({
                    "step": produced+1, "weights": w.tolist(),
                    "reward": [float(r1),float(r2),float(r3),float(r4)],
                    "cid": int(cid), "doc_idx": int(ridx), "best_score": float(best_score),
                    "advantage": float(advantage),
                    "baseline": float(v_pred if args.use_critic else baseline)
                }, ensure_ascii=False) + "\n")

            # 8.5 写出
            fout.write(json.dumps({
                "docid": docs[ridx]["docid"],
                "doctext": p_text,
                "question": q
            }, ensure_ascii=False) + "\n")

            selected_per_cluster[cid] += 1
            produced += 1
            pbar.update(1)
        pbar.close()

    # 9) 统计文件
    stats = {
        "K": int(centers.size(0)),
        "num_docs": int(N),
        "generated": int(total_to_gen),
        "cluster_sizes": list(map(int, np.bincount(labels).tolist()))
    }
    with open(os.path.join(cfg.out_dir, "cluster_stats.json"), "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    with open(os.path.join(cfg.out_dir, "selected_docids.txt"), "w", encoding="utf-8") as sf:
        sf.write("\n".join([docs[i]["docid"] for i in sorted(list(covered))]))

    print(f"✅ Done: {total_to_gen} samples → {out_queries}")
    print("👉 直接把 generated_queries.jsonl 喂给训练/测试管线即可。")

if __name__ == "__main__":
    main()
