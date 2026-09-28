import json, re, random, sys, glob, os

path = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
random.seed(0)

# 收集所有玩家 train.jsonl 或单个 merged
if os.path.isdir(path):
    files = sorted(glob.glob(os.path.join(path, "*/train.jsonl")))
    files = [f for f in files if not f.split("/")[-2].endswith("_sample")]
    lines = []
    for f in files:
        with open(f) as fh:
            lines += [l for l in fh if l.strip()]
else:
    lines = [l for l in open(path) if l.strip()]

print(f"路径: {path}")
print(f"总轨迹: {len(lines)}")
samp = random.sample(lines, min(n_sample, len(lines)))

rx = re.compile(r"<\|action_start\|>(-?\d+) (-?\d+) (-?\d+)(.*?)<\|action_end\|>", re.S)
total = 0
zero_mouse = 0
zero_and_haskey = 0      # dx=dy=0 但有按键(走路等, 合理保留)
zero_and_empty = 0       # dx=dy=0 且无任何按键(纯空闲, 应被95%过滤,剩5%)
nonzero = 0
for l in samp:
    r = json.loads(l)
    for m in r["messages"]:
        if m["role"] != "assistant":
            continue
        mt = rx.search(m["content"])
        if not mt:
            continue
        dx, dy = int(mt.group(1)), int(mt.group(2))
        rest = mt.group(4)  # action_sep 分隔的按键部分
        # 去掉 action_sep 和空格, 看是否有任何按键字母
        keys = re.sub(r"<\|action_sep\|>", " ", rest).strip()
        has_key = len(keys) > 0
        total += 1
        if dx == 0 and dy == 0:
            zero_mouse += 1
            if has_key:
                zero_and_haskey += 1
            else:
                zero_and_empty += 1
        else:
            nonzero += 1

print(f"总step: {total}")
print(f"dx=dy=0 占比: {zero_mouse/total*100:.1f}%")
print(f"  ├─ 其中【有按键】(走路/技能等, 合理): {zero_and_haskey/total*100:.1f}%  (占零位移的 {zero_and_haskey/zero_mouse*100:.1f}%)")
print(f"  └─ 其中【完全空】(无键无鼠, 纯空闲残留): {zero_and_empty/total*100:.1f}%  (占零位移的 {zero_and_empty/zero_mouse*100:.1f}%)")
print(f"有鼠标移动 占比: {nonzero/total*100:.1f}%")
