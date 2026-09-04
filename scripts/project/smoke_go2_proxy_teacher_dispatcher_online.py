#!/usr/bin/env python3
"""Online seven-terrain proxy-teacher dispatcher smoke; performs no training."""

import argparse, hashlib, json, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher

RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--registry", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--seed", type=int, default=20260829, help="Smoke scenario seed, not a training seed.")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0: parser.error("--steps must be positive")
output = Path(args.output).expanduser().resolve()
if output.exists(): parser.error(f"refusing to overwrite: {output}")
args.headless = True
app = AppLauncher(args).app

print("[ONLINE_SMOKE] Isaac Sim application launched", flush=True)
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from rsl_rl.modules import ActorCriticRecurrent  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
print("[ONLINE_SMOKE] Runtime imports and Unitree task registration complete", flush=True)

TASK = "Unitree-Go2-Velocity-MultiTerrain-HeightScan"
TERRAINS = ["flat", "random_rough", "pyramid_slope", "inverted_pyramid_slope",
            "pyramid_stairs", "inverted_pyramid_stairs", "boxes"]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def columns(cfg):
    names=list(cfg.sub_terrains); props=[float(x.proportion) for x in cfg.sub_terrains.values()]
    total=sum(props); cumulative=[]; running=0
    for x in props: running+=x/total; cumulative.append(running)
    return [names[next(i for i,u in enumerate(cumulative) if c/int(cfg.num_cols)+.001<u)] for c in range(int(cfg.num_cols))]

def policy(actor,critic,actions,hidden,device):
    p=ActorCriticRecurrent(obs=TensorDict({"policy":torch.zeros(1,actor,device=device),"critic":torch.zeros(1,critic,device=device)},batch_size=[1]),
        obs_groups={"policy":["policy"],"critic":["critic"]},num_actions=actions,
        actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],
        critic_hidden_dims=[512,256,128],activation="elu",init_noise_std=.8,rnn_type="gru",rnn_hidden_dim=hidden,rnn_num_layers=1).to(device)
    p.eval(); return p

def main():
    reg_path=Path(args.registry).resolve(); reg=json.loads(reg_path.read_text())
    device=args.device
    print(f"[ONLINE_SMOKE] Parsing task configuration: {TASK}", flush=True)
    env_cfg=parse_env_cfg(TASK,device=device,num_envs=70,use_fabric=not args.disable_fabric,entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed=args.seed; env_cfg.curriculum.terrain_levels=None; env_cfg.curriculum.lin_vel_cmd_levels=None
    gen=env_cfg.scene.terrain.terrain_generator; gen.seed=args.seed; gen.curriculum=True; gen.num_rows=10; gen.num_cols=20
    env_cfg.scene.terrain.max_init_terrain_level=9
    print("[ONLINE_SMOKE] Creating 70-environment seven-terrain matrix", flush=True)
    base=gym.make(TASK,cfg=env_cfg)
    raw=base.unwrapped; colnames=columns(gen); types=[]; levels=[]
    for terrain in TERRAINS:
        cs=[i for i,n in enumerate(colnames) if n==terrain]
        if not cs: raise RuntimeError(f"missing columns: {terrain}")
        types += [cs[i%len(cs)] for i in range(10)]; levels += list(range(10))
    raw.scene.terrain.terrain_types[:]=torch.tensor(types,device=raw.device)
    raw.scene.terrain.terrain_levels[:]=torch.tensor(levels,device=raw.device)
    raw.scene.terrain.env_origins[:]=raw.scene.terrain.terrain_origins[raw.scene.terrain.terrain_levels,raw.scene.terrain.terrain_types]
    base.reset()
    print("[ONLINE_SMOKE] Environment reset complete", flush=True)
    if isinstance(base.unwrapped,DirectMARLEnv): base=multi_agent_to_single_agent(base)
    env=RslRlVecEnvWrapper(base,clip_actions=100.0); obs=env.get_observations()
    if obs["policy"].shape!=(70,162) or obs["critic"].shape!=(70,177): raise RuntimeError(f"observation schema: {obs.batch_size}")
    unique={}; env_ids={}
    for tid,cfg in reg["teachers"].items():
        ck=Path(cfg["checkpoint"]); actual=sha(ck)
        if actual!=cfg["sha256"]: raise RuntimeError(f"SHA mismatch: {tid}")
        p=policy(162,177,12,256,raw.device); payload=torch.load(ck,map_location=raw.device,weights_only=False)
        p.load_state_dict(payload["model_state_dict"],strict=True); unique[tid]=p
        env_ids[tid]=torch.tensor(
            [j*10+k for j,t in enumerate(TERRAINS) for k in range(10) if reg["terrain_dispatch"][t]==tid],
            device=raw.device,
        )
    print(f"[ONLINE_SMOKE] Loaded {len(unique)} unique proxy checkpoints", flush=True)
    student=policy(162,177,12,256,raw.device)
    stats={t:{"samples":0,"mse_sum":0.,"teacher_abs_sum":0.,"student_abs_sum":0.,"terminations":0} for t in TERRAINS}
    nonfinite=0
    with torch.no_grad():
        for step in range(args.steps):
            student_actions=student.act_inference(obs)
            targets=torch.empty_like(student_actions)
            for tid,p in unique.items():
                ids=env_ids[tid]; targets[ids]=p.act_inference(obs[ids])
            nonfinite += int((~torch.isfinite(student_actions)).sum()+(~torch.isfinite(targets)).sum())
            sq=(student_actions-targets).square().mean(dim=1)
            for j,t in enumerate(TERRAINS):
                ids=slice(j*10,(j+1)*10); s=stats[t]; s["samples"]+=10
                s["mse_sum"]+=float(sq[ids].sum()); s["teacher_abs_sum"]+=float(targets[ids].abs().mean(dim=1).sum()); s["student_abs_sum"]+=float(student_actions[ids].abs().mean(dim=1).sum())
            obs,_,dones,_=env.step(student_actions); student.reset(dones)
            for tid,p in unique.items(): p.reset(dones[env_ids[tid]])
            for j,t in enumerate(TERRAINS): stats[t]["terminations"]+=int(dones[j*10:(j+1)*10].sum())
            if step == 0 or (step + 1) % 16 == 0 or step + 1 == args.steps:
                print(f"[ONLINE_SMOKE] rollout step {step + 1}/{args.steps}", flush=True)
    if nonfinite: raise RuntimeError(f"non-finite values: {nonfinite}")
    for s in stats.values():
        n=s["samples"]; s["mean_imitation_mse"]=s.pop("mse_sum")/n; s["mean_teacher_abs_action"]=s.pop("teacher_abs_sum")/n; s["mean_student_abs_action"]=s.pop("student_abs_sum")/n
    result={"status":"pass","task":TASK,"scenario_seed":args.seed,"training_seed":None,"steps":args.steps,"num_envs":70,"terrain_id_in_student_observation":False,"student_actions_drive_environment":True,"unique_teacher_count":4,"registry_sha256":sha(reg_path),"nonfinite_count":nonfinite,"per_terrain":stats}
    output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); env.close()

if __name__=="__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        print("[ONLINE_SMOKE] FAILED with traceback:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        app.close()
    raise SystemExit(exit_code)
