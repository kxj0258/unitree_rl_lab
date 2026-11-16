"""MuJoCo implementation of the frozen Go2 actor observation/control contract."""
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from .teacher_bank import ObservationAdapter, gait_clock, sha256_file


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class PolicyRuntime:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if sha256_file(directory / "policy.onnx") != self.manifest["onnx_sha256"]:
            raise ValueError("export checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "policy.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        cfg = self.manifest["teacher_contract"]["network"]
        self.hidden = np.zeros((cfg["rnn_num_layers"], 1, cfg["rnn_hidden_dim"]), np.float32)

    def act(self, observation, reset=False):
        action, hidden = self.session.run(None, {"observation": observation,
                                                "hidden_in": self.hidden,
                                                "reset": np.array([reset], np.float32)})
        if not np.isfinite(action).all() or not np.isfinite(hidden).all():
            raise ValueError("non-finite policy output")
        self.hidden = hidden
        return action[0]


class Go2Mujoco:
    def __init__(self, scene_path, manifest, physics_dt=.005):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.entry = manifest["teacher_contract"]
        self.adapter = ObservationAdapter(self.entry)
        self.names = manifest["joint_names"]
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names])
        if np.any(self.joint_ids < 0):
            raise ValueError("MuJoCo model is missing a policy joint")
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.motor_ids = []
        for jid in self.joint_ids:
            motors = np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)
            if len(motors) != 1:
                raise ValueError("expected one motor per policy joint")
            self.motor_ids.append(int(motors[0]))
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_id < 0 or self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("expected Go2 free root named base")
        control = self.entry["control"]
        self.dt = control["step_dt"]
        if physics_dt <= 0 or abs(round(self.dt / physics_dt) * physics_dt - self.dt) > 1e-9:
            raise ValueError("physics_dt must divide control_dt")
        self.decimation = round(self.dt / physics_dt)
        self.model.opt.timestep = physics_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.default_q = np.array(control["default_joint_pos"])
        self.actuator = manifest["actuator"]
        if self.actuator["min_delay"] or self.actuator["max_delay"]:
            raise ValueError("delayed actuators require an explicit delay buffer")
        # External PD supplies damping; do not double count Menagerie's passive
        # damping=2. Frictionloss is a Coulomb approximation, recorded in report.
        self.model.dof_damping[self.vadr] = 0
        self.model.dof_frictionloss[self.vadr] = .01
        self.model.actuator_ctrllimited[self.motor_ids] = 0  # our torque-speed envelope is authoritative
        self.initial_height = manifest["initial_base_height"]
        self.action_cfg = control["actions"]["JointPositionAction"]
        self.last_action = np.zeros(12, np.float32)
        self.steps = 0
        self.scan = self.entry["height_scanner"]
        pattern = self.scan["pattern_cfg"]
        if self.scan["ray_alignment"] != "yaw" or pattern["direction"] != [0., 0., -1.]:
            raise ValueError("M2 scanner requires yaw-aligned downward rays")
        x = np.arange(-pattern["size"][0]/2, pattern["size"][0]/2 + 1e-9, pattern["resolution"])
        y = np.arange(-pattern["size"][1]/2, pattern["size"][1]/2 + 1e-9, pattern["resolution"])
        xx, yy = np.meshgrid(x, y, indexing="xy" if pattern["ordering"] == "xy" else "ij")
        self.rays = np.stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)), -1) + self.scan["offset"]["pos"]
        if len(self.rays) != 117:
            raise ValueError("height scan grid must have 117 rays")
        self.reset()

    def reset(self, yaw=0., joint_jitter=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0., 0., self.initial_height]
        self.data.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        self.data.qpos[self.qadr] = self.default_q + (0. if joint_jitter is None else joint_jitter)
        self.last_action.fill(0)
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

    @property
    def yaw(self):
        r = self.data.xmat[self.base_id].reshape(3, 3)
        return np.arctan2(r[1, 0], r[0, 0])

    def velocity_body(self):
        # Isaac's root velocities refer to the root body's COM, rotated into
        # the link frame. mjOBJ_BODY uses the COM-centred spatial velocity.
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        return rotation.T @ velocity[3:], rotation.T @ velocity[:3]

    def command(self, vx, heading_target):
        cfg = self.entry["command"]
        if vx < 0:
            raise ValueError("reverse command is outside the contract")
        limit = min(cfg["max_yaw_rate"], vx / cfg["minimum_turn_radius"],
                    cfg["maximum_lateral_acceleration"] / max(vx, 1e-6)) if vx >= cfg["minimum_turning_speed"] else 0.
        yaw_rate = np.clip(cfg["heading_control_stiffness"] * wrap_angle(heading_target-self.yaw), -limit, limit)
        return np.array([vx, 0., yaw_rate], np.float32)

    def height_scan(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        base = self.data.xpos[self.base_id]
        starts = self.rays @ rotation.T + base
        heights = []
        # Model collision/visual robot geoms use groups 3/2. Group 0 contains
        # the world terrain; robot surfaces must not contaminate the scan.
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for start in starts:
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(self.model, self.data, start, np.array([0., 0., -1.]),
                                     groups, True, -1, hit)
            if distance < 0:
                raise ValueError("height scan missed terrain; do not silently zero-fill")
            hit_z = start[2] - distance
            # cfg.offset.z only offsets ray starts. Isaac sensor.pos_w is the
            # tracked base pose, not base+20m. Default height_scan offset=.5m.
            heights.append(base[2] - hit_z - .5)
        return np.asarray(heights, np.float32)

    def observation(self, command):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        _, omega = self.velocity_body()
        raw = {"base_ang_vel": omega, "projected_gravity": rotation.T @ np.array([0., 0., -1.]),
               "velocity_commands": command, "joint_pos_rel": self.data.qpos[self.qadr]-self.default_q,
               "joint_vel_rel": self.data.qvel[self.vadr], "last_action": self.last_action,
               "height_scan": self.height_scan()}
        terms = self.entry["observations"]["policy"]
        if any(t["name"] == "gait_clock" for t in terms):
            params = next(t["params"] for t in terms if t["name"] == "gait_clock")
            raw["gait_clock"] = gait_clock(torch.tensor([self.steps]), self.dt, torch.tensor([command[0]]), params)[0].numpy()
        tensors = {k: torch.from_numpy(np.asarray(v, np.float32).reshape(1, -1)) for k, v in raw.items()}
        return self.adapter.build(tensors).numpy()

    def advance(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("invalid action")
        cfg = self.action_cfg
        limits = np.asarray(cfg["clip"])
        target = np.clip(action * cfg["scale"] + cfg["offset"], limits[:, 0], limits[:, 1])
        motor = self.actuator
        for _ in range(self.decimation):
            velocity = self.data.qvel[self.vadr]
            torque = motor["stiffness"] * (target-self.data.qpos[self.qadr]) - motor["damping"] * velocity
            max_torque = np.where(velocity*torque > 0, motor["Y1"], motor["Y2"])
            factor = np.clip((motor["X2"]-np.abs(velocity))/(motor["X2"]-motor["X1"]), 0, 1)
            torque = np.clip(torque, -max_torque*factor, max_torque*factor)
            torque -= motor["Fs"] * np.tanh(velocity/motor["Va"]) + motor["Fd"] * velocity
            self.data.ctrl[self.motor_ids] = torque
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.last_action = action.copy()
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise ValueError("non-finite MuJoCo state")

    def termination(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        if np.arccos(np.clip(rotation[2, 2], -1., 1.)) > 1.0:
            return "bad_orientation"
        for contact in self.data.contact:
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            if 0 in bodies and self.base_id in bodies:
                return "base_contact"
        return None
