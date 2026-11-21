# Leg bring-up plan: assembled legs moving on the gantry

Goal: both legs fully assembled, hung on the gantry (no torso), and every joint following a scripted
trajectory sent on `/leg/<side>/joint_targets` through `joint_command`. Walking, IMU and RL are out
of scope (listed at the end).

Status as of 2026-11-16, from `main` (1133f33a) and the GitHub issues and PRs. Hardware status comes
from issue comments, the newest from 2026-11-05. **Check the hardware boxes against the real legs
before using this.**

## Critical path at a glance

```mermaid
flowchart LR
    classDef crit fill:#ffe0e0,stroke:#d00000,stroke-width:3px,color:#5a0000
    classDef goal fill:#cfe2ff,stroke:#0a58ca,stroke-width:3px,color:#052c65

    A["Motors and screws<br/>delivered"]:::crit ==> B["Legs<br/>assembled"]:::crit
    B ==> I["Harness fastened,<br/>full range of motion"]:::crit
    C ==> D["Selected motor drive<br/>answer on the bus"]:::crit
    D ==> E["Directions set,<br/>calibrated on gantry"]:::crit
    E ==> F["Dry run through<br/>joint_command"]:::crit
    F ==> G["MIT gains<br/>#483"]:::crit
    G ==> H["One joint<br/>at a time"]:::crit
    H ==> I["All drives <br/> answer on bus"]:::crit
    I ==> J["Scripted trajectory,<br/>both legs, video"]:::goal
```

The long pole is hardware. Software (#477) can start today and finishes before the motors and screws
arrive if it is started now. Two unknowns could each add their own long step: the **motor-model
question** (D0) and the **ankle linkage** (D1). Answer both this week.

## Full dependency graph

```mermaid
flowchart TD
    classDef done fill:#d4edda,stroke:#28a745,color:#155724
    classDef open fill:#fff3cd,stroke:#b58900,color:#533f03
    classDef gap fill:#ffffff,stroke:#dc3545,color:#721c24,stroke-dasharray:6 3
    classDef crit fill:#ffe0e0,stroke:#d00000,stroke-width:3px,color:#5a0000
    classDef risk fill:#ffe0e0,stroke:#d00000,stroke-width:3px,color:#5a0000,stroke-dasharray:6 3
    classDef goal fill:#cfe2ff,stroke:#0a58ca,stroke-width:4px,color:#052c65

    subgraph MECH["Mechanical"]
        M0["Ankle FEA, ankle/shin hardstops,<br/>printed bearing substitutes<br/>#318 #320 #445"]:::done
        M1["All 12 leg motors delivered<br/>#478"]:::crit
        M2["Leg screws delivered<br/>#478"]:::crit
        M3["Both legs assembled,<br/>every joint turns by hand<br/>through its full range"]:::crit
        M4["Hip hardstops<br/>#321"]:::open
        M5d["Gantry hanging interface designed<br/>#322"]:::done
        M5["Build the stand: pelvis plate<br/>on the gantry<br/>NO ISSUE"]:::gap
        M6["Legs hung on gantry,<br/>feet clear of the floor"]:::crit
    end

    subgraph ELEC["Electrical (#478)"]
        E2["Shared CAN + power harness,<br/>XT60 branch per motor"]:::done
        E1["Connector BOM for motors<br/>without XT30 2+2: order, solder"]:::open
        E3["Set CAN IDs with CubeMars tool<br/>L 40-45, R 56-61"]:::open
        E4["120 ohm termination<br/>at both ends"]:::open
        E5["Harness fastened behind legs,<br/>slack through full range"]:::crit
        E6["48 V supply sized for 12 leg drives<br/>+ hardware E-stop in reach<br/>NO ISSUE"]:::gap
        E9["Bus check: all 12 drives answer<br/>on the leg CANable<br/>candump, /interfacing/motorFeedback"]:::crit
    end

    subgraph CFG["Motor config and open questions"]
        D0["D0: confirm motor per joint<br/>hip yaw AK60-39 or AK80-64?<br/>hip roll AK80-64: AK 2.0 or V3, kt?<br/>NO ISSUE (asked in #522 #531 #538)"]:::risk
        S0["Fix mit_profiles.yaml and<br/>leg yaml to match;<br/>enable hip roll 41/57"]:::crit
        D1["D1: does the ankle linkage couple<br/>pitch and roll motors? ratio?<br/>NO ISSUE"]:::risk
        S8["If coupled: motor-to-joint<br/>ankle transform in joint_command<br/>NO ISSUE"]:::gap
    end

    subgraph SW["joint_command leg support (#477)"]
        S1["LegPose.msg, leg topic skeleton,<br/>leg yaml, calibrate --limb leg<br/>#498 #522 #531"]:::done
        X3["Merge #543 / #545 fail-loudly first<br/>(same files as the leg work)"]:::open
        S4["Load legs from yaml; refuse null<br/>direction and zero MIT gains;<br/>duplicate-ID check"]:::open
        S5["legPoseToMotorCmds: same layers<br/>as the arm (clamp, low-pass,<br/>rate limit, calibration)"]:::open
        S6["legPoseCallback: reject<br/>non-finite angles, log"]:::open
        S7["Unit tests + offline SIL;<br/>arms unchanged with legs off"]:::open
    end

    subgraph BUP["Bring-up on the gantry"]
        B1["Set direction per joint by hand,<br/>motor vs URDF"]:::crit
        B2["calibrate_arm.py --limb leg<br/>--side left / right"]:::crit
        B3["Dry run: joint_command output<br/>matches the measured pose"]:::crit
        B4["MIT gains per joint: hold<br/>horizontal and extended<br/>#483"]:::crit
        B5["One joint at a time,<br/>small slow moves"]:::crit
        G["GOAL: both legs follow a<br/>scripted trajectory, on video<br/>#477"]:::goal
    end

    subgraph NOTNEEDED["Not needed for this goal"]
        X1["Right leg IDs: #530 says 50-55,<br/>main (#522) says 56-61"]:::open
        X2["Merge #530 two-bus can_node<br/>(arms + legs at the same time)"]:::open
    end

    M0 --> M3
    M1 ==> M3
    M2 ==> M3
    M1 --> E3
    M1 --> D0
    D0 ==> S0
    S0 --> E9
    M3 ==> E5
    M2 --> E5
    E1 --> E5
    E2 --> E5
    E5 ==> E9
    E3 --> E9
    E4 --> E9
    E6 --> E9
    M3 ==> M6
    M5d --> M5
    M5 --> M6
    E9 ==> B1
    B1 ==> B2
    M6 ==> B2
    S1 --> B2
    X3 -.-> S4
    S1 --> S4
    S4 --> S5
    S4 --> S6
    S5 --> S7
    S6 --> S7
    S7 --> B3
    B2 ==> B3
    B3 ==> B4
    B4 ==> B5
    M4 --> B5
    D1 -.-> S8
    S8 -.-> B5
    B5 ==> G
    X1 --> X2
```

**Legend:** thick arrows and red nodes = critical path. Dashed red = unknowns that could join the
critical path. Green = done. Yellow = open, not critical. Dashed white = needed but has no issue yet.

## Steps and where they stand

| Node | Step | Tracked in | Assignee | Status / blocker |
|---|---|---|---|---|
| M1, M2 | Leg motors and screws delivered | #478 (comment 2026-11-04) | none | Not all motors had arrived; screws pending. #477 (2026-11-04) says "the leg is built". Confirm which is true now. |
| M3 | Both legs assembled | none | none | Waits on M1, M2. |
| M4 | Hip hardstops | #321 | junomu25 | Open. Needed before full-range hip moves (B5), not before small moves. |
| M5 | Build the gantry stand / pelvis plate | none (#322 closed, design only) | none | **No issue for the build.** Without a torso, the hips need something to bolt to. |
| E1, E3, E4, E5 | Connectors, CAN IDs, termination, routing | #478 (due 2026-11-07, overdue) | none | Harness branches done. IDs wait on M1. Routing waits on M3 and screws. |
| E6 | 48 V supply and E-stop for the legs | none | none | **No issue.** The bus bar work (#471, #475) only covers the 10 arm motors. |
| D0, S0 | Confirm motor models; fix `mit_profiles.yaml` | questions in #522, #531, #538 | none | Hip roll 41/57 is commented out, so can_node won't drive it. Hip yaw model is unknown, and a wrong one makes the MIT scaling wrong. |
| D1, S8 | Ankle linkage coupling | none (#538 uses a placeholder ratio of 1.5) | none | If the two ankle motors move pitch and roll together, `joint_command` needs a transform before the ankles can move. |
| S1 | Leg msg, topic skeleton, yaml, calibration script | #498, #522, #531 | merged | Done 2026-11-14. |
| S4-S7 | Leg path in `joint_command` | #477 (TODOs in `joint_command_node.cpp` and `joint_command_core.cpp`) | juliejyyang, AydanLing, reneeyyx, anitajiang8 | Not started. `legPoseToMotorCmds` returns no commands and `legs:` throws at startup. Can start today. |
| X3 | Fail-loudly PRs | #543, #545 | none | Open. They edit the same loader and core code that S4/S5 change. Merge them first or expect conflicts. |
| B1-B3 | Directions, calibration, dry run | #477 | | Waits on E9, M6, S7. |
| B4 | MIT gains for legs | #483 (due 2026-11-13, overdue) | none | Depends on #480 (closed, merged into #477) and #482. |
| B5, G | Single-joint moves, then scripted video | #477 | | Final step. |
| X1, X2 | Two-bus can_node | #530 (approved, open); #482 closed though #530 is not merged | sardormirk | **Not on the critical path.** For legs-only bring-up, point the single-bus `can_node` (`params.yaml` `device_path`) at the leg CANable. Before #530 merges, fix its right-leg IDs (50-55) to match `main` (56-61). |

## Decisions to make now

1. **Motor inventory (D0).** Record the actual model and firmware of every leg motor, then fix
   `mit_profiles.yaml` and `leg_actuators.yaml` to match. File an issue for this.
2. **Ankle linkage (D1).** Ask mechanical whether ankle pitch and roll each have their own motor
   or are coupled through the linkage, and what the ratio is. If coupled, file an issue for the
   `joint_command` transform. It may then take longer than the rest of the software.
3. **One bus or two for the legs.** At `joint_command`'s 50 Hz, 12 drives on one 1 Mbit/s bus use
   about 16 % of it, so one shared leg bus (#477, #478) is fine for this goal. The 250 Hz walking
   loop (#196) needs one bus per leg (#197 comment). Revisit then.
4. **File the missing issues:** gantry stand build (M5), leg power and E-stop (E6), motor
   inventory (D0), ankle transform (S8, if needed).

## Safety gates (from `.claude/skills/real-hardware-safety`)

These are already in the graph; they are the steps not to skip:

- Every move goes through `joint_command`. Never publish `MotorCmd` directly.
- Hardware E-stop on the 48 V supply within reach for every powered test (E6).
- Calibrate hanging (M6 before B2). Do a read-only dry run before the first live command (B3).
- One joint at a time with the others `active: false`, small slow moves from the measured pose
  (B5). Raise `velocity_max` (30 deg/s) or gains only with bench data.

## After this goal: walking (not in the graph)

IMU on CAN (#195), 250 Hz leg loop (#196), actuator models and sysid (#335, #538), retraining on
the new leg (#153), sim2sim check (#293), CAN telemetry dashboard (#214), CAN-FD (#189, #193).
