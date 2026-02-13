# Case Study 1 Review Bundle

This bundle contains the YAML scenarios and controller definitions requested for review.

## Included YAML configs
- configs/scenario_case_study1.yaml
- configs/scenario_paper_figure.yaml
- configs/scenario_disturbance_low.yaml
- configs/scenario_disturbance_medium.yaml
- configs/scenario_disturbance_high.yaml
- configs/scenario_disturbance_very_high.yaml

## Included controller code
- controllers/openloop.py
- controllers/pid.py
- controllers/pid_agentic.py
- controllers/agentic_supervisor.py
- core/pid_core.py

## Key functions (quick index)
- `controllers/openloop.py`: `OpenLoopController.compute_rpms(...)`
- `controllers/pid.py`: `PIDController.compute_rpms(...)`
- `controllers/pid_agentic.py`: `PIDAgenticController.compute_rpms(...)`
- `controllers/agentic_supervisor.py`: `AgenticSupervisor.step(...)`
- `core/pid_core.py`: `PIDCore.step(...)`

## Focused diff file
A focused diff for core controller/scenario files is provided in:
- `changes.patch`
