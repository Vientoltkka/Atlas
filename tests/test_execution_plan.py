from __future__ import annotations

from core.planner import ExecutionPlan, ExecutionStep, Planner


def test_simple_goal_generates_single_execution_plan_step() -> None:
    planner = Planner()

    plan = planner.create_execution_plan("Lee el archivo README.md")

    assert isinstance(plan, ExecutionPlan)
    assert plan.goal == "Lee el archivo README.md"
    assert plan.estimated_steps == 1
    assert plan.required_tools == ("read_file",)
    assert plan.requires_confirmation is False
    assert plan.status == "planned"
    assert plan.ordered_steps == (
        ExecutionStep(
            id="step_1",
            description="Read requested file content.",
            tool="read_file",
            dependencies=(),
            status="pending",
            arguments={},
        ),
    )


def test_complex_goal_generates_ordered_execution_plan() -> None:
    planner = Planner()

    plan = planner.create_execution_plan(
        "Lee README.md y copia su contenido en resumen.txt"
    )

    assert plan.estimated_steps == 2
    assert plan.required_tools == ("read_file", "write_file")
    assert plan.requires_confirmation is True
    assert plan.ordered_steps[0].id == "step_1"
    assert plan.ordered_steps[0].tool == "read_file"
    assert plan.ordered_steps[0].dependencies == ()
    assert plan.ordered_steps[1].id == "step_2"
    assert plan.ordered_steps[1].tool == "write_file"
    assert plan.ordered_steps[1].dependencies == ("step_1",)
    assert "Multi-step plan must preserve dependency order." in plan.detected_risks
    assert (
        "Step step_2 uses confirmation-gated tool 'write_file'."
        in plan.detected_risks
    )


def test_planner_always_returns_execution_plan_for_unknown_goal() -> None:
    planner = Planner()

    plan = planner.create_execution_plan("Haz algo magico con mi ordenador")

    assert isinstance(plan, ExecutionPlan)
    assert plan.estimated_steps == 1
    assert plan.required_tools == ()
    assert plan.requires_confirmation is False
    assert plan.ordered_steps[0].tool == "direct_response"


def test_execution_plan_generation_does_not_execute_tools(tmp_path) -> None:
    target = tmp_path / "resumen.txt"
    planner = Planner()

    plan = planner.create_execution_plan(f"Escribe hola en {target}")

    assert plan.required_tools == ("write_file",)
    assert plan.requires_confirmation is True
    assert target.exists() is False


def test_execution_step_accepts_structured_arguments() -> None:
    step = ExecutionStep(
        id="step_1",
        description="Read file.",
        tool="read_file",
        arguments={"path": "README.md", "flag": True, "count": 2},
    )

    assert dict(step.arguments) == {
        "path": "README.md",
        "flag": True,
        "count": 2,
    }


def test_execution_step_arguments_default_is_independent() -> None:
    first = ExecutionStep("step_1", "First.", "read_file")
    second = ExecutionStep("step_2", "Second.", "read_file")

    assert first.arguments == {}
    assert second.arguments == {}
    assert first.arguments is not second.arguments


def test_execution_step_arguments_are_top_level_read_only() -> None:
    step = ExecutionStep(
        id="step_1",
        description="Read file.",
        tool="read_file",
        arguments={"path": "README.md"},
    )

    try:
        step.arguments["other"] = "value"  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("step arguments must be read-only")
