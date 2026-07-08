import verifiers.v1 as vf

from symbolic_tool_calling_v1.models import BenchmarkState, BenchmarkTask


class SymbolicState(vf.State):
    task: BenchmarkTask | None = None
    environment: BenchmarkState | None = None
