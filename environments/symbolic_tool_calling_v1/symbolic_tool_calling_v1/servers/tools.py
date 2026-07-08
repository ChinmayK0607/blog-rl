import verifiers.v1 as vf

from symbolic_tool_calling_v1.engine import apply_action
from symbolic_tool_calling_v1.models import Action
from symbolic_tool_calling_v1.state import SymbolicState


class SymbolicTools(vf.Toolset[vf.ToolsetConfig, SymbolicState]):
    TOOL_PREFIX = "symbolic"

    def _run(self, tool: str, **arguments: str) -> str:
        if self.state.task is None or self.state.environment is None:
            raise RuntimeError("symbolic environment was not initialized")
        state, observation = apply_action(
            self.state.task,
            self.state.environment,
            Action(tool=tool, arguments=arguments),
        )
        self.state.environment = state
        return observation

    @vf.tool
    def inspect(self) -> str:
        """Inspect the current room, its exits, and visible objects."""
        return self._run("inspect")

    @vf.tool
    def move(self, direction: str) -> str:
        """Move through an exit in the named direction."""
        return self._run("move", direction=direction)

    @vf.tool
    def pickup(self, item: str) -> str:
        """Pick up a visible item in the current room."""
        return self._run("pickup", item=item)

    @vf.tool
    def use(self, item: str) -> str:
        """Use an item or activate a visible switch."""
        return self._run("use", item=item)

    @vf.tool
    def query(self) -> str:
        """Query a terminal in the current room for access information."""
        return self._run("query")

    @vf.tool
    def submit(self, system: str, code: str) -> str:
        """Submit an access code to the named target system."""
        return self._run("submit", system=system, code=code)


if __name__ == "__main__":
    SymbolicTools.run()
