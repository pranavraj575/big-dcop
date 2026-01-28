"""

Miracle: guess randomly and hope we do alright


Example
^^^^^^^
::

    pydcop solve --algo miracle tests/instances/graph_coloring1.yaml


"""
import logging
import random
from pydcop.algorithms import ComputationDef
from pydcop.infrastructure.computations import (
    VariableComputation,
    DcopComputation,
)

def build_computation(comp_def: ComputationDef) -> DcopComputation:
    """Build a DSA computation

    Parameters
    ----------
    comp_def: a ComputationDef object
        the definition of the DSA computation

    Returns
    -------
    MessagePassingComputation
        a message passing computation that implements the DSA algorithm for
        one variable.

    """
    return MiracleComputation(comp_def=comp_def)


class MiracleComputation(VariableComputation):
    def __init__(self, comp_def):
        super().__init__(comp_def.node.variable, comp_def)

        assert comp_def.algo.algo == "miracle"
        self.constraints = comp_def.node.constraints

        self.current_assignment = {}
        self._start_handle = None
        self._tick_handle = None

    def on_start(self):
        value=random.choice(self._variable.domain)
        cost=self._variable.cost_for_val(value)
        self.value_selection(value, cost)
        self.finished()
        self.stop()

    def on_stop(self):
        if self._tick_handle:
            self.remove_periodic_action(self._tick_handle)
        else:
            self.logger.warning(
                f"Stopping a miracle computation {self.variable} that never really started ! "
                "no _tick_handle"
            )

    def on_pause(self, paused: bool):
        if not paused:
            # when resuming (i.e. leaving pause) we can simply drop any pending message
            # as A-DSA is asynchronous and periodic
            self._paused_messages_post.clear()
            self._paused_messages_recv.clear()
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Dropping all message from pause on {self.name}")
