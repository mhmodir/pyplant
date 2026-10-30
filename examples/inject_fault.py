"""Inject a fault into a running plant and watch the controller respond.

    python examples/run_stack.py          # in one terminal
    python examples/inject_fault.py       # in another

Prints the level and the controller output once a second. The level dips and
recovers; the controller output settles about ten percent higher. The fault is
invisible in the measurement and lives entirely in the controller output,
which is the closed-loop masking effect a diagnosis model has to see through.
"""

import time

from pyplant.workbench.clients import StackClients
from pyplant.workbench.workspace import Ports

clients = StackClients(Ports.from_base(5400))

baseline = clients.state()["values"]
print(f"before   LT101 {baseline['LT101']:6.1f} mm   OP {baseline['LIC101_OP']:5.1f} %")

fault = clients.inject("leak", magnitude=5e-5, fault_id="demo_leak")
print(f"injected {fault['fault_id']} ({fault['kind']} on {fault['target']})\n")

try:
    for second in range(90):
        values = clients.state()["values"]
        print(f"t+{second:02d}s   LT101 {values['LT101']:6.1f} mm   "
              f"OP {values['LIC101_OP']:5.1f} %")
        time.sleep(1.0)
finally:
    clients.clear_faults()
    print("\nfault cleared")
