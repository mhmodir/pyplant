"""Start the whole stack from a script, without the workbench GUI.

    python examples/run_stack.py

Starts the plant, the PLC, the historian and the operator HMI as four
subprocesses, in dependency order, and stops them again on Enter. The command
line each service was started with is printed, so you can see exactly what the
workbench does when you press Start stack.
"""

from pyplant.workbench.supervisor import Supervisor
from pyplant.workbench.workspace import Workspace


def main() -> None:
    workspace = Workspace("./work/script", base_port=5400)
    supervisor = Supervisor(workspace)
    try:
        for service in supervisor.start():
            print(f"{service.name:<10} pid={service.pid:<7} port={service.port}")
            print(f"           {service.command}")
        print()
        print(f"operator HMI   {supervisor.hmi_url}")
        print(f"fault API      {supervisor.plant_control_url}/docs")
        print(f"history API    {supervisor.history_url}/api/health")
        input("\nrunning - press Enter to stop ")
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
