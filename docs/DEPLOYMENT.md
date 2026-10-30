# Deployment

Four ways to run pyplant, from simplest to most production-shaped.

| | what you get | needs |
| --- | --- | --- |
| [Workbench](#1-the-workbench) | the full GUI, one command | Python 3.11+ |
| [By hand](#2-the-services-by-hand) | the four services in separate terminals | Python 3.11+ |
| [Docker, one container](#3-docker-one-container) | the workbench, nothing installed locally | Docker |
| [Docker, four containers](#4-docker-four-containers) | the services on their own network, as in a plant | Docker Compose |

## Install

Python **3.11 or newer** is required; `pyplant.hmi.alarms` uses `enum.StrEnum`.

```bash
git clone https://github.com/mhmodir/pyplant.git
cd pyplant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml,ui,dev]"
```

The extras are independent:

| extra | adds | for |
| --- | --- | --- |
| *(none)* | numpy, pandas, pymodbus, asyncua, duckdb, fastapi, uvicorn, httpx | running the four services |
| `ml` | scikit-learn, pyarrow, joblib | generating datasets, training detectors |
| `ui` | dash, plotly | the workbench GUI |
| `dev` | pytest | the test suite |

With conda instead of venv:

```bash
conda create -n pyplant python=3.12 -y
conda activate pyplant
python -m pip install -e ".[ml,ui,dev]"
```

Or from the pinned-range requirements files:

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

Check it:

```bash
python -c "import pyplant.workbench.app; print('ok')"
pytest -q -m "not slow"            # about two minutes
```

## 1. The workbench

```bash
pyplant-workbench --workspace ./work/demo --base-port 5400
```

Open <http://127.0.0.1:8050>. The Quick start tab runs a scenario offline in a
few seconds with nothing else running. Press **Start stack** to launch the
plant, PLC, historian and HMI as subprocesses; the workbench then drives them
over HTTP.

| option | default | meaning |
| --- | --- | --- |
| `--workspace` | `./work/demo` | directory for config, datasets, models and history |
| `--base-port` | `5400` | first of six consecutive ports for the services |
| `--host` | `127.0.0.1` | interface the GUI binds to; `0.0.0.0` to expose it |
| `--port` | `8050` | GUI port |

Several people on one machine each need their own workspace and base port:

```bash
pyplant-workbench --workspace ./work/alice --base-port 5400 --port 8050
pyplant-workbench --workspace ./work/bob   --base-port 5500 --port 8051
```

## 2. The services by hand

The same four commands the workbench runs; it shows them on its Runtime tab.
Start them in this order, because each one connects to the one before.

```bash
python -m pyplant.services.plant     --port 5400 --control-port 5401
python -m pyplant.services.plc       --plant-port 5400 --port 5402 --opcua-port 5403
python -m pyplant.services.historian --endpoint opc.tcp://127.0.0.1:5403/pyplant/ \
                                     --db history.duckdb --api-port 5404
python -m pyplant.services.hmi       --endpoint opc.tcp://127.0.0.1:5403/pyplant/ \
                                     --history http://127.0.0.1:5404 --port 5405
```

Or all four from a script:

```bash
python examples/run_stack.py
```

Then the operator page is at <http://127.0.0.1:5405> and the fault API, with an
interactive form for every endpoint, at <http://127.0.0.1:5401/docs>.

Inject a fault from the command line:

```bash
# Linux and macOS
curl -X POST http://127.0.0.1:5401/api/fault \
     -H 'Content-Type: application/json' \
     -d '{"kind":"leak","id":"L1","magnitude":5e-5}'

# Windows PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:5401/api/fault -Method Post `
  -ContentType "application/json" -Body '{"kind":"leak","id":"L1","magnitude":5e-5}'
```

Or with `python examples/inject_fault.py`, which also prints the controller's
response once a second.

### Service options

| service | option | default |
| --- | --- | --- |
| plant | `--dt` | `0.05` s integration step |
| plant | `--speed` | `1.0` real-time factor |
| plant | `--start` | `steady` (or `empty`) |
| plant | `--no-noise` | instrument noise is on by default |
| plc | `--period` | `0.10` s scan period |
| plc | `--setpoint` | `300` mm |
| plc | `--no-opcua` | OPC UA is on by default |
| historian | `--heartbeat` | `30` s maximum gap between stored samples |
| historian | `--flush` | `1.0` s between database writes |
| hmi | `--poll` | `0.5` s |

### Training from the command line

```bash
pyplant-train --out artifacts --duration 3600
```

Generates the scenario grid, fits the PCA baseline and the autoencoder on
normal runs only, evaluates both on held-out runs, and writes
`dataset.parquet`, the fitted detectors, a manifest and `metrics.json` to
`artifacts/`. For a quick run: `--duration 1200 --magnitudes 1 --repeats 1`.

The command line trains those two. The workbench's Model tab offers all four
methods (z-score limits, PCA, isolation forest and the autoencoder) and
compares whichever you select.

## 3. Docker, one container

```bash
docker compose up --build
```

Open <http://localhost:8050>. One container runs the workbench, which
supervises the four services inside it. Only the GUI port is published; the
service ports stay internal. The workspace is kept in the `workbench-data`
volume, so datasets and trained models survive a restart.

Without Compose:

```bash
docker build -t pyplant .
docker run --rm -p 8050:8050 -v pyplant-data:/data pyplant
```

The container runs as an unprivileged user and has a health check on the GUI.

## 4. Docker, four containers

```bash
docker compose --profile services up --build plant plc historian hmi
```

Each service in its own container on a private network, reaching the others by
name (`plant`, `plc`, `historian`): a field device, a controller, a historian
and an operator station. There is no workbench in this mode; the HMI is the
interface.

| URL | what |
| --- | --- |
| <http://localhost:8080> | operator HMI |
| <http://localhost:8000/docs> | fault injection API |
| `opc.tcp://localhost:4840/pyplant/` | OPC UA, for an external client such as UaExpert |

Startup order is handled by `depends_on` plus restart policies: if the PLC
comes up before the plant is listening it exits and is restarted, which is
what a real controller does when its field device is late. The historian and
HMI retry their OPC UA connection for about twenty seconds.

## Configuration

Each workspace keeps its settings in `config.json`, written on first use and
editable from the workbench or by hand:

| section | holds |
| --- | --- |
| `ports` | the six service ports |
| `speed`, `dt` | real-time factor and integration step |
| `plant` | tank areas, pump curve, valve sizing, orifice coefficients |
| `loop` | PID gains, setpoint, scan period |
| `data` | scenario grid size, window, stride, threshold quantile, seed |

Everything else in a workspace is generated and can be deleted:
`dataset.parquet`, `artifacts/`, `history.duckdb`, `jobs/` and `runs.jsonl`.
The `work/` directory is ignored by git for this reason.

## Platform notes

**Windows timer resolution.** `speed` multiplies how fast the plant runs, so it
divides the scan period. Windows schedules sleeps in steps of about 15.6 ms, so
above roughly `speed=4` the 100 ms scan period falls below what the OS can
deliver and `SCAN_OVERRUNS` climbs on the HMI. The loop still controls; the
scan engine is reporting, correctly, that it is missing deadlines. Linux and
macOS handle `speed=20` comfortably.

| platform | sleep granularity | safe speed |
| --- | --- | --- |
| Linux, macOS | about 1 ms | 20 or more |
| Windows | about 15.6 ms | 4 or less |

**Slow faults need speed.** The plant's open-loop time constant is 72 s. A
sticking valve takes several time constants to settle into a limit cycle, so at
1x it takes ten minutes or so to become obvious; at 10x, about one.

**Shutdown on Windows.** `terminate()` kills a process outright on Windows
rather than asking it to exit, so the historian may lose the last second or two
of buffered samples when the stack stops.

## Troubleshooting

**`TypeError: unsupported operand type(s) for |: 'type' and 'type'`**
You are on Python 3.9 or older. Check with `python --version`; with conda, the
prompt should name your environment, not `(base)`.

**`ERROR: File "setup.py" not found. Directory cannot be installed in editable mode`**
pip is older than 21.3. `python -m pip install --upgrade pip`, then reinstall.

**`ModuleNotFoundError: No module named 'pyplant.hist'`**, or any `pyplant.*` module
Either the project is not installed (`pip install -e .` from the directory
holding `pyproject.toml`), or an unrelated package on PyPI that is also named
`pyplant` is installed and shadowing it. Check with
`python -c "import pyplant; print(pyplant.__file__)"`; if it points into
`site-packages` and you used `-e`, run `pip uninstall pyplant` and reinstall.

**`conda-hook.ps1 cannot be loaded because running scripts is disabled`**
A PowerShell policy, not a pyplant problem. Run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and reopen the
terminal, or use Command Prompt instead.

**VS Code runs the wrong Python**
`Ctrl+Shift+P`, *Python: Select Interpreter*, and pick the environment you
installed into. The Run button and new terminals both follow that choice.

**`ports already in use`** when starting the stack
Another workspace or a stale process holds them. Use a different
`--base-port`, or find the process: `lsof -i :5400` on Linux and macOS,
`netstat -ano | findstr 5400` on Windows.

**A port shows nothing in the browser**
Only the HTTP ports serve pages: the fault API, the historian API, the HMI and
the workbench. The Modbus and OPC UA ports speak binary protocols, so a browser
gets no response from them. That is expected.

**`{"detail":"Method Not Allowed"}`** opening `/api/fault`
That endpoint takes POST and a browser address bar sends GET. Use the form at
`/docs`, the Faults tab in the workbench, or one of the commands above.

**`curl` in PowerShell rejects `-H` and `-d`**
PowerShell aliases `curl` to `Invoke-WebRequest`. Use `curl.exe`, or
`Invoke-RestMethod` as shown above.

**`Resource In Use Error`** from DuckDB
Something other than the historian opened the database file. Only the
historian may write it; everything else should query the historian's HTTP API.

**`SCAN_OVERRUNS` keeps climbing**
`speed` is too high for the platform; see the timer note above.

**A sticking-valve or frozen-sensor fault does nothing**
Both need the loop to move. Leave instrument noise on (it is by default), raise
`speed`, and for a frozen level transmitter, change the setpoint after
injecting: the controller then chases a reading that cannot follow.

## Rebuilding the documentation

The PDFs in `docs/pdf/` are built from the LaTeX in `docs/manual/`:

```bash
cd docs/manual
latexmk -pdf main.tex          # the manual
latexmk -pdf quickguide.tex    # the quick-start guide
latexmk -pdf poster.tex        # needs the montserrat font package
cp main.pdf ../pdf/pyplant-manual.pdf
cp quickguide.pdf ../pdf/pyplant-quickguide.pdf
cp poster.pdf ../pdf/pyplant-poster.pdf
```

A TeX Live installation with `latexmk` and the `lmodern` fonts is enough for
the manual and the quick guide. The poster additionally needs `montserrat`,
which on Debian and Ubuntu is in `texlive-fonts-extra`.

The walkthrough animation is rendered from the manual's screenshots:

```bash
pip install opencv-python pillow
python docs/manual/make_animation.py     # writes pyplant_walkthrough.gif and .mp4
```

Only the GIF is committed. The MP4 is ignored by git; regenerate it, or attach
it to a GitHub release, if you need it.
