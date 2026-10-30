# Design notes

How pyplant was built, one layer at a time, and what each layer taught.

This is an engineering log rather than a manual. Every section records a
decision and the reason for it, and a good number record a bug that only
appeared once something was measured: a PID that slammed the valve shut on its
first scan, a timer that never fired, a stiction model that produced no limit
cycle, a detector judged against the wrong instrument. Those are the parts
worth reading.

For how to use the software, see the [manual](pdf/pyplant-manual.pdf). For how
it is put together, see [ARCHITECTURE.md](ARCHITECTURE.md). Commands and ports
are in [DEPLOYMENT.md](DEPLOYMENT.md); they are deliberately not repeated here.

## Contents

| step | layer | the lesson |
| --- | --- | --- |
| [0](#step-0-the-tag-dictionary) | tag dictionary | one source of truth for every scale factor |
| [1](#step-1-the-plant) | physics | close the pump/valve algebraic loop analytically |
| [2](#step-2-the-runner) | stepper | three rates, and a real zero-order hold |
| [3](#step-3-the-controller) | PID and PLC logic | initialise from feedback; protect on flow, not pressure |
| [4](#step-4-the-scan-engine) | scan engine | drift-free scheduling; count scans, not seconds |
| [5](#step-5-over-the-wire) | Modbus TCP | no block may be written by two parties |
| [6](#step-6-opc-ua) | OPC UA | a server is a view of the store, not a copy |
| [7](#step-7-the-historian) | historian | carry values forward; never interpolate |
| [8](#step-8-the-hmi) | HMI and alarms | ISA-18.2 states; a transient must stay visible |
| [9](#step-9-faults-and-scenarios) | faults | noise is not decoration |
| [10](#step-10-dataset-and-detector) | detection | split by run; judge sensor faults by what the loop does |

## Step 0: the tag dictionary

`pyplant/tags.py` is the single source of truth for every signal. Each tag
carries its description, engineering unit and range, alarm limits, Modbus table
and address, integer scale factor, owning process, and OPC UA browse path.

Every other module in the project imports its I/O definitions from here.
Nothing else is permitted to hard-code an address or a scale factor. This is
the cheapest possible insurance against the failure mode where a level reads
4520 in the PLC and 452.0 in the historian.

### Wire representation

Analog values go into 16-bit holding registers as `round(value * scale)`,
transmitted as an unsigned two's-complement word. Scale factors are chosen per
tag so the engineering range fits inside int16, and this is checked at import,
not at runtime. Out-of-range values are clamped rather than wrapped, so a bad
write produces a pegged reading instead of a sign flip.

Booleans go into coils as 0 or 1.

### Ownership

`owner` names the process that is the source of truth for a tag. A process
writes only what it owns and treats everything else as read-only. That rule is
what stops the plant from writing controller outputs and the HMI from writing
measurements.

### Blocks

Addresses are grouped into declared `Block`s so clients can poll a whole span
in one Modbus transaction. Blocks are declared explicitly rather than derived
from the tags, and import fails if any block contains a gap, if two blocks
overlap, or if a tag sits in a block owned by a different process.

## Step 1: the plant

`pyplant/core/hybrid_tank.py` holds the physics as pure functions. SI units
throughout, no sockets, no sleeping, no global state.

    PMP101 --> FCV101 --> [ Tank 1 ] --> orifice --> [ Tank 2 ] --> drain
                             |                          |
                           LT101                      LT102

A centrifugal pump feeds tank 1 through an equal-percentage control valve with
a 2 s actuator lag. Tank 1 drains by gravity into tank 2, which drains to
atmosphere. LIC101 will hold tank 1 level by stroking FCV101; tank 2 is
uncontrolled, so faults reach it with a lag.

Valve flow depends on pump pressure and pump pressure depends on flow, so the
two are solved together. With a quadratic pump curve the loop closes in closed
form, which means no iteration and no stiffness at a 50 ms step.

Three fields on `Params` are fault knobs sitting at their healthy values:
`pump_gain` for impeller wear, `cda_12_gain` for fouling of the interconnect,
and `cda_leak` for a hole in tank 1. Step 9 mutates them; nothing else should.

`measure()` is the only function here that knows a tag name, and the only
place SI turns into engineering units.

### Sizing

| quantity | value |
| --- | --- |
| tank area, each | 0.020 m^2 |
| overflow lip | 600 mm |
| pump shutoff / max flow | 60 kPa / 25 L/min |
| design point | 66% valve, 300 mm, 10 L/min, 50 kPa |
| open-loop tau at design point | 72 s |
| usable level range | roughly 30 to 580 mm (valve 0.30 to 0.79) |

`PT101` was respanned from 0-400 to 0-100 kPa once the pump curve was fixed,
with its high-high limit at 55 kPa just under the 60 kPa deadhead.

## Step 2: the runner

`pyplant/runners/base.py` holds `Simulation`, the one loop body in the project.
The fast runner drives it with no clock to generate datasets; the real-time
runner in step 5 drives the same object against a wall clock and serves it over
Modbus. Neither owns the physics, so the live plant and the training data
cannot drift apart.

Three rates, deliberately different:

| rate | default | what it governs |
| --- | --- | --- |
| `dt` | 50 ms | integration accuracy |
| `control_period` | 100 ms | what the controller sees |
| `log_period` | 1 s | what the historian stores |

Inputs are held between scans, so a controller written against `Simulation`
gets a real zero-order hold at its own sample time. Skipping that is the usual
reason a PID looks better in simulation than it behaves on a PLC.
`control_period` must be a whole number of `dt`, checked at construction, and
`t` accumulates from an integer step count rather than by repeated addition.

Measured throughput is about **1900x real time**: a simulated hour in under two
seconds, so a few hundred scenarios is a coffee break rather than an overnight
job.

`Trajectory` records engineering units and includes `FCV101_OP` and
`PMP101_CMD` alongside the measurements, because the diagnosis model needs the
controller output as well as the process value.

## Step 3: the controller

`pyplant/plc/pid.py` is a velocity-form PID; `pyplant/plc/program.py` is the
PLC program as a pure function of an input image table plus retained state;
`pyplant/plc/harness.py` wires the two to the plant in-process, and is replaced
by Modbus at step 5.

Velocity form because the output *is* the integrator state. Anti-windup and
bumpless transfer both fall out of that rather than being bolted on: when the
output clamps the increment is discarded, so it leaves the limit on the first
scan the error reverses; in manual the output is whatever the operator wrote,
so returning to auto continues from there. Derivative acts on filtered
measurement, not error, so a setpoint step produces no kick.

Tuning is IMC at the design point (K about 15.4 mm/%, tau 72 s, lambda 13 s):
`kc = 0.35`, `ti = 72 s`, no derivative. Measured closed-loop response to a
50 mm setpoint step is no overshoot, 2% settling in about 40 s, zero offset.

### Protection

| function | trigger | action |
| --- | --- | --- |
| `ALM_LT101_HI` | LT101 above 500 mm | alarm |
| `ALM_LT101_LOLO` | LT101 below 50 mm | alarm |
| `ALM_PT101_HIHI` | PT101 above 58 kPa | alarm |
| `ILK_PMP101` | LT101 above 550 mm, or FT101 below 1 L/min for 15 s | latched trip |

All thresholds are read from the tag dictionary, so a respanned instrument
cannot leave a stale limit behind in the logic. Alarms carry hysteresis; the
minimum-flow trip carries an on-delay.

### Three things that only appeared once the loop closed

**The controller must be initialised from valve feedback.** A velocity-form PID
starting at 0% slams shut the valve of a plant already running at 66%, the pump
dead-heads, and the whole rig trips on the first scan. The program now seeds the
output from ZT101 on first scan.

**Pump protection is minimum flow, not high pressure.** A centrifugal pump
against a shut valve rises to shutoff head, 60 kPa here, so any high-pressure
trip below that fires on legitimate low-output excursions. The real hazard is
no flow, so the trip is low flow on a 15 s delay and PT101 high is an alarm.

**PLC timers count scans, not seconds.** Accumulating `elapsed += 0.1` reaches
0.9999999999999999 after ten scans, and a timer compared with `>=` against a
1.0 s preset never fires at all.

### The property the whole project exists to demonstrate

With a leak opened under a lined-out loop, the level returns to setpoint
exactly and the controller output moves by about 10%:

| | LT101 | LIC101_OP |
| --- | --- | --- |
| before | 300.00 mm | 66.18% |
| after | 300.00 mm | 75.80% |

A diagnosis model fed the process value alone cannot see this fault at all.
That is why `feature_tags()` includes OP and the alarm bits.

## Step 4: the scan engine

`pyplant/plc/scan.py` turns `Program` into a PLC. Every scan does three things
in order: read inputs, execute logic against the frozen image, write outputs.
All I/O sits behind the `IOLink` protocol, so step 5 changes the transport to
Modbus without touching the engine or the program.

`pyplant/clock.py` holds `Ticker`, the drift-free scheduler both the PLC and
the plant run on. `await asyncio.sleep(period)` in a loop schedules the next
wake relative to when the last one *finished*, so every microsecond of work is
added permanently to the phase. `Ticker` computes absolute targets from a fixed
origin instead:

    target(n) = origin + n * period / speed

Work eats into slack, never into phase. With a 70 ms body on a 100 ms period,
100 ticks land at exactly 10.000 s rather than 17.000 s, and
`test_schedule_does_not_drift_when_the_body_burns_time` asserts that to
1e-12.

`pyplant/runners/realtime.py` runs the plant as its own task on its own clock.
The zero-order hold that `Simulation` faked in step 2 is now physical: the PLC
writes `sim.u` when it scans, and the plant keeps integrating with whatever it
last received. Both tasks share one asyncio loop and neither awaits mid-mutation,
which is the entire concurrency argument and stops holding the moment anyone
adds a thread.

### The engine measures itself

`SCAN_MS`, `SCAN_OVERRUNS` and `PLC_HEARTBEAT` are real tags, trended on the
HMI like any process value. `speed` is a real-time factor, so a one-hour
scenario runs in 36 seconds at `speed=100`.

Two behaviours worth knowing. The published statistics are always the previous
scan's, because this scan's duration is not known until its write completes,
which is what real controllers do. And when the engine simply cannot keep up,
`late_ms` grows without bound against the fixed grid while the overrun count
just increments once per scan; the lateness trend is the signal, not the count.

## Step 5: over the wire

Two processes now, talking Modbus TCP.

    [ plant node ]  <--- Modbus --->  [ PLC node ]  <--- Modbus --->  [ HMI ]
     physics + server                 client + server                  client
     port 5020                        port 5021

The plant is a Modbus slave, the way a field device is. It has no controller,
no alarms and no memory of intent: it applies whatever is currently in its
command registers, so if the PLC stops writing, the valve holds its last
position, which is what a real one does.

The PLC is both roles at once. It polls the plant below (`PLANT_MEAS`,
`PLANT_STATUS`), writes commands down (`PLC_CMD`, `PLC_CMD_BITS`), and serves
the SCADA layer above. It also mirrors the plant measurements into its own
store so the HMI polls one address instead of two. Four round trips per scan,
measured at **0.7 ms** against a 100 ms budget.

`Program` and `ScanEngine` are untouched from steps 3 and 4. Only the `IOLink`
changed.

### No block may be written by two parties

The faceplate was one block through step 4, holding both the setpoint the
operator writes and the readback the PLC publishes. That loses updates: an
operator write landing between the PLC's read and its publish is overwritten
100 ms later. The faceplate is now split.

| block | direction |
| --- | --- |
| `OPER_SP` HR 20-21, `OPER_CMD` COIL 11-13 | HMI writes, PLC reads |
| `PLC_FACE` HR 22-24, `PLC_DIAG`, `PLC_ALARMS` | PLC publishes, HMI reads |

`Owner.OPER` now exists for exactly this, and the invariant that the writable
tag set equals the operator-owned tag set is asserted in the test suite.

### Reads are by block, never by tag

One `read_holding_registers` for six measurements is one round trip; six
separate reads are six. `TagStore` and `TagClient` only expose whole-block
operations, and a partial publish is refused rather than leaving a stale word
beside a fresh one inside what a client reads as a single snapshot.

### pymodbus is pinned to 3.8.x

The 3.15 line is midway through replacing the datastore with `SimData` /
`SimDevice` and no longer exposes a way to update served values from outside
the server, which is precisely what a live plant does twenty times a second.
3.8.6 has the stable `ModbusSlaveContext` API.

## Step 6: OPC UA

The boundary. Below it everything is registers and scale factors; above it the
historian, HMI and inference service speak tag names with units attached, and
none of the three imports a Modbus symbol.

The address space is generated from `TAGS`, never hand-listed. Each tag's
`area` becomes its folder path:

    Plant/Unit1/LT101          Plant/Unit1/FT101
    PLC/LIC101/LIC101_PV       PLC/LIC101/LIC101_SP     (writable)
    PLC/Diagnostics/SCAN_MS    PLC/Alarms/ILK_PMP101

Every analog node carries `EngineeringUnits` and `EURange` built from its
declared span, so `LT101` reports `("mm", 0.0, 600.0)` and `LIC101_ERR` reports
`("mm", -600.0, 600.0)`. A client can scale a trend correctly without being
told anything about the plant, and a test asserts the metadata matches the tag
dictionary for every analog tag rather than for a hand-picked few.

The server is a **view of the same `TagStore`** the Modbus server holds, not a
second process image. Values are pushed outward on a refresh tick, only for
nodes that actually changed, and client writes are pushed back into the store.

### Two write paths onto one tag

An operator setpoint can arrive over OPC UA while a refresh is midway through
pushing the old store value outward, and the naive implementation silently
reverts the write about one refresh period later. `refresh()` therefore tracks
the last value each writable tag was *seen* to hold in the store, and only
overwrites a writable node when the store changed underneath it — a real
external write — never merely because a refresh happened to run at the wrong
moment. `test_a_client_write_survives_the_next_refresh` runs twenty refresh
cycles against a fresh setpoint to pin it.

### Subscriptions

`OpcClient.subscribe` is push-on-change at a publishing interval. This is why
the historian in step 7 is not a poller.

## Step 7: the historian

`pyplant/hist/db.py` is a DuckDB store in long format, one row per sample:
`(ts, tag, value, quality)`. Wide tables look tidier until someone adds a tag
and needs a migration. Everything that wants a rectangle asks `get_window()`
for one.

`pyplant/services/historian.py` subscribes over OPC UA rather than polling,
buffers notifications, and flushes in batches. Row-at-a-time inserts are the
usual way to make DuckDB look slow.

### Resampling is last-value-carried-forward, never interpolation

Samples arrive by exception, so between two rows the signal was *held*, not
drifting. Interpolating would invent a ramp on a valve position and fractional
values on an alarm bit, and a model trained on that learns the resampler
instead of the plant. DuckDB's `ASOF JOIN` gives step-hold semantics directly,
which is most of why it is here.

    db.get_window(["LT101", "LIC101_OP"], t0, t1, freq=1.0)

Values carry forward from before the window opens too, so a tag that has been
quiet for an hour still has a value at the first grid instant.

### Exception reporting plus a heartbeat

A subscription already pushes only on change. What it cannot give you is an
anchor for a tag quiet since before the window, so each tag is also re-stored
after `heartbeat` seconds of silence. Store on change, and never let the gap
exceed a maximum.

### `covered_span`, not `span`

`span()` is the extent of the whole database, and a window opened there almost
always starts with nulls: tags report milliseconds apart, so the earliest
sample belongs to exactly one of them. `covered_span(tags)` returns the
interval over which *every* requested tag has a value, which is what step 11
should ask for.

### Timestamps are wall clock

As a real historian's are. At `speed > 1` the recorded timeline is compressed
by that factor. Step 10's training data comes from the fast runner in plant
time and never touches this database.

## Step 8: the HMI

`pyplant/hmi/alarms.py` is the ISA-18.2 state machine, `pyplant/hmi/app.py` is
a FastAPI service in front of the OPC UA server and the historian, and
`pyplant/hmi/page.html` is the operator page: an SVG mimic with live tank
levels, an LIC101 faceplate, a trend drawn from `get_window`, and the alarm
list. The HMI holds no process state of its own.

### Alarms are a state machine, not a boolean

A flag that turns a row red collapses four situations into two:

| state | meaning |
| --- | --- |
| `NORMAL` | quiet, nobody needs to act |
| `UNACK_ALARM` | active, nobody has looked at it |
| `ACK_ALARM` | active, somebody has taken responsibility |
| `RTN_UNACK` | it cleared before anyone looked |

`RTN_UNACK` is the point of the whole thing. A condition that comes and goes
between two operator glances leaves no trace in a flag-based list, and the
transient nobody saw is very often the interesting one. Here it stays visible
until acknowledged.

Shelving is a deliberate, expiring silence with the reason recorded, capped at
eight hours. Unshelving restores the state the condition justifies *now*, not
the one it had when it was shelved: an alarm that cleared while shelved comes
back as `RTN_UNACK`, because it did happen and somebody should know.

### Two rules the API enforces

Only tags the dictionary marks writable can be written, and the write goes to
OPC UA rather than any local copy — the HMI is a client of the control system,
not a participant in it. Setpoints outside their span are rejected with a 422
rather than silently clamped, so a fat-fingered 3000 mm produces a visible
error instead of a value the operator never notices.

`/api/reset` pulses the reset bit rather than setting it: a latched interlock
needs an edge, not a level.

### A display may reference points that have no publisher

`ALM_HEALTH` has no source until the inference service exists in step 11. The
HMI probes its tag list at connect, keeps what the server actually publishes,
and reports the rest as `missing`. A display configured for a point its
controller does not expose should degrade to a blank field, not take the page
down.

## Step 9: faults and scenarios

Faults are objects applied to a running plant, never edits to the physics.
They enter at exactly one of three places, and which one is the most important
thing about each:

| entry point | faults |
| --- | --- |
| parameter | `Leak`, `Fouling`, `PumpWear` |
| command | `ValveStiction` |
| signal | `SensorBias`, `SensorDrift`, `SensorStuck`, `SensorNoise` |

Injection lives in `Simulation.integrate()`, the one stepper both runners
share, so the same fault behaves identically in a dataset run and on the live
plant node. `sim.true_measurements` is what the instruments would read if none
of them were lying; `sim.measurements` is what the control system sees.

A transmitter reading 30 mm high and a genuine 30 mm rise produce identical
numbers in `LT101`. They differ only in what the rest of the plant does, which
is what makes diagnosis a real problem rather than a lookup.

Every fault carries a start, an optional end, and a ramp. Ramp length is not
cosmetic: a set of scenarios containing only step changes trains a detector
that is useless on the faults that creep.

### Instrument noise is not decoration

Every scenario carries `DEFAULT_NOISE`, applied as `benign` faults — recorded
in the fault set, never counted as a positive label. Without it a lined-out
loop never moves its valve at all, so **stiction produces no limit cycle and
cannot be detected**, and a detector trained on noiseless signals learns a
threshold that collapses on the first real transmitter it meets.

### Stiction measured

Deadband 10%, slip 7%, injected into a lined-out loop at constant setpoint:

| | LT101 peak-to-peak | OP peak-to-peak | mean \|OP - ZT101\| |
| --- | --- | --- | --- |
| healthy | 2.65 mm | 0.92 % | 0.12 % |
| sticking | 30.1 mm | 11.6 % | 4.44 % |

The command-versus-stem gap is the signature that identifies it, and it is
invisible to anything looking only at `LT101`.

### The cascade worth demonstrating

`sensor_stuck` freezes `LT101`. The controller chases a reading that cannot
move, winds the valve shut, starves the pump, and the minimum-flow interlock
trips at t = 1985 s. One frozen instrument, and the loop destroys itself. No
part of that was scripted.

### Ground truth

`run_scenario` returns the trajectory, a per-sample label frame, and an event
table with every fault's start, end, ramp and magnitude. `grid()` sweeps each
mode across magnitudes with randomised onsets and per-run seeds, so a model
cannot learn "the fault starts at 24 minutes" instead of learning the fault.

### Live injection

`create_control_app(node)` exposes fault injection on the plant, deliberately
separate from anything the control system can see. An HMI or a diagnosis model
must not be able to discover a fault except by observing the plant.

## Step 10: dataset and detector

58 runs of one hour each, generated in a few minutes at roughly 1900x real
time, then the detectors fitted on normal data only and measured on held-out
runs. The project started with two detectors, PCA and an autoencoder; z-score
limits and an isolation forest were added later, so the four are now compared
side by side. Current numbers are in the [README](../README.md#results).

The baseline matters more than it looks. A learned detector's numbers mean
nothing without a classical method beside them, and if the autoencoder cannot
beat PCA on this plant, the honest report says so.

### Physics-informed channels do the isolation

Three channels are computed rather than measured, each because a
first-principles argument says the fault lives in a difference no single tag
contains:

| channel | argument | isolates |
| --- | --- | --- |
| `FLOW_IMBALANCE` | FT101 - FT102 is zero by conservation of mass | leaks |
| `HEAD_RESIDUAL` | PT101 minus what the pump curve predicts at that flow | pump wear |
| `OP_MINUS_ZT` | command minus stem position, zero on a healthy valve | stiction |

Giving a model the residual of a physical relationship is cheaper than making
it learn the relationship.

How reliably each channel is picked depends on the method. Stiction is found
through `OP_MINUS_ZT` by all four, every time. Leaks are attributed to the mass
balance in all twelve runs by three methods but only eight by the autoencoder,
and PCA never attributes pump wear to the pump curve. The per-method table is
in the [README](../README.md#physics-does-the-isolation). The strongest
detector not being the strongest isolator was the most useful thing adding the
second pair of methods revealed.

### Split by scenario, never by sample

Windows overlap, and neighbouring windows from one run are nearly identical. A
random sample split puts window 41 in training and window 42 in test; the model
scores brilliantly on data it has already seen and the number is not slightly
optimistic, it is meaningless. Whole runs are split, and each run has its own
seed, magnitude and onset, so held-out runs differ in the ways that matter
operationally.

Training uses normal runs only. A plant has years of normal operation and a
handful of documented failures, so a supervised classifier would be trained on
the rarest thing it will ever see and could not recognise a mode nobody
labelled. Fault data measures the result and never fits it.

### Metrics an operator would recognise

Window accuracy is close to useless: a fault left on for half a run makes it
mostly a measure of how long the fault was on. Instead: **false alarms per
day** on held-out normal runs, **detection delay** in seconds from the declared
onset, and **isolation** against the channel physics names.

Detections require three consecutive flagged windows. A detector that flickers
above threshold one window in twenty has a fine-looking detection rate and is
unusable — and the same persistence rule counts false alarms, so both numbers
describe the same detector. Raw window flag rate is 0.14%; persistent
excursions on normal runs are zero.

### Two methodology bugs the numbers exposed

**Isolation was judged too early.** Fouling ramps over twenty minutes, and five
minutes in it is still a transient that implicates the controlled level rather
than the downstream one the settled fault points at. The ramp now travels with
the dataset and isolation waits for it to finish.

**The expectation for sensor faults was wrong.** In a closed loop a level
transmitter reading 25 mm high is driven back to setpoint by the controller:
the reading looks perfect and the true level sits 25 mm low. The signature
appears everywhere *except* the faulty instrument, so "the top channel should
be LT101" is exactly backwards. Sensor modes are reported by consistency rather
than scored against a mistaken expectation.

### The health index

Raw reconstruction error has no stable scale — retrain and it changes
completely. `HEALTH_INDEX` pins the threshold at 50 with a decade per 50
points, so an alarm limit set by an alarm engineer survives a retrain.
