# cyberflight

A (very) digital driving range for golf simulators.

> **Early alpha** — in active development. Expect breaking changes, missing features, and rough edges.

![Empty Range](./screens/empty_range.png "Empty Range")

## Launch Monitor Integration

cyberflight receives live shot data from [flighthook](https://github.com/divotmaker/flighthook), an open-source golf automation bus. flighthook connects to your launch monitor and broadcasts shot data over a local WebSocket — cyberflight subscribes as a client to render ball flights in real time.

### Supported launch monitors (via flighthook)

- **FlightScope Mevo+** — WiFi, via [ironsight](https://github.com/divotmaker/ironsight)
- **FlightScope Gen2** — WiFi, via ironsight

### Setup

1. Start flighthook and connect to your launch monitor
2. Start cyberflight — it connects to flighthook at `ws://localhost:5880/frp`; for
   flighthook on another machine, pass `--frp ws://<host>:5880/frp`. While
   flighthook is not running, cyberflight retries every 5 s. If something
   accepts the connection but never answers, cyberflight keeps waiting and
   logs a warning every 30 s; restart cyberflight to reconnect.
3. Send it
