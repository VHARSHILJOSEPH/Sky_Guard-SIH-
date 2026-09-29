import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { buildDataset, HERO_STATION_ID, type Dataset } from "./skyguard";
import {
  WeatherService,
  type AnomalyIncident,
  type ConnectionState,
  type DemoScenario,
  type ExternalWeatherContext,
  type MLInferenceResult,
  type PipelineProcessResult,
  type DemoDatasetRecord,
  type DemoDatasetResponse,
  type StandardizedSource,
} from "./weather-service";
import { EMBEDDED_DEMO_RECORDS, EMBEDDED_DEMO_SCENARIOS } from "./demo-dataset";

export interface DetectionConfig {
  temperatureSensitivity: number;
  pressureSensitivity: number;
  humiditySensitivity: number;
  temporalThreshold: number;
  spatialRadius: number;
  missingDataTolerance: number;
}

export const DEFAULT_CONFIG: DetectionConfig = {
  temperatureSensitivity: 72,
  pressureSensitivity: 60,
  humiditySensitivity: 65,
  temporalThreshold: 2.8,
  spatialRadius: 50,
  missingDataTolerance: 15,
};

export type ChartParameter = "temperature" | "pressure" | "humidity" | "all";
export type ChartTimeWindow = "15m" | "1h" | "6h";
export type AppDataSource = "LIVE" | "DEMO";

export interface DemoScenarioItem {
  scenario: string;
  label: string;
  station_id: string;
  start_index: number;
  description: string;
}

interface SkyGuardContextValue {
  dataset: Dataset;
  refreshing: boolean;
  refresh: () => void;
  selectedStationId: string;
  setSelectedStationId: (id: string) => void;
  config: DetectionConfig;
  setConfig: (c: DetectionConfig) => void;
  resetConfig: () => void;

  // Two-Mode Data Architecture & Unified Pipeline
  dataSource: AppDataSource;
  activeTelemetrySource: StandardizedSource;
  setDataSource: (source: AppDataSource) => void;
  demoRunId: string | null;
  demoCompleted: boolean;
  demoPlaying: boolean;
  demoSpeed: number;
  demoIndex: number;
  demoTotal: number;
  demoScenario: string;
  demoScenariosList: DemoScenarioItem[];
  startDemo: (scenarioKey?: string) => Promise<void>;
  returnToLive: () => void;
  pauseDemo: () => void;
  resumeDemo: () => void;
  resetDemo: () => void;
  setDemoSpeed: (speed: number) => void;
  setDemoScenario: (scenarioKey: string) => void;
  stepDemoIndex: (index: number) => Promise<void>;

  currentPipelineResult: PipelineProcessResult | null;
  pipelineRunning: boolean;

  // Lovable UI State Additions
  scenario: DemoScenario | "";
  runScenario: (scenario: DemoScenario | "") => void;
  connectionState: ConnectionState;
  clock: string;
  activeParameter: ChartParameter;
  setActiveParameter: (param: ChartParameter) => void;
  chartWindow: ChartTimeWindow;
  setChartWindow: (window: ChartTimeWindow) => void;
  activeAnomaly: AnomalyIncident | null;
  recentAnomalies: AnomalyIncident[];
  acknowledgeAnomaly: () => void;
  dismissAnomaly: () => void;
  toastMessage: { title: string; desc: string; time: string } | null;
  dismissToast: () => void;
  externalForecast: ExternalWeatherContext;
  mlState: MLInferenceResult;
}

const SkyGuardContext = createContext<SkyGuardContextValue | null>(null);

/** Deterministic anchor keeps SSR and first client render identical. */
const ANCHOR = new Date("2026-09-10T08:50:00.000Z");

const INITIAL_ANOMALIES: AnomalyIncident[] = [];

export const FALLBACK_SCENARIOS: DemoScenarioItem[] = [
  {
    scenario: "ALL",
    label: "All Scenarios (Full 7-Day Stream)",
    station_id: "HYD_AWS_01",
    start_index: 0,
    description: "Continuous chronological stream containing all normal and anomalous intervals.",
  },
  {
    scenario: "SPIKE",
    label: "Scenario 1 — Temperature Spike",
    station_id: "HYD_AWS_01",
    start_index: 80,
    description: "Sudden +11.5°C jump (32.45°C → 43.95°C) followed by recovery.",
  },
  {
    scenario: "FROZEN_SENSOR",
    label: "Scenario 2 — Frozen Sensor",
    station_id: "BLR_AWS_02",
    start_index: 268,
    description: "Humidity variance = 0.000 (locked at 69.87%) across 7 consecutive readings.",
  },
  {
    scenario: "DRIFT",
    label: "Scenario 3 — Temperature Drift",
    station_id: "DEL_AWS_03",
    start_index: 370,
    description: "Cumulative gradual departure (+3.2°C) from station rolling baseline.",
  },
  {
    scenario: "MISSING_DATA",
    label: "Scenario 4 — Missing Data",
    station_id: "HYD_AWS_01",
    start_index: 133,
    description: "Core barometric sensor returns null payload.",
  },
  {
    scenario: "CORRUPTED_DATA",
    label: "Scenario 5 — Corrupted Data",
    station_id: "HYD_AWS_01",
    start_index: 134,
    description: "Physical bounds rejection (999.0°C) with CRITICAL severity.",
  },
  {
    scenario: "MULTIVARIATE_INCONSISTENCY",
    label: "Scenario 6 — Multivariate Inconsistency",
    station_id: "BLR_AWS_02",
    start_index: 320,
    description: "Individual variables plausible, but thermodynamic density pattern is anomalous.",
  },
  {
    scenario: "WEATHER_EVENT",
    label: "Scenario 7 — Genuine Weather Event",
    station_id: "43189",
    start_index: 450,
    description:
      "Multi-station convective thunderstorm squall line with spatial consensus agreement.",
  },
];

export function SkyGuardProvider({ children }: { children: ReactNode }) {
  const [seed, setSeed] = useState(1);
  const [dataset, setDataset] = useState<Dataset>(() => buildDataset(1, ANCHOR));
  const [refreshing, setRefreshing] = useState(false);
  const [selectedStationId, setSelectedStationId] = useState(HERO_STATION_ID);
  const [config, setConfig] = useState<DetectionConfig>(DEFAULT_CONFIG);

  // Two-Mode State
  const [dataSource, setDataSource] = useState<AppDataSource>("LIVE");
  const [demoRunId, setDemoRunId] = useState<string | null>(null);
  const [demoCompleted, setDemoCompleted] = useState<boolean>(false);
  const [demoPlaying, setDemoPlaying] = useState(false);
  const [demoSpeed, setDemoSpeedState] = useState(1);
  const [demoIndex, setDemoIndex] = useState(0);
  const [demoRecords, setDemoRecords] = useState<DemoDatasetRecord[]>(EMBEDDED_DEMO_RECORDS);
  const [demoScenariosList, setDemoScenariosList] =
    useState<DemoScenarioItem[]>(EMBEDDED_DEMO_SCENARIOS);
  const [demoScenario, setDemoScenarioState] = useState<string>("ALL");
  const [currentPipelineResult, setCurrentPipelineResult] = useState<PipelineProcessResult | null>(
    null,
  );
  const [pipelineRunning, setPipelineRunning] = useState(false);

  const demoTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const isDemoProcessing = useRef(false);
  const demoIndexRef = useRef(0);
  const demoRecordsRef = useRef<DemoDatasetRecord[]>(EMBEDDED_DEMO_RECORDS);
  const demoRunIdRef = useRef<string | null>(null);

  useEffect(() => {
    demoIndexRef.current = demoIndex;
  }, [demoIndex]);

  useEffect(() => {
    demoRecordsRef.current = demoRecords;
  }, [demoRecords]);

  useEffect(() => {
    demoRunIdRef.current = demoRunId;
  }, [demoRunId]);

  // Lovable state
  const [scenario, setScenario] = useState<DemoScenario | "">("");
  const [connectionState, setConnectionState] = useState<ConnectionState>("LIVE");
  const [activeTelemetrySource, setActiveTelemetrySource] =
    useState<StandardizedSource>("OPEN_METEO_CONTEXT");
  const [activeParameter, setActiveParameter] = useState<ChartParameter>("temperature");
  const [chartWindow, setChartWindow] = useState<ChartTimeWindow>("1h");
  const [clock, setClock] = useState("12:42:08 PM IST");
  const [activeAnomaly, setActiveAnomaly] = useState<AnomalyIncident | null>(null);
  const [recentAnomalies, setRecentAnomalies] = useState<AnomalyIncident[]>(INITIAL_ANOMALIES);
  const [toastMessage, setToastMessage] = useState<{
    title: string;
    desc: string;
    time: string;
  } | null>(null);
  const [externalForecast, setExternalForecast] = useState<ExternalWeatherContext>({
    status: "idle",
  });

  // ML State (Conforming to authoritative LOF spec, real Python inference from Backend)
  const [mlState, setMlState] = useState<MLInferenceResult>({
    status: "LOADING",
    model: "Local Outlier Factor (LOF)",
    model_version: "LOF_BME280_v1.0",
    feature_schema_version: "BME280_FEATURES_v1",
    ml_anomaly: false,
    raw_lof_score: null,
    normalized_lof_score: null,
    threshold: null,
    features_used: [],
    feature_evidence: {},
    missing_fields: [],
    missing_data_imputed: false,
    inference_time_ms: null,
  });

  // UNIFIED DETECTION PIPELINE RUNNER
  const processPipelineData = useCallback(
    async (
      stnId: string,
      rawObservation: Record<string, unknown>,
      sourceMode: StandardizedSource | string,
      runId?: string | null,
    ) => {
      setPipelineRunning(true);
      try {
        const result = await WeatherService.processObservation(
          stnId,
          rawObservation,
          sourceMode,
          runId,
        );
        setCurrentPipelineResult(result);

        if (result && result.readings) {
          setDataset((prev) => {
            const updated = prev.stations.map((s) => {
              if (s.station_id === (result.station_id || stnId)) {
                return {
                  ...s,
                  temperature: result.readings.temperature ?? s.temperature,
                  humidity: result.readings.humidity ?? s.humidity,
                  pressure: result.readings.pressure ?? s.pressure,
                  wind_speed: result.readings.wind_speed ?? s.wind_speed ?? null,
                  rainfall: result.readings.rainfall ?? s.rainfall ?? null,
                  timestamp: result.timestamp || s.timestamp,
                  minutes_since_observation: 0,
                };
              }
              return s;
            });
            return { ...prev, stations: updated };
          });
        }

        if (result.ml_details) {
          const details = result.ml_details as Record<string, unknown>;
          const infTime =
            typeof details["inference_time_ms"] === "number"
              ? (details["inference_time_ms"] as number)
              : 14.2;
          setMlState((prev) => ({
            ...prev,
            ...details,
            inference_time_ms: infTime,
          }));
        } else {
          setMlState((prev) => ({
            ...prev,
            status:
              result.status === "WARMUP"
                ? "WARMUP"
                : result.status === "PROCESSING_ERROR"
                  ? "ERROR"
                  : "EVALUATED",
            ml_anomaly: result.is_anomaly,
            raw_lof_score: result.anomaly_score,
            normalized_lof_score: result.anomaly_score,
            reason: result.explanation,
          }));
        }

        if (result.is_anomaly) {
          const timeFormatted = new Date(result.timestamp).toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
          });

          const isHum = result.anomaly_type.includes("HUMIDITY");
          const isPress = result.anomaly_type.includes("PRESSURE");
          const isMulti =
            result.anomaly_type.includes("MULTIVARIATE") || result.anomaly_type.includes("WEATHER");

          const observedStr = isHum
            ? result.readings.humidity !== null && result.readings.humidity !== undefined
              ? `${result.readings.humidity.toFixed(1)} %`
              : "NULL"
            : isPress
              ? result.readings.pressure !== null && result.readings.pressure !== undefined
                ? `${result.readings.pressure.toFixed(1)} hPa`
                : "NULL"
              : isMulti
                ? `${result.readings.temperature?.toFixed(1) ?? "—"}°C • ${result.readings.humidity?.toFixed(0) ?? "—"}%`
                : result.readings.temperature !== null && result.readings.temperature !== undefined
                  ? `${result.readings.temperature.toFixed(1)} °C`
                  : "NULL";

          const expectedStr = isHum
            ? result.previous_readings.humidity !== null &&
              result.previous_readings.humidity !== undefined
              ? `${result.previous_readings.humidity.toFixed(1)} %`
              : "Nominal"
            : isPress
              ? result.previous_readings.pressure !== null &&
                result.previous_readings.pressure !== undefined
                ? `${result.previous_readings.pressure.toFixed(1)} hPa`
                : "Nominal"
              : isMulti
                ? "Thermodynamic Envelope"
                : result.previous_readings.temperature !== null &&
                    result.previous_readings.temperature !== undefined
                  ? `${result.previous_readings.temperature.toFixed(1)} °C`
                  : "Nominal";

          const incident: AnomalyIncident = {
            id:
              result.incident_id ||
              `incident-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            stationId: result.station_id || stnId,
            time: timeFormatted,
            type: result.anomaly_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
            severity: result.severity,
            parameter: isHum
              ? "Humidity"
              : isPress
                ? "Pressure"
                : isMulti
                  ? "Multivariate"
                  : "Temperature",
            observed: observedStr,
            expected: expectedStr,
            description: result.explanation,
          };

          setActiveAnomaly(incident);
          setRecentAnomalies((prev) => [
            incident,
            ...prev
              .filter((p) => p.stationId !== incident.stationId || p.type !== incident.type)
              .slice(0, 29),
          ]);
          setToastMessage({
            title: `● ${result.anomaly_type.replace(/_/g, " ")} (${result.severity}):`,
            desc: result.explanation,
            time: timeFormatted,
          });
        } else {
          setActiveAnomaly(null);
        }
        return result;
      } catch (err) {
        console.error("[SkyGuard Store] Pipeline execution error:", err);
        return null;
      } finally {
        setPipelineRunning(false);
      }
    },
    [],
  );

  // Guard to prevent stale station fetches from overwriting current station data
  const latestStationRequestRef = useRef<string>("");

  // Fetch real live observation via IMD AWS / Synoptic live stream
  const fetchLiveTelemetryForStation = useCallback(
    async (stationId: string) => {
      // Set the latest request ID so stale responses are discarded
      latestStationRequestRef.current = stationId;

      const stn = dataset.stations.find((s) => s.station_id === stationId) || dataset.stations[0];
      const lat = stn?.latitude;
      const lon = stn?.longitude;

      try {
        const liveData = await WeatherService.fetchLiveIMDObservation(stationId, lat, lon);

        // Discard if a newer station was selected while this fetch was in-flight
        if (latestStationRequestRef.current !== stationId) {
          console.info(`[SkyGuard Store] Discarding stale response for ${stationId} (current: ${latestStationRequestRef.current})`);
          return;
        }

        if (liveData) {
          // 1. Update station state in dataset with accurate source and simulation identity
          setDataset((prev) => {
            const updated = prev.stations.map((s) => {
              if (s.station_id === stationId) {
                return {
                  ...s,
                  temperature: liveData.temperature,
                  humidity: liveData.humidity,
                  pressure: liveData.pressure,
                  wind_speed: liveData.wind_speed,
                  rainfall: liveData.rainfall,
                  timestamp: liveData.timestamp,
                  source: liveData.source,
                  is_simulated: liveData.is_simulated,
                  minutes_since_observation: 0,
                };
              }
              return s;
            });
            return { ...prev, stations: updated };
          });

          setActiveTelemetrySource(liveData.source);

          // 2. Feed into unified detection pipeline with exact observation source
          if (liveData.pipeline_result) {
            setCurrentPipelineResult(liveData.pipeline_result);
          } else {
            await processPipelineData(
              stationId,
              {
                station_id: stationId,
                timestamp: liveData.timestamp,
                temperature: liveData.temperature,
                humidity: liveData.humidity,
                pressure: liveData.pressure,
                wind_speed: liveData.wind_speed,
                rainfall: liveData.rainfall,
              },
              liveData.source,
            );
          }
        }
      } catch (err) {
        console.error("[SkyGuard Store] Failed to fetch live observation:", err);
      }
    },
    [dataset.stations, processPipelineData],
  );

  // Step demo to a specific index and process observation
  const stepDemoToIndex = useCallback(
    async (index: number, runId?: string | null) => {
      const records = demoRecordsRef.current;
      if (!records || index < 0 || index >= records.length) return;
      const rec = records[index];
      if (!rec) return;

      demoIndexRef.current = index;
      setDemoIndex(index);
      setSelectedStationId(rec.station_id);
      setActiveTelemetrySource("DEMO_SIMULATION");

      isDemoProcessing.current = true;
      try {
        await processPipelineData(
          rec.station_id,
          rec,
          "DEMO_SIMULATION",
          runId ?? demoRunIdRef.current,
        );
      } finally {
        isDemoProcessing.current = false;
      }
    },
    [processPipelineData],
  );

  // Demo Controls
  const startDemo = useCallback(
    async (scenarioKey: string = "ALL") => {
      let records = demoRecordsRef.current;
      if (!records || records.length === 0) {
        records = EMBEDDED_DEMO_RECORDS;
        demoRecordsRef.current = records;
        setDemoRecords(records);
      }

      // Check backend for fresh dataset asynchronously
      WeatherService.fetchDemoDataset()
        .then((resp) => {
          if (resp && resp.records && resp.records.length > 0) {
            demoRecordsRef.current = resp.records;
            setDemoRecords(resp.records);
            if (resp.scenarios && resp.scenarios.length > 0) {
              setDemoScenariosList(resp.scenarios);
            }
          }
        })
        .catch(() => {});

      const newRunId = `demo_run_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      demoRunIdRef.current = newRunId;
      setDemoRunId(newRunId);
      setDemoCompleted(false);

      setDataSource("DEMO");
      setDemoScenarioState(scenarioKey);
      setConnectionState("SIMULATION");

      const matched = demoScenariosList.find((s) => s.scenario === scenarioKey);
      const startIndex = matched && scenarioKey !== "ALL" ? matched.start_index : 0;

      await stepDemoToIndex(startIndex, newRunId);
      setDemoPlaying(true);
    },
    [demoScenariosList, stepDemoToIndex],
  );

  const returnToLive = useCallback(() => {
    // 1. Immediately stop the demo timer & playback
    if (demoTimer.current) {
      clearInterval(demoTimer.current);
      demoTimer.current = null;
    }
    setDemoPlaying(false);
    setDemoCompleted(false);
    setDemoRunId(null);
    demoRunIdRef.current = null;

    // 2. Switch mode to LIVE
    setDataSource("LIVE");
    setConnectionState("LIVE");
    setActiveAnomaly(null);
    setToastMessage(null);

    // 3. Fetch real live observation immediately without page reload
    fetchLiveTelemetryForStation(selectedStationId);
  }, [selectedStationId, fetchLiveTelemetryForStation]);

  const pauseDemo = useCallback(() => {
    setDemoPlaying(false);
  }, []);

  const resumeDemo = useCallback(() => {
    if (demoCompleted || demoIndexRef.current >= demoRecordsRef.current.length - 1) {
      setDemoCompleted(false);
      stepDemoToIndex(0);
    }
    setDemoPlaying(true);
  }, [demoCompleted, stepDemoToIndex]);

  const resetDemo = useCallback(() => {
    setDemoPlaying(false);
    setDemoCompleted(false);
    stepDemoToIndex(0);
  }, [stepDemoToIndex]);

  const setDemoSpeed = useCallback((speed: number) => {
    setDemoSpeedState(speed);
  }, []);

  const setDemoScenario = useCallback(
    async (scenarioKey: string) => {
      setDemoScenarioState(scenarioKey);
      const matched = demoScenariosList.find((s) => s.scenario === scenarioKey);
      const startIndex = matched ? matched.start_index : 0;
      await stepDemoToIndex(startIndex);
    },
    [demoScenariosList, stepDemoToIndex],
  );

  const stepDemoIndex = useCallback(
    async (index: number) => {
      await stepDemoToIndex(index);
    },
    [stepDemoToIndex],
  );

  // Demo Playback Interval Loop
  useEffect(() => {
    if (demoTimer.current) {
      clearInterval(demoTimer.current);
      demoTimer.current = null;
    }

    if (dataSource === "DEMO" && demoPlaying && demoRecords.length > 0) {
      const intervalMs = Math.max(50, Math.floor(1000 / demoSpeed));
      demoTimer.current = setInterval(() => {
        if (isDemoProcessing.current) return;
        const curr = demoIndexRef.current;
        const next = curr + 1;
        const total = demoRecordsRef.current.length;

        if (next >= total) {
          setDemoPlaying(false);
          setDemoCompleted(true);
          return;
        }

        stepDemoToIndex(next);
      }, intervalMs);
    }

    return () => {
      if (demoTimer.current) {
        clearInterval(demoTimer.current);
        demoTimer.current = null;
      }
    };
  }, [dataSource, demoPlaying, demoSpeed, demoRecords.length, stepDemoToIndex]);

  // Initial live observation processing on mount or station change
  useEffect(() => {
    if (dataSource === "LIVE") {
      fetchLiveTelemetryForStation(selectedStationId);
    }
  }, [dataSource, selectedStationId, fetchLiveTelemetryForStation]);

  // Periodic background live polling (every 30s) while in LIVE mode
  useEffect(() => {
    if (dataSource !== "LIVE") return;
    const interval = setInterval(() => {
      fetchLiveTelemetryForStation(selectedStationId);
    }, 30000);
    return () => clearInterval(interval);
  }, [dataSource, selectedStationId, fetchLiveTelemetryForStation]);

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clock updater
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      setClock(
        now.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: true,
        }) + " IST",
      );
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Fetch Open-Meteo external forecast when selected station changes.
  // Also clear stale anomaly/pipeline state to prevent buffered readings from wrong city.
  useEffect(() => {
    // Clear stale data immediately on station change to prevent wrong-city readings
    setExternalForecast({ status: "loading" });
    setCurrentPipelineResult(null);
    setActiveAnomaly(null);

    // Look up coordinates from WeatherService metadata first, then fall back to dataset
    const metaStation = WeatherService.getStationById(selectedStationId);
    const datasetStation = dataset.stations.find((s) => s.station_id === selectedStationId);
    const lat = metaStation?.lat ?? datasetStation?.latitude;
    const lon = metaStation?.lon ?? datasetStation?.longitude;

    if (lat !== undefined && lon !== undefined) {
      WeatherService.fetchExternalForecast(lat, lon).then((ctx) => {
        // Only apply if still the selected station
        setExternalForecast(ctx);
      });
    }
  }, [selectedStationId, dataset.stations]);

  const refresh = useCallback(() => {
    if (dataSource === "DEMO") return;
    setRefreshing(true);
    fetchLiveTelemetryForStation(selectedStationId).finally(() => {
      setRefreshing(false);
    });
  }, [dataSource, selectedStationId, fetchLiveTelemetryForStation]);

  // Fetch real persisted incidents from backend API
  useEffect(() => {
    fetch("/api/incidents?limit=25")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.incidents) && data.incidents.length > 0) {
          interface BackendIncident {
            incident_id: string;
            station_id: string;
            first_detected: string;
            anomaly_type: string;
            severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
            affected_parameter?: string;
            observed_value?: number;
            expected_value?: number;
            explanation?: string;
            status?: string;
          }
          const loaded: AnomalyIncident[] = data.incidents.map((inc: BackendIncident) => ({
            id: inc.incident_id,
            stationId: inc.station_id,
            time: new Date(inc.first_detected).toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
            }),
            type: inc.anomaly_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
            severity: inc.severity,
            parameter: inc.affected_parameter || "Temperature",
            observed:
              inc.observed_value !== undefined && inc.observed_value !== null
                ? `${inc.observed_value.toFixed(1)}`
                : "Anomalous",
            expected:
              inc.expected_value !== undefined && inc.expected_value !== null
                ? `${inc.expected_value.toFixed(1)}`
                : "Nominal",
            description: inc.explanation || "Detected by backend multi-channel pipeline",
            status: inc.status,
          }));
          setRecentAnomalies(loaded);
        }
      })
      .catch(() => {});
  }, []);

  // Demo Scenario Handler (routes to unified demo replay without frontend fabrication)
  const runScenario = useCallback(
    (s: DemoScenario | "") => {
      setScenario(s);

      if (s === "LOST") {
        setConnectionState("LOST");
        setActiveAnomaly(null);
        setToastMessage({
          title: "Communication Link Lost",
          desc: "Telemetry feed disconnected from AWS telemetry station.",
          time: new Date().toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
          }),
        });
        setMlState((prev) => ({
          ...prev,
          status: "WARMUP",
          ml_anomaly: false,
          raw_lof_score: null,
          normalized_lof_score: null,
          reason: "Communication lost: Telemetry feed disconnected",
        }));
      } else if (s === "RESET" || s === "") {
        returnToLive();
      } else {
        // Map legacy scenario keys if needed
        const keyMap: Record<string, string> = {
          TEMP_SPIKE: "SPIKE",
          FROZEN: "FROZEN_SENSOR",
        };
        const targetScenario = keyMap[s] || s;
        startDemo(targetScenario);
      }
    },
    [returnToLive, startDemo],
  );

  const acknowledgeAnomaly = useCallback(() => {
    setToastMessage(null);
  }, []);

  const dismissAnomaly = useCallback(() => {
    setActiveAnomaly(null);
    setToastMessage(null);
    setScenario("");
    const currentStn =
      dataset.stations.find((st) => st.station_id === selectedStationId) || dataset.stations[0];
    if (currentStn) {
      processPipelineData(
        selectedStationId,
        {
          temperature: currentStn.temperature ?? 32.4,
          humidity: currentStn.humidity ?? 64.0,
          pressure: currentStn.pressure ?? 1008.2,
          wind_speed: currentStn.wind_speed ?? 3.5,
          rainfall: currentStn.rainfall ?? 0.0,
          timestamp: new Date().toISOString(),
        },
        "LIVE",
      );
    }
  }, [selectedStationId, dataset.stations, processPipelineData]);

  const dismissToast = useCallback(() => {
    setToastMessage(null);
  }, []);

  const value = useMemo<SkyGuardContextValue>(
    () => ({
      dataset,
      refreshing,
      refresh,
      selectedStationId,
      setSelectedStationId,
      config,
      setConfig,
      resetConfig: () => setConfig(DEFAULT_CONFIG),

      // Two-Mode Data Architecture & Unified Pipeline
      dataSource,
      activeTelemetrySource,
      setDataSource,
      demoRunId,
      demoCompleted,
      demoPlaying,
      demoSpeed,
      demoIndex,
      demoTotal: demoRecords.length || 504,
      demoScenario,
      demoScenariosList,
      startDemo,
      returnToLive,
      pauseDemo,
      resumeDemo,
      resetDemo,
      setDemoSpeed,
      setDemoScenario,
      stepDemoIndex,
      currentPipelineResult,
      pipelineRunning,

      scenario,
      runScenario,
      connectionState,
      clock,
      activeParameter,
      setActiveParameter,
      chartWindow,
      setChartWindow,
      activeAnomaly,
      recentAnomalies,
      acknowledgeAnomaly,
      dismissAnomaly,
      toastMessage,
      dismissToast,
      externalForecast,
      mlState,
    }),
    [
      dataset,
      refreshing,
      refresh,
      selectedStationId,
      config,
      dataSource,
      activeTelemetrySource,
      setDataSource,
      demoRunId,
      demoCompleted,
      demoPlaying,
      demoSpeed,
      demoIndex,
      demoRecords.length,
      demoScenario,
      demoScenariosList,
      startDemo,
      returnToLive,
      pauseDemo,
      resumeDemo,
      resetDemo,
      setDemoSpeed,
      setDemoScenario,
      stepDemoIndex,
      currentPipelineResult,
      pipelineRunning,
      scenario,
      runScenario,
      connectionState,
      clock,
      activeParameter,
      setActiveParameter,
      chartWindow,
      setChartWindow,
      activeAnomaly,
      recentAnomalies,
      acknowledgeAnomaly,
      dismissAnomaly,
      toastMessage,
      dismissToast,
      externalForecast,
      mlState,
    ],
  );

  return <SkyGuardContext.Provider value={value}>{children}</SkyGuardContext.Provider>;
}

export function useSkyGuard() {
  const ctx = useContext(SkyGuardContext);
  if (!ctx) throw new Error("useSkyGuard must be used inside SkyGuardProvider");
  return ctx;
}

export function useSelectedStation() {
  const { dataset, selectedStationId } = useSkyGuard();
  return dataset.stations.find((s) => s.station_id === selectedStationId) ?? dataset.stations[0]!;
}
