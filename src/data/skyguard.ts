/**
 * SkyGuard AI — centralized DEMO data provider.
 *
 * Every page reads from this module. When authorized IMD AWS API access is
 * granted, only the generator functions below need replacing — the exported
 * shapes (Station, Observation, AnomalyEvent) and the accessor functions
 * getStations / getObservations / getAnomalies stay identical.
 */

export type StationStatus = "NORMAL" | "WARNING" | "ANOMALY" | "CRITICAL" | "OFFLINE";
export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type SensorHealth = "HEALTHY" | "WARNING" | "CRITICAL" | "OFFLINE";

export type AnomalyType =
  | "Temperature Spike"
  | "Pressure Anomaly"
  | "Humidity Anomaly"
  | "Frozen Sensor"
  | "Sensor Drift"
  | "Missing Data"
  | "Multivariate Inconsistency"
  | "Spatial Inconsistency";

export const ANOMALY_TYPES: AnomalyType[] = [
  "Temperature Spike",
  "Pressure Anomaly",
  "Humidity Anomaly",
  "Frozen Sensor",
  "Sensor Drift",
  "Missing Data",
  "Multivariate Inconsistency",
  "Spatial Inconsistency",
];

export interface Station {
  station_id: string;
  station_name: string;
  state: string;
  district: string;
  latitude: number;
  longitude: number;
  timestamp: string;
  temperature: number | null;
  pressure: number | null;
  humidity: number | null;
  wind_speed?: number | null;
  rainfall?: number | null;
  status: StationStatus;
  anomaly_type: AnomalyType | null;
  severity: Severity | null;
  reason: string | null;
  anomaly_count: number;
  health: SensorHealth;
  sensors: {
    temperature: SensorHealth;
    pressure: SensorHealth;
    humidity: SensorHealth;
  };
  maintenance_priority: "P1" | "P2" | "P3" | "P4";
  minutes_since_observation: number;
  is_simulated: boolean;
  station_type?: "IMD_AWS_REFERENCE" | "SIMULATED_INDICATIVE";
  source?: "IMD_AWS" | "DEMO_SIMULATION" | "OPEN_METEO_CONTEXT" | "OFFLINE_PREVIEW";
}

export interface Observation {
  station_id: string;
  hour: string;
  temperature: number | null;
  pressure: number | null;
  humidity: number | null;
  expected_temperature: number;
  expected_pressure: number;
  expected_humidity: number;
  imputed: boolean;
  anomaly: boolean;
}

export interface AnomalyEvent {
  id: string;
  station_id: string;
  station_name: string;
  state: string;
  anomaly_type: AnomalyType;
  severity: Severity;
  observed: string;
  expected: string;
  time: string;
  reason: string;
  parameter: "Temperature" | "Pressure" | "Humidity" | "Multivariate";
}

/* ------------------------------------------------------------------ */
/* seeded pseudo-randomness so refreshes stay realistic and repeatable */
/* ------------------------------------------------------------------ */

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const round = (v: number, d = 1) => Math.round(v * 10 ** d) / 10 ** d;

interface StationSeed {
  id: string;
  name: string;
  state: string;
  district: string;
  lat: number;
  lon: number;
  baseTemp: number;
  basePressure: number;
  baseHumidity: number;
  is_simulated?: boolean;
  station_type?: "IMD_AWS_REFERENCE" | "SIMULATED_INDICATIVE";
}

/** DEMO station inventory — indicative locations only, not the actual IMD AWS network. */
const STATION_SEEDS: StationSeed[] = [
  {
    id: "43189",
    name: "Vijayawada (AWS014)",
    state: "Andhra Pradesh",
    district: "NTR",
    lat: 16.5062,
    lon: 80.648,
    baseTemp: 32.4,
    basePressure: 1008.2,
    baseHumidity: 64,
    is_simulated: false,
    station_type: "IMD_AWS_REFERENCE",
  },
  {
    id: "43150",
    name: "Visakhapatnam (AWS008)",
    state: "Andhra Pradesh",
    district: "Visakhapatnam",
    lat: 17.6868,
    lon: 83.2185,
    baseTemp: 29.8,
    basePressure: 1012.4,
    baseHumidity: 78,
    is_simulated: false,
    station_type: "IMD_AWS_REFERENCE",
  },
  {
    id: "43245",
    name: "Tirupati (AWS021)",
    state: "Andhra Pradesh",
    district: "Tirupati",
    lat: 13.6288,
    lon: 79.4192,
    baseTemp: 34.1,
    basePressure: 1004.8,
    baseHumidity: 52,
    is_simulated: false,
    station_type: "IMD_AWS_REFERENCE",
  },
  {
    id: "AWS-101",
    name: "New Delhi",
    state: "Delhi",
    district: "New Delhi",
    lat: 28.6139,
    lon: 77.209,
    baseTemp: 33.4,
    basePressure: 1004.2,
    baseHumidity: 48,
  },
  {
    id: "AWS-102",
    name: "Jaipur",
    state: "Rajasthan",
    district: "Jaipur",
    lat: 26.9124,
    lon: 75.7873,
    baseTemp: 35.1,
    basePressure: 1002.8,
    baseHumidity: 39,
  },
  {
    id: "AWS-103",
    name: "Lucknow",
    state: "Uttar Pradesh",
    district: "Lucknow",
    lat: 26.8467,
    lon: 80.9462,
    baseTemp: 32.7,
    basePressure: 1005.1,
    baseHumidity: 57,
  },
  {
    id: "AWS-104",
    name: "Hyderabad",
    state: "Telangana",
    district: "Rangareddy",
    lat: 17.385,
    lon: 78.4867,
    baseTemp: 32.4,
    basePressure: 1007.4,
    baseHumidity: 62,
  },
  {
    id: "AWS-105",
    name: "Ahmedabad",
    state: "Gujarat",
    district: "Ahmedabad",
    lat: 23.0225,
    lon: 72.5714,
    baseTemp: 34.6,
    basePressure: 1003.6,
    baseHumidity: 45,
  },
  {
    id: "AWS-106",
    name: "Mumbai",
    state: "Maharashtra",
    district: "Mumbai Suburban",
    lat: 19.076,
    lon: 72.8777,
    baseTemp: 30.2,
    basePressure: 1008.9,
    baseHumidity: 76,
  },
  {
    id: "AWS-107",
    name: "Bhopal",
    state: "Madhya Pradesh",
    district: "Bhopal",
    lat: 23.2599,
    lon: 77.4126,
    baseTemp: 31.8,
    basePressure: 1006.2,
    baseHumidity: 53,
  },
  {
    id: "AWS-108",
    name: "Nagpur",
    state: "Maharashtra",
    district: "Nagpur",
    lat: 21.1458,
    lon: 79.0882,
    baseTemp: 33.9,
    basePressure: 1005.4,
    baseHumidity: 47,
  },
  {
    id: "AWS-109",
    name: "Vijayawada",
    state: "Andhra Pradesh",
    district: "NTR",
    lat: 16.5062,
    lon: 80.648,
    baseTemp: 33.1,
    basePressure: 1007.9,
    baseHumidity: 68,
  },
  {
    id: "AWS-110",
    name: "Bengaluru",
    state: "Karnataka",
    district: "Bengaluru Urban",
    lat: 12.9716,
    lon: 77.5946,
    baseTemp: 27.6,
    basePressure: 1009.3,
    baseHumidity: 64,
  },
  {
    id: "AWS-111",
    name: "Chennai",
    state: "Tamil Nadu",
    district: "Chennai",
    lat: 13.0827,
    lon: 80.2707,
    baseTemp: 32.2,
    basePressure: 1008.6,
    baseHumidity: 74,
  },
  {
    id: "AWS-112",
    name: "Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    lat: 22.5726,
    lon: 88.3639,
    baseTemp: 31.5,
    basePressure: 1007.1,
    baseHumidity: 78,
  },
  {
    id: "AWS-113",
    name: "Bhubaneswar",
    state: "Odisha",
    district: "Khordha",
    lat: 20.2961,
    lon: 85.8245,
    baseTemp: 32.0,
    basePressure: 1007.6,
    baseHumidity: 71,
  },
  {
    id: "AWS-114",
    name: "Patna",
    state: "Bihar",
    district: "Patna",
    lat: 25.5941,
    lon: 85.1376,
    baseTemp: 32.9,
    basePressure: 1005.8,
    baseHumidity: 63,
  },
  {
    id: "AWS-115",
    name: "Ranchi",
    state: "Jharkhand",
    district: "Ranchi",
    lat: 23.3441,
    lon: 85.3096,
    baseTemp: 29.4,
    basePressure: 1006.9,
    baseHumidity: 60,
  },
  {
    id: "AWS-116",
    name: "Guwahati",
    state: "Assam",
    district: "Kamrup Metropolitan",
    lat: 26.1445,
    lon: 91.7362,
    baseTemp: 30.1,
    basePressure: 1006.4,
    baseHumidity: 80,
  },
  {
    id: "AWS-117",
    name: "Dehradun",
    state: "Uttarakhand",
    district: "Dehradun",
    lat: 30.3165,
    lon: 78.0322,
    baseTemp: 26.8,
    basePressure: 999.4,
    baseHumidity: 66,
  },
  {
    id: "AWS-118",
    name: "Srinagar",
    state: "Jammu & Kashmir",
    district: "Srinagar",
    lat: 34.0837,
    lon: 74.7973,
    baseTemp: 21.3,
    basePressure: 995.2,
    baseHumidity: 55,
  },
  {
    id: "AWS-119",
    name: "Pune",
    state: "Maharashtra",
    district: "Pune",
    lat: 18.5204,
    lon: 73.8567,
    baseTemp: 29.8,
    basePressure: 1008.1,
    baseHumidity: 58,
  },
  {
    id: "AWS-120",
    name: "Thiruvananthapuram",
    state: "Kerala",
    district: "Thiruvananthapuram",
    lat: 8.5241,
    lon: 76.9366,
    baseTemp: 30.6,
    basePressure: 1009.8,
    baseHumidity: 82,
  },
  /* nearby satellite stations used by the spatial-consistency demo around AWS-104 */
  {
    id: "AWS-121",
    name: "Medchal",
    state: "Telangana",
    district: "Medchal-Malkajgiri",
    lat: 17.6288,
    lon: 78.4813,
    baseTemp: 32.1,
    basePressure: 1007.2,
    baseHumidity: 61,
  },
  {
    id: "AWS-122",
    name: "Shamshabad",
    state: "Telangana",
    district: "Rangareddy",
    lat: 17.2403,
    lon: 78.4294,
    baseTemp: 32.5,
    basePressure: 1007.6,
    baseHumidity: 60,
  },
  {
    id: "AWS-123",
    name: "Sangareddy",
    state: "Telangana",
    district: "Sangareddy",
    lat: 17.6248,
    lon: 78.0865,
    baseTemp: 31.9,
    basePressure: 1007.1,
    baseHumidity: 59,
  },
];

/** Scripted DEMO conditions so the judging walkthrough is deterministic. */
interface ScriptedCase {
  status: StationStatus;
  anomaly_type: AnomalyType;
  severity: Severity;
  reason: string;
  anomaly_count: number;
  sensor: "temperature" | "pressure" | "humidity";
}

const SCRIPTED: Record<string, ScriptedCase> = {
  "AWS-104": {
    status: "ANOMALY",
    anomaly_type: "Temperature Spike",
    severity: "HIGH",
    reason:
      "Rapid deviation from learned temporal behaviour while nearby stations remained normal.",
    anomaly_count: 3,
    sensor: "temperature",
  },
  "AWS-102": {
    status: "WARNING",
    anomaly_type: "Sensor Drift",
    severity: "MEDIUM",
    reason: "Slow upward offset in temperature relative to the learned regional baseline.",
    anomaly_count: 2,
    sensor: "temperature",
  },
  "AWS-112": {
    status: "ANOMALY",
    anomaly_type: "Humidity Anomaly",
    severity: "MEDIUM",
    reason: "Relative humidity exceeded the expected band without a corresponding pressure change.",
    anomaly_count: 2,
    sensor: "humidity",
  },
  "AWS-118": {
    status: "OFFLINE",
    anomaly_type: "Missing Data",
    severity: "MEDIUM",
    reason: "No observation received within the expected reporting interval.",
    anomaly_count: 1,
    sensor: "temperature",
  },
  "AWS-108": {
    status: "WARNING",
    anomaly_type: "Pressure Anomaly",
    severity: "LOW",
    reason: "Pressure tendency deviates mildly from the learned diurnal pattern.",
    anomaly_count: 1,
    sensor: "pressure",
  },
  "AWS-113": {
    status: "CRITICAL",
    anomaly_type: "Frozen Sensor",
    severity: "CRITICAL",
    reason: "Identical humidity value repeated across consecutive reporting intervals.",
    anomaly_count: 4,
    sensor: "humidity",
  },
  "AWS-115": {
    status: "WARNING",
    anomaly_type: "Multivariate Inconsistency",
    severity: "MEDIUM",
    reason: "Temperature and humidity moved in a combination rarely observed at this station.",
    anomaly_count: 2,
    sensor: "humidity",
  },
};

export const HERO_STATION_ID = "43189";
export const HERO_ANOMALY_HOUR = "14:00";

export function isImdStation(stationId: string): boolean {
  return ["43189", "43150", "43245"].includes(stationId);
}

/* ------------------------------------------------------------------ */
/* dataset construction                                                */
/* ------------------------------------------------------------------ */

export interface Dataset {
  generatedAt: Date;
  stations: Station[];
  observations: Record<string, Observation[]>;
  anomalies: AnomalyEvent[];
}

const HOURS = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, "0")}:00`);

function diurnal(hour: number) {
  // peak near 14:00, nocturnal trough near 05:00
  return Math.sin(((hour - 8.5) / 24) * 2 * Math.PI);
}

function healthFor(count: number, status: StationStatus): SensorHealth {
  if (status === "OFFLINE") return "OFFLINE";
  if (status === "CRITICAL" || count >= 4) return "CRITICAL";
  if (status === "ANOMALY" || status === "WARNING" || count >= 2) return "WARNING";
  return "HEALTHY";
}

function priorityFor(health: SensorHealth, count: number): Station["maintenance_priority"] {
  if (health === "CRITICAL") return "P1";
  if (health === "OFFLINE") return "P2";
  if (health === "WARNING") return count >= 3 ? "P2" : "P3";
  return "P4";
}

function buildStation(seed: StationSeed, rand: () => number, now: Date): Station {
  const scripted = SCRIPTED[seed.id];
  const hour = now.getHours() + now.getMinutes() / 60;
  // Diurnal variation: peak around 14:30, trough in early morning
  const drift = diurnal(hour) * 4.2;

  let temperature: number | null = round(seed.baseTemp + drift + (rand() - 0.5) * 0.8);
  let pressure: number | null = round(seed.basePressure - drift * 0.4 + (rand() - 0.5) * 1.0, 1);
  let humidity: number | null = Math.min(
    99,
    Math.max(25, Math.round(seed.baseHumidity - drift * 2.0 + (rand() - 0.5) * 3)),
  );

  const status: StationStatus = scripted?.status ?? "NORMAL";
  let minutes = Math.round(2 + rand() * 12);

  if (scripted?.anomaly_type === "Humidity Anomaly") humidity = Math.min(99, (humidity ?? 60) + 22);
  if (scripted?.anomaly_type === "Frozen Sensor") humidity = 71;
  if (scripted?.anomaly_type === "Pressure Anomaly") pressure = round((pressure ?? 1005) - 6.4, 1);
  if (scripted?.anomaly_type === "Sensor Drift") temperature = round((temperature ?? 30) + 2.7);
  if (status === "OFFLINE") {
    temperature = null;
    pressure = null;
    humidity = null;
    minutes = 187;
  }

  const anomaly_count = scripted?.anomaly_count ?? 0;
  const health = healthFor(anomaly_count, status);
  const sensors = {
    temperature: "HEALTHY" as SensorHealth,
    pressure: "HEALTHY" as SensorHealth,
    humidity: "HEALTHY" as SensorHealth,
  };
  if (scripted) sensors[scripted.sensor] = health;
  if (status === "OFFLINE") {
    sensors.temperature = "OFFLINE";
    sensors.pressure = "OFFLINE";
    sensors.humidity = "OFFLINE";
  }

  const ts = new Date(now.getTime() - minutes * 60000);

  return {
    station_id: seed.id,
    station_name: seed.name,
    state: seed.state,
    district: seed.district,
    latitude: seed.lat,
    longitude: seed.lon,
    timestamp: ts.toISOString(),
    temperature,
    pressure,
    humidity,
    status,
    anomaly_type: scripted?.anomaly_type ?? null,
    severity: scripted?.severity ?? null,
    reason: scripted?.reason ?? null,
    anomaly_count,
    health,
    sensors,
    maintenance_priority: priorityFor(health, anomaly_count),
    minutes_since_observation: minutes,
    is_simulated: seed.is_simulated ?? !["43189", "43150", "43245"].includes(seed.id),
    station_type: ["43189", "43150", "43245"].includes(seed.id)
      ? "IMD_AWS_REFERENCE"
      : "SIMULATED_INDICATIVE",
    source: "DEMO_SIMULATION",
  };
}

function buildObservations(station: Station, seed: StationSeed, rand: () => number): Observation[] {
  return HOURS.map((hour, i) => {
    const wave = diurnal(i) * 3.2;
    const expected_temperature = round(seed.baseTemp + wave);
    const expected_pressure = round(seed.basePressure - diurnal(i) * 0.9, 1);
    const expected_humidity = Math.round(seed.baseHumidity - wave * 1.4);

    let temperature: number | null = round(expected_temperature + (rand() - 0.5) * 0.9);
    let pressure: number | null = round(expected_pressure + (rand() - 0.5) * 0.8, 1);
    let humidity: number | null = Math.round(expected_humidity + (rand() - 0.5) * 4);
    const anomaly = false;
    let imputed = false;

    if (station.status === "OFFLINE" && i > 19) {
      temperature = null;
      pressure = null;
      humidity = null;
      imputed = true;
    }

    if (station.anomaly_type === "Frozen Sensor" && i > 15) humidity = 71;

    return {
      station_id: station.station_id,
      hour,
      temperature,
      pressure,
      humidity,
      expected_temperature,
      expected_pressure,
      expected_humidity,
      imputed,
      anomaly,
    };
  });
}

function buildAnomalies(stations: Station[], now: Date): AnomalyEvent[] {
  const events: AnomalyEvent[] = [];
  const fmt = (d: Date) =>
    `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;

  stations
    .filter((s) => s.anomaly_type)
    .forEach((s, idx) => {
      const t = new Date(now.getTime() - (idx * 17 + 4) * 60000);
      let observed = "—";
      let expected = "—";
      let parameter: AnomalyEvent["parameter"] = "Multivariate";

      if (s.anomaly_type === "Temperature Spike" || s.anomaly_type === "Sensor Drift") {
        parameter = "Temperature";
        observed = s.temperature !== null ? `${s.temperature.toFixed(1)} °C` : "No data";
        expected = "31.5 – 34.2 °C";
      } else if (s.anomaly_type === "Pressure Anomaly") {
        parameter = "Pressure";
        observed = s.pressure !== null ? `${s.pressure.toFixed(1)} hPa` : "No data";
        expected = "1004.0 – 1009.0 hPa";
      } else if (s.anomaly_type === "Humidity Anomaly" || s.anomaly_type === "Frozen Sensor") {
        parameter = "Humidity";
        observed = s.humidity !== null ? `${s.humidity} %` : "No data";
        expected = "55 – 78 %";
      } else if (s.anomaly_type === "Missing Data") {
        observed = "No observation";
        expected = "1 record / 10 min";
      }

      events.push({
        id: `EVT-${s.station_id}-${idx}`,
        station_id: s.station_id,
        station_name: s.station_name,
        state: s.state,
        anomaly_type: s.anomaly_type!,
        severity: s.severity ?? "LOW",
        observed,
        expected,
        time: fmt(t),
        reason: s.reason ?? "Detected pattern deviation.",
        parameter,
      });
    });

  const order: Record<Severity, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  return events.sort((a, b) => order[a.severity] - order[b.severity]);
}

export function buildDataset(seedValue = 1, now = new Date()): Dataset {
  const rand = mulberry32(seedValue * 7919 + 13);
  const stations = STATION_SEEDS.map((s) => buildStation(s, rand, now));
  const observations: Record<string, Observation[]> = {};
  stations.forEach((station, i) => {
    observations[station.station_id] = buildObservations(station, STATION_SEEDS[i]!, rand);
  });
  return {
    generatedAt: now,
    stations,
    observations,
    anomalies: buildAnomalies(stations, now),
  };
}

/* ------------------------------------------------------------------ */
/* provider API — swap these for the IMD AWS client once authorized    */
/* ------------------------------------------------------------------ */

export function getStations(dataset: Dataset): Station[] {
  return dataset.stations;
}

export function getObservations(dataset: Dataset, stationId: string): Observation[] {
  return dataset.observations[stationId] ?? [];
}

export function getAnomalies(dataset: Dataset): AnomalyEvent[] {
  return dataset.anomalies;
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

export function haversineKm(a: Station, b: Station) {
  const R = 6371;
  const dLat = ((b.latitude - a.latitude) * Math.PI) / 180;
  const dLon = ((b.longitude - a.longitude) * Math.PI) / 180;
  const la1 = (a.latitude * Math.PI) / 180;
  const la2 = (b.latitude * Math.PI) / 180;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export function nearbyStations(dataset: Dataset, stationId: string, radiusKm: number) {
  const origin = dataset.stations.find((s) => s.station_id === stationId);
  if (!origin) return [];
  return dataset.stations
    .filter((s) => s.station_id !== stationId)
    .map((s) => ({ station: s, distance: haversineKm(origin, s) }))
    .filter((x) => x.distance <= radiusKm)
    .sort((a, b) => a.distance - b.distance);
}

export const STATUS_LABEL: Record<StationStatus, string> = {
  NORMAL: "Normal",
  WARNING: "Warning",
  ANOMALY: "Anomaly",
  CRITICAL: "Critical",
  OFFLINE: "Offline",
};

export function statusTone(status: StationStatus | SensorHealth) {
  switch (status) {
    case "NORMAL":
    case "HEALTHY":
      return "normal";
    case "WARNING":
      return "warning";
    case "ANOMALY":
    case "CRITICAL":
      return "critical";
    default:
      return "offline";
  }
}

export function severityTone(severity: Severity) {
  if (severity === "CRITICAL" || severity === "HIGH") return "critical";
  if (severity === "MEDIUM") return "warning";
  return "info";
}

export function formatTime(iso: string) {
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function formatDateTime(iso: string) {
  const d = new Date(iso);
  return `${d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" })} ${formatTime(iso)} IST`;
}
