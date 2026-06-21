// Mock data grounded in the real Bengaluru dataset described in the brief.
// Counts, durations and rates match the spec exactly.

export const DATA_NOTE =
  "Predictive model trained on 8,173 historical Bengaluru traffic events (Nov 2023 – Apr 2024). Resource recommendations are AI-generated; actuals are supplemented with calibrated synthetic data.";

export type Priority = "High" | "Low";
export type EventStatus = "active" | "resolved" | "closed";

export interface CityEvent {
  id: string;
  cause: string;
  type: "planned" | "unplanned";
  corridor: string;
  junction: string;
  zone: string;
  priority: Priority;
  closure: boolean;
  status: EventStatus;
  startedMinAgo: number;
  durationEstHrs: number;
  confidence: number; // 0-100
  recOfficers: number;
  recBarricades: number;
  // For SVG map positioning (relative %, our stylized Bengaluru canvas)
  lat: number;
  lng: number;
}

export const CORRIDORS = [
  { name: "Mysore Road", events: 743 },
  { name: "Bellary Road 1", events: 610 },
  { name: "Tumkur Road", events: 458 },
  { name: "Bellary Road 2", events: 379 },
  { name: "Hosur Road", events: 298 },
  { name: "ORR North 1", events: 275 },
  { name: "Old Madras Road", events: 263 },
  { name: "Magadi Road", events: 245 },
  { name: "ORR East 1", events: 244 },
  { name: "Non-corridor", events: 180 },
];

export const JUNCTIONS = [
  { name: "MekhriCircle", events: 64, lat: 13.0084, lng: 77.5906 },
  { name: "AyyappaTempleJunc", events: 49, lat: 12.9785, lng: 77.5332 },
  { name: "SatteliteBusStandJunc", events: 43, lat: 12.9556, lng: 77.5385 },
  { name: "YeshwanthpuraCircle", events: 38, lat: 13.0227, lng: 77.5704 },
  { name: "YelhankaCircle", events: 34, lat: 13.1007, lng: 77.5963 },
  { name: "SilkBoardJunc", events: 33, lat: 12.9176, lng: 77.6244 },
  { name: "JalahalliCross", events: 32, lat: 13.0371, lng: 77.5255 },
  { name: "Nagavara-ORR", events: 32, lat: 13.0416, lng: 77.6248 },
  { name: "K R Circle", events: 31, lat: 12.9716, lng: 77.5946 },
];

export const ZONES = [
  "Central Zone 1",
  "Central Zone 2",
  "West Zone 1",
  "West Zone 2",
  "North Zone 1",
  "North Zone 2",
  "South Zone 1",
  "South Zone 2",
  "East Zone 1",
  "East Zone 2",
];

export const EVENT_CAUSES = [
  { value: "vehicle_breakdown", label: "Vehicle Breakdown", count: 4896, avgHrs: 0.8, closurePct: 4 },
  { value: "construction", label: "Construction", count: 649, avgHrs: 13.3, closurePct: 27 },
  { value: "water_logging", label: "Water Logging", count: 458, avgHrs: 14.1, closurePct: 35 },
  { value: "accident", label: "Accident", count: 365, avgHrs: 0.8, closurePct: 12 },
  { value: "tree_fall", label: "Tree Fall", count: 284, avgHrs: 10.6, closurePct: 22 },
  { value: "road_conditions", label: "Road Conditions", count: 170, avgHrs: 10.9, closurePct: 18 },
  { value: "congestion", label: "Congestion Cluster", count: 136, avgHrs: 1.2, closurePct: 6 },
  { value: "public_event", label: "Public Event", count: 84, avgHrs: 2.1, closurePct: 46 },
  { value: "procession", label: "Procession", count: 72, avgHrs: 0.9, closurePct: 38 },
  { value: "vip_movement", label: "VIP Movement", count: 20, avgHrs: 1.2, closurePct: 80 },
  { value: "protest", label: "Protest", count: 15, avgHrs: 3.4, closurePct: 60 },
  { value: "pot_holes", label: "Pot Holes", count: 12, avgHrs: 18.7, closurePct: 8 },
  { value: "others", label: "Others", count: 30, avgHrs: 9.2, closurePct: 10 },
];

// Sample active events (synthetic but consistent with real distribution)
export const ACTIVE_EVENTS: CityEvent[] = [
  {
    id: "FKID000412",
    cause: "vip_movement",
    type: "planned",
    corridor: "Mysore Road",
    junction: "K R Circle",
    zone: "Central Zone 1",
    priority: "High",
    closure: true,
    status: "active",
    startedMinAgo: 18,
    durationEstHrs: 1.2,
    confidence: 88,
    recOfficers: 10,
    recBarricades: 4,
    lat: 12.9716, lng: 77.5946,
  },
  {
    id: "FKID000418",
    cause: "water_logging",
    type: "unplanned",
    corridor: "Hosur Road",
    junction: "SilkBoardJunc",
    zone: "South Zone 1",
    priority: "High",
    closure: true,
    status: "active",
    startedMinAgo: 142,
    durationEstHrs: 14.1,
    confidence: 76,
    recOfficers: 6,
    recBarricades: 8,
    lat: 12.9176, lng: 77.6244,
  },
  {
    id: "FKID000419",
    cause: "vehicle_breakdown",
    type: "unplanned",
    corridor: "Bellary Road 1",
    junction: "MekhriCircle",
    zone: "North Zone 1",
    priority: "Low",
    closure: false,
    status: "active",
    startedMinAgo: 34,
    durationEstHrs: 0.8,
    confidence: 92,
    recOfficers: 3,
    recBarricades: 1,
    lat: 13.0084, lng: 77.5906,
  },
  {
    id: "FKID000421",
    cause: "construction",
    type: "planned",
    corridor: "Tumkur Road",
    junction: "JalahalliCross",
    zone: "West Zone 1",
    priority: "High",
    closure: true,
    status: "active",
    startedMinAgo: 360,
    durationEstHrs: 13.3,
    confidence: 81,
    recOfficers: 5,
    recBarricades: 6,
    lat: 13.0371, lng: 77.5255,
  },
  {
    id: "FKID000425",
    cause: "accident",
    type: "unplanned",
    corridor: "ORR North 1",
    junction: "Nagavara-ORR",
    zone: "North Zone 2",
    priority: "High",
    closure: false,
    status: "active",
    startedMinAgo: 12,
    durationEstHrs: 0.9,
    confidence: 84,
    recOfficers: 4,
    recBarricades: 2,
    lat: 13.0416, lng: 77.6248,
  },
  {
    id: "FKID000427",
    cause: "tree_fall",
    type: "unplanned",
    corridor: "Magadi Road",
    junction: "AyyappaTempleJunc",
    zone: "West Zone 2",
    priority: "Low",
    closure: false,
    status: "active",
    startedMinAgo: 95,
    durationEstHrs: 10.6,
    confidence: 71,
    recOfficers: 2,
    recBarricades: 2,
    lat: 12.9785, lng: 77.5332,
  },
  {
    id: "FKID000430",
    cause: "public_event",
    type: "planned",
    corridor: "Old Madras Road",
    junction: "YeshwanthpuraCircle",
    zone: "East Zone 1",
    priority: "High",
    closure: true,
    status: "active",
    startedMinAgo: 50,
    durationEstHrs: 2.1,
    confidence: 86,
    recOfficers: 15,
    recBarricades: 8,
    lat: 13.0227, lng: 77.5704,
  },
  {
    id: "FKID000431",
    cause: "congestion",
    type: "unplanned",
    corridor: "Bellary Road 2",
    junction: "YelhankaCircle",
    zone: "North Zone 1",
    priority: "Low",
    closure: false,
    status: "active",
    startedMinAgo: 6,
    durationEstHrs: 1.2,
    confidence: 79,
    recOfficers: 2,
    recBarricades: 0,
    lat: 13.1007, lng: 77.5963,
  },
];

export function causeMeta(value: string) {
  return EVENT_CAUSES.find((c) => c.value === value) ?? EVENT_CAUSES[0];
}

export const KPI = {
  readiness: 72,
  activeEvents: 1007,
  predictedHighRisk: 47,
  activeRoadClosures: 38,
  avgResolutionHrs: 3.66,
  totalHistorical: 8173,
};
