// Display helpers, ported from app/pages/3_🔍_Who_Is_Lying.py (keep them in sync).

export const fmtInt = (x: number) => Math.round(x).toLocaleString('en-US');

export const fmtPct = (x: number, signed = false) =>
  `${signed && x > 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;

/** plain-language labels for every property in the released gold standard */
export const PROP_LABEL: Record<string, string> = {
  capacity_retention_at_temp_pct: 'capacity retained at temperature',
  charge_temp_range_c: 'charging temperature range',
  charge_voltage_v: 'charge voltage',
  cycle_life_cycles: 'cycle life',
  cycle_life_retention_pct: 'end-of-life capacity threshold',
  discharge_cutoff_v: 'discharge cut-off voltage',
  discharge_temp_range_c: 'discharge temperature range',
  fast_charge_current_a: 'fast-charge current',
  fast_discharge_current_a: 'fast discharge current',
  gravimetric_energy_density_wh_kg: 'energy density by weight',
  internal_impedance_mohm: 'internal impedance',
  mass_g: 'weight',
  max_charge_current_a: 'max charge current',
  max_charge_voltage_v: 'max charge voltage',
  max_cont_discharge_a: 'max continuous discharge current',
  max_pulse_discharge_a: 'max pulse discharge current',
  minimum_capacity_ah: 'minimum capacity',
  nominal_capacity_ah: 'nominal capacity',
  nominal_voltage_v: 'nominal voltage',
  operating_temp_range_c: 'operating temperature range',
  rated_capacity_ah: 'rated capacity',
  std_charge_current_a: 'standard charge current',
  std_charge_time_h: 'standard charge time',
  std_discharge_current_a: 'standard discharge current',
  storage_capacity_recovery_pct: 'capacity recovered after storage',
  storage_capacity_remaining_pct: 'capacity remaining after storage',
  storage_temp_range_c: 'storage temperature range',
  volumetric_energy_density_wh_l: 'energy density by volume',
};

// unit from the property-key suffix (order matters: most specific first)
const UNIT_SUFFIX: [string, string][] = [
  ['_cycles', 'cycles'], ['_pct', '%'], ['_mohm', 'mΩ'], ['_wh_kg', 'Wh/kg'], ['_wh_l', 'Wh/L'],
  ['_ah', 'Ah'], ['_c', '°C'], ['_v', 'V'], ['_a', 'A'], ['_h', 'h'], ['_g', 'g'],
];

export function propUnit(prop: string): string {
  return UNIT_SUFFIX.find(([s]) => prop.endsWith(s))?.[1] ?? '';
}

export function propLabel(prop: string): string {
  if (PROP_LABEL[prop]) return PROP_LABEL[prop];
  const suffix = UNIT_SUFFIX.find(([s]) => prop.endsWith(s))?.[0];
  return (suffix ? prop.slice(0, -suffix.length) : prop).replace(/_/g, ' ');
}

const num = (x: unknown) => String(Number(x)); // 1000.0 -> 1000, 47.5 -> 47.5

/** Parse a stored value: '[10.0, 45.0]' -> [10, 45], '1000.0' -> 1000, else the raw string. */
function parseValue(raw: unknown): number | [number, number] | string {
  const s = String(raw).trim();
  const range = s.match(/^\[\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\]$/);
  if (range) return [Number(range[1]), Number(range[2])];
  if (s !== '' && !Number.isNaN(Number(s))) return Number(s);
  return s;
}

/** '[10.0, 45.0]' -> '10–45 °C'; '1000.0' -> '1000 cycles' */
export function fmtValue(prop: string, raw: unknown): string {
  const unit = propUnit(prop);
  const v = parseValue(raw);
  let core: string;
  if (Array.isArray(v)) {
    const [lo, hi] = v;
    core = `${num(lo)}${lo < 0 ? ' to ' : '–'}${num(hi)}`; // '−5 to 50', '10–45'
  } else {
    core = typeof v === 'number' ? num(v) : v;
  }
  return unit ? `${core} ${unit}`.trim() : core;
}

// differing-condition rendering for same-value mismatches (from the stored rationale)
const COND_LABEL: Record<string, [string, (v: string) => string]> = {
  discharge_c_rate: ['discharge speed during the test', (v) => `${v}C`],
  retention_pct: ['capacity level that counts as end of life', (v) => `${v}%`],
};

/** 'discharge_c_rate: 5 vs 10' -> ['discharge speed…', '5C', '10C'] */
export function parseConditions(detail: string | null): [string, string, string] | null {
  const m = (detail ?? '').match(/^\s*(\w+):\s*(.+?)\s+vs\s+(.+?)\s*$/);
  if (!m) return null;
  const [, key, a, b] = m;
  const [label, fmt] = COND_LABEL[key] ?? [key.replace(/_/g, ' '), (v: string) => v];
  return [label, fmt(a), fmt(b)];
}

