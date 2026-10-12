<script setup lang="ts">
// The prediction on a cycles axis: the uncertainty band as a bar, the best estimate as a dot.
import { computed } from 'vue'
import { fmtInt } from '../format'

const props = defineProps<{ point: number; lo: number; hi: number }>()

const W = 640, H = 92, PAD = 14, Y = 34
const max = computed(() => Math.max(2400, props.hi * 1.15))
const x = (v: number) => PAD + (v / max.value) * (W - 2 * PAD)
const ticks = computed(() => {
  const step = max.value > 3000 ? 1000 : 500
  return Array.from({ length: Math.floor(max.value / step) + 1 }, (_, i) => i * step)
})
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="rangebar" role="img"
       :aria-label="`Best estimate ${fmtInt(point)} cycles, likely range ${fmtInt(lo)} to ${fmtInt(hi)}`">
    <line :x1="PAD" :x2="W - PAD" :y1="Y" :y2="Y" class="axis" />
    <text :x="(x(lo) + x(hi)) / 2" :y="Y - 18" text-anchor="middle" class="band-label">{{ fmtInt(lo) }} – {{ fmtInt(hi) }}</text>
    <g v-for="t in ticks" :key="t">
      <line :x1="x(t)" :x2="x(t)" :y1="Y + 14" :y2="Y + 19" class="tick" />
      <text :x="x(t)" :y="Y + 33" text-anchor="middle" class="tick-label">{{ fmtInt(t) }}</text>
    </g>
    <g>
      <rect :x="x(lo)" :y="Y - 10" :width="Math.max(x(hi) - x(lo), 4)" height="20" rx="4" class="band" />
      <circle :cx="x(point)" :cy="Y" r="9" class="dot" />
      <rect :x="x(lo) - 8" :y="Y - 18" :width="Math.max(x(hi) - x(lo), 4) + 16" height="36" fill="transparent">
        <title>Likely {{ fmtInt(lo) }} – {{ fmtInt(hi) }} cycles · best estimate {{ fmtInt(point) }}</title>
      </rect>
    </g>
  </svg>
</template>

<style scoped>
.rangebar { width: 100%; height: auto; display: block; overflow: visible; }
.axis { stroke: var(--border); stroke-width: 1; }
.tick { stroke: var(--text-muted); stroke-width: 1; opacity: .6; }
.tick-label { fill: var(--text-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.band-label { fill: var(--text-muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }
.band { fill: var(--primary); opacity: .25; }
.dot { fill: var(--primary); stroke: var(--surface); stroke-width: 2; }
</style>
