<script setup lang="ts">
// Measured cycle lives as a 30-bin histogram, with the datasheet claim and the median marked.
import { computed, ref } from 'vue'
import { fmtInt } from '../format'

const props = defineProps<{ values: number[]; claim?: number; median?: number }>()

const BINS = 30
const W = 720, H = 360, M = { top: 34, right: 18, bottom: 52, left: 52 }
const iw = W - M.left - M.right, ih = H - M.top - M.bottom

const domain = computed(() => {
  const xs = [...props.values, ...(props.claim != null ? [props.claim] : [])]
  return [Math.min(...props.values), Math.max(...xs)] as const
})

const bins = computed(() => {
  // equal-width bins over the measured range (as matplotlib's hist(bins=30))
  const lo = Math.min(...props.values), hi = Math.max(...props.values)
  const width = (hi - lo) / BINS || 1
  const counts = Array.from({ length: BINS }, (_, i) => ({ x0: lo + i * width, x1: lo + (i + 1) * width, n: 0 }))
  for (const v of props.values) counts[Math.min(Math.floor((v - lo) / width), BINS - 1)].n++
  return counts
})

// round y ticks (1, 2, 5, 10, …) with the top tick above the tallest bar
const yStep = computed(() => {
  const m = Math.max(...bins.value.map((b) => b.n), 1)
  return [1, 2, 5, 10, 20, 50, 100].find((s) => m / s <= 6) ?? 200
})
const yMax = computed(() => Math.ceil(Math.max(...bins.value.map((b) => b.n), 1) / yStep.value) * yStep.value)

const pad = computed(() => (domain.value[1] - domain.value[0]) * 0.04)
const x = (v: number) => M.left + ((v - domain.value[0] + pad.value) / (domain.value[1] - domain.value[0] + 2 * pad.value)) * iw
const y = (n: number) => M.top + ih - (n / yMax.value) * ih

const xTicks = computed(() => {
  const [a, b] = domain.value
  const step = b - a > 1500 ? 500 : 250
  const out = []
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) out.push(t)
  return out
})
const yTicks = computed(() => Array.from({ length: yMax.value / yStep.value + 1 }, (_, i) => i * yStep.value))

const hovered = ref<number | null>(null)
const tip = computed(() => {
  if (hovered.value == null) return null
  const b = bins.value[hovered.value]
  const cx = x((b.x0 + b.x1) / 2)
  return { b, left: (cx / W) * 100, flip: cx > W * 0.7 }
})

// labels sit on the side facing away from the other line, unless that runs off the chart
const claimRight = computed(() => props.claim != null && x(props.claim) < W * 0.72
  && (props.median == null || props.claim >= props.median))
const medianRight = computed(() => props.median != null && x(props.median) < W * 0.72
  && !(props.claim != null && props.claim > props.median && x(props.median) - M.left > 180))
</script>

<template>
  <div class="hist">
    <svg :viewBox="`0 0 ${W} ${H}`" role="img"
         :aria-label="`Histogram of ${values.length} measured cycle lives`">
      <g class="grid">
        <line v-for="t in yTicks" :key="t" :x1="M.left" :x2="W - M.right" :y1="y(t)" :y2="y(t)" />
      </g>
      <g class="axis">
        <text v-for="t in yTicks" :key="t" :x="M.left - 8" :y="y(t) + 4" text-anchor="end">{{ t }}</text>
        <text v-for="t in xTicks" :key="t" :x="x(t)" :y="M.top + ih + 20" text-anchor="middle">{{ fmtInt(t) }}</text>
        <text :x="M.left + iw / 2" :y="H - 8" text-anchor="middle" class="axis-title">cycles until the battery faded to 80% capacity</text>
        <text :transform="`translate(14 ${M.top + ih / 2}) rotate(-90)`" text-anchor="middle" class="axis-title">tested cells (n={{ values.length }})</text>
      </g>

      <g>
        <rect v-for="(b, i) in bins" :key="i" class="bar" :class="{ dim: hovered != null && hovered !== i }"
              :x="x(b.x0) + 1" :y="y(b.n)" :width="Math.max(x(b.x1) - x(b.x0) - 2, 1)" :height="M.top + ih - y(b.n)"
              :rx="b.n ? 2 : 0" />
      </g>

      <g v-if="median != null" class="median">
        <line :x1="x(median)" :x2="x(median)" :y1="M.top - 6" :y2="M.top + ih" />
        <text :x="x(median) + (medianRight ? 8 : -8)" :y="M.top + ih * 0.42" :text-anchor="medianRight ? 'start' : 'end'">
          half the cells died here · {{ fmtInt(median) }}
        </text>
      </g>
      <g v-if="claim != null" class="claim">
        <line :x1="x(claim)" :x2="x(claim)" :y1="M.top - 6" :y2="M.top + ih" />
        <text :x="x(claim) + (claimRight ? 8 : -8)" :y="M.top + 8" :text-anchor="claimRight ? 'start' : 'end'">
          the promise · {{ fmtInt(claim) }}
        </text>
      </g>

      <!-- hit targets wider than the bars -->
      <rect v-for="(b, i) in bins" :key="`hit${i}`" :x="x(b.x0)" :y="M.top" :width="x(b.x1) - x(b.x0)" :height="ih"
            fill="transparent" @mouseenter="hovered = i" @mouseleave="hovered = null" />
    </svg>

    <div v-if="tip" class="tip" :style="{ left: tip.left + '%' }" :class="{ flip: tip.flip }">
      <strong>{{ tip.b.n }} {{ tip.b.n === 1 ? 'cell' : 'cells' }}</strong>
      <span>{{ fmtInt(tip.b.x0) }} – {{ fmtInt(tip.b.x1) }} cycles</span>
    </div>
  </div>
</template>

<style scoped>
.hist { position: relative; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
.grid line { stroke: var(--border); stroke-width: 1; }
.axis text { fill: var(--text-muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }
.axis .axis-title { font-size: 13px; }
.bar { fill: var(--primary); transition: opacity .1s; }
.bar.dim { opacity: .45; }
.median line { stroke: var(--text); stroke-width: 2; }
.median text, .claim text { paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
.median text { fill: var(--text); font-size: 13px; }
.claim line { stroke: var(--accent); stroke-width: 3; stroke-dasharray: 8 5; }
.claim text { fill: var(--text); font-size: 13.5px; font-weight: 700; }
.tip {
  position: absolute; top: 8px; transform: translateX(12px); pointer-events: none;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 10px; font-size: 13px; display: flex; flex-direction: column; white-space: nowrap;
  box-shadow: 0 4px 14px rgba(0, 0, 0, .12);
}
.tip.flip { transform: translateX(calc(-100% - 12px)); }
</style>
