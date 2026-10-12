<script setup lang="ts">
// The graph — the SIMILAR_TO neighbourhood of one cell instance (api: /api/graph).
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { DataSet } from 'vis-data'
import { Network, type Edge, type Node } from 'vis-network'
import { api, type Graph, type Instances } from '../api'
import { fmtInt } from '../format'

const meta = ref<Instances | null>(null)
const center = ref('')
const view = ref('behavior')
const k = ref(5)
const data = ref<Graph | null>(null)
const error = ref('')
const loading = ref(false)
const el = ref<HTMLDivElement | null>(null)
let network: Network | null = null

// viridis (matplotlib), 9 stops — node colour = measured cycle life, as in the paper's fade figure
const VIRIDIS = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725']
function viridis(t: number): string {
  const s = Math.min(Math.max(t, 0), 1) * (VIRIDIS.length - 1)
  const i = Math.min(Math.floor(s), VIRIDIS.length - 2), f = s - i
  const a = VIRIDIS[i].match(/\w\w/g)!.map((h) => parseInt(h, 16))
  const b = VIRIDIS[i + 1].match(/\w\w/g)!.map((h) => parseInt(h, 16))
  return '#' + a.map((v, j) => Math.round(v + (b[j] - v) * f).toString(16).padStart(2, '0')).join('')
}
const NO_LIFE = '#9ca3af'

const lifeRange = computed(() => {
  const lives = (data.value?.neighbors ?? []).map((n) => n.life).filter((l): l is number => l != null)
  if (data.value?.center_life != null) lives.push(data.value.center_life)
  return lives.length ? [Math.min(...lives), Math.max(...lives)] as const : null
})
const color = (life: number | null) => {
  const r = lifeRange.value
  if (life == null || !r) return NO_LIFE
  return viridis(r[1] === r[0] ? 0.5 : (life - r[0]) / (r[1] - r[0]))
}
const gradient = `linear-gradient(90deg, ${VIRIDIS.join(', ')})`

const dark = window.matchMedia('(prefers-color-scheme: dark)')
const ink = () => getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#1f2937'
const muted = () => getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#64748b'
const surface = () => getComputedStyle(document.documentElement).getPropertyValue('--surface').trim() || '#fff'

function draw() {
  const g = data.value
  if (!g || !el.value) return
  const nodes: Node[] = [{
    id: g.center, label: g.center, size: 28, borderWidth: 3, shape: 'dot',
    color: { background: color(g.center_life), border: ink() },
    title: g.center_life != null ? `${g.center} — cycle life ${fmtInt(g.center_life)}` : g.center,
    font: { size: 16, bold: { color: ink() } } as Node['font'],
  }]
  const edges: Edge[] = []
  for (const n of g.neighbors) {
    nodes.push({
      id: n.nbr, label: n.nbr, size: 16, shape: 'dot',
      color: { background: color(n.life), border: muted() },
      title: n.life != null ? `${n.nbr} — cycle life ${fmtInt(n.life)}, policy ${n.policy}` : n.nbr,
    })
    edges.push({
      from: g.center, to: n.nbr, width: 1 + 8 * n.weight, dashes: n.same_group ? [12, 9] : false,
      title: `weight ${n.weight.toFixed(3)}${n.same_group ? ' (same policy group)' : ''}`,
    })
  }
  const options = {
    nodes: { font: { color: ink(), size: 14, strokeWidth: 3, strokeColor: surface() }, borderWidth: 2 },
    edges: { color: { color: muted(), opacity: 0.55 }, smooth: false },
    physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { springLength: 140 }, stabilization: { iterations: 250 } },
    interaction: { hover: true, tooltipDelay: 80 },
  }
  network?.destroy()
  network = new Network(el.value, { nodes: new DataSet(nodes), edges: new DataSet(edges) }, options)
  network.once('stabilizationIterationsDone', () => network?.setOptions({ physics: { enabled: false } }))
}

async function load() {
  if (!center.value) return
  loading.value = true
  try {
    data.value = await api.graph(center.value, view.value, k.value)
    error.value = ''
    draw()
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    meta.value = await api.instances()
    // start on the cell the predictor's first preset uses, when present
    center.value = meta.value.instances.includes('b1c29') ? 'b1c29' : meta.value.instances[0] ?? ''
  } catch (e) {
    error.value = `Knowledge graph unavailable — ${(e as Error).message}`
  }
  dark.addEventListener('change', draw)
})
onBeforeUnmount(() => {
  dark.removeEventListener('change', draw)
  network?.destroy()
})

watch([center, view, k], load)
</script>

<template>
  <div class="page">
    <header class="page-head">
      <h1>🕸️ The graph</h1>
      <p>Each tested cell is linked to its most similar cells — by how it was charged (<em>condition</em>) or by how it faded early on (<em>behavior</em>). The predictor's data support is read off these links.</p>
    </header>

    <div v-if="error" class="notice notice-error">{{ error }}</div>

    <section v-if="meta" class="card controls">
      <div class="field">
        <label for="center">Cell instance</label>
        <select id="center" v-model="center">
          <option v-for="id in meta.instances" :key="id" :value="id">{{ id }}</option>
        </select>
      </div>
      <div class="field">
        <label>Similarity view</label>
        <div class="segmented" role="group" aria-label="Similarity view">
          <button v-for="v in meta.views" :key="v" :aria-pressed="v === view" @click="view = v">{{ v }}</button>
        </div>
      </div>
      <div class="field">
        <label for="k">Neighbors <output>{{ k }}</output></label>
        <input id="k" v-model.number="k" type="range" :min="meta.k[0]" :max="meta.k[1]" step="1">
      </div>
    </section>

    <section class="card graph-card" :class="{ stale: loading }">
      <div ref="el" class="network" />
      <div class="legend small">
        <span class="scale">
          <span>cycle life</span>
          <span v-if="lifeRange" class="num">{{ fmtInt(lifeRange[0]) }}</span>
          <i class="ramp" :style="{ background: gradient }" />
          <span v-if="lifeRange" class="num">{{ fmtInt(lifeRange[1]) }}</span>
        </span>
        <span><i class="line" /> edge width = similarity weight</span>
        <span><i class="line dashed" /> same charge-policy group (excluded from coverage under grouped evaluation)</span>
      </div>
    </section>

    <section v-if="data" class="card">
      <h2>Neighbors of {{ data.center }}<template v-if="data.center_life != null"> · cycle life {{ fmtInt(data.center_life) }}</template></h2>
      <div class="table-wrap nb">
        <table>
          <thead><tr><th>Neighbor</th><th class="num">Weight</th><th>Same policy group</th><th class="num">Cycle life</th><th>Charge policy</th></tr></thead>
          <tbody>
            <tr v-for="n in data.neighbors" :key="n.nbr">
              <td><i class="dot" :style="{ background: color(n.life) }" /> {{ n.nbr }}</td>
              <td class="num">{{ n.weight.toFixed(3) }}</td>
              <td>{{ n.same_group ? 'yes' : 'no' }}</td>
              <td class="num">{{ n.life != null ? fmtInt(n.life) : '—' }}</td>
              <td class="mono">{{ n.policy }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.controls { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1.4fr); gap: 20px; align-items: end; }
.controls .field { margin-top: 0; }
.graph-card { padding: 0; overflow: hidden; transition: opacity .15s; }
.stale { opacity: .6; }
.network { height: 540px; width: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; padding: 12px 18px; border-top: 1px solid var(--border); color: var(--text-muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.ramp { display: inline-block; width: 120px; height: 10px; border-radius: 3px; }
.line { display: inline-block; width: 26px; border-top: 3px solid var(--text-muted); }
.line.dashed { border-top-style: dashed; }
.num { font-variant-numeric: tabular-nums; }
.nb { margin-top: 12px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: middle; margin-right: 4px; }
@media (max-width: 860px) {
  .controls { grid-template-columns: minmax(0, 1fr); }
  .network { height: 420px; }
}
</style>
