<script setup lang="ts">
// Promise vs reality — the manufacturer's cycle-life claim over what was measured (api: /api/promise).
import { onMounted, ref, watch } from 'vue'
import { api, type Cell, type Promise_ } from '../api'
import LifeHistogram from '../components/LifeHistogram.vue'
import { fmtInt, fmtPct, fmtValue, propLabel } from '../format'

const cells = ref<Cell[]>([])
const model = ref('')
const data = ref<Promise_ | null>(null)
const error = ref('')
const loading = ref(false)

onMounted(async () => {
  try {
    cells.value = (await api.cells()).cells
    model.value = cells.value[0]?.model ?? ''
  } catch (e) {
    error.value = `Knowledge graph unavailable — ${(e as Error).message}`
  }
})

watch(model, async (m) => {
  if (!m) return
  loading.value = true
  try {
    data.value = await api.promise(m)
    error.value = ''
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="page">
    <header class="page-head">
      <h1>⚖️ Promise vs reality</h1>
      <p>The cycle life a datasheet promises, drawn over how long independently tested cells of the same model actually lasted.</p>
    </header>

    <div v-if="error" class="notice notice-error">{{ error }}</div>

    <div v-if="cells.length" class="segmented" role="group" aria-label="Commercial cell">
      <button v-for="c in cells" :key="c.model" :aria-pressed="c.model === model" @click="model = c.model">
        {{ c.model }}
      </button>
    </div>

    <div v-if="!data && !error" class="loading">Loading…</div>

    <template v-else-if="data">
      <div v-if="data.mismatch" class="notice notice-warn">{{ data.mismatch }}</div>

      <div v-else-if="data.values.length" class="grid-promise" :class="{ stale: loading }">
        <section class="card">
          <LifeHistogram :values="data.values" :claim="data.discrepancy?.claim" :median="data.discrepancy?.med" />
          <p class="muted small caption">
            Every bar counts real tested cells from the knowledge graph (n={{ data.values.length }}, the same
            measurement set the stored comparison uses); the dashed orange line is the manufacturer's datasheet claim.
          </p>
        </section>

        <section class="card side">
          <template v-if="data.discrepancy">
            <div class="stats">
              <div class="stat">
                <span class="stat-label">Promised</span>
                <span class="stat-value">{{ fmtInt(data.discrepancy.claim) }}</span>
                <span class="muted small">cycles</span>
              </div>
              <div class="stat">
                <span class="stat-label">Measured (median)</span>
                <span class="stat-value">{{ fmtInt(data.discrepancy.med) }}</span>
                <span class="stat-delta">
                  {{ data.discrepancy.gap < 0 ? '↓' : '↑' }} {{ fmtPct(data.discrepancy.gap, true) }} vs claim
                </span>
              </div>
            </div>
            <div v-if="!data.discrepancy.comparable" class="notice notice-warn">
              ⚠️ Test conditions differ — context, not a verdict.
            </div>
            <details>
              <summary>ℹ️ How this works</summary>
              <ul>
                <li><strong>Claim</strong>: “{{ data.discrepancy.claim_conditions }}” (datasheet wording, page-referenced in the graph).</li>
                <li><strong>Measured</strong>: {{ data.discrepancy.n }} cells, {{ data.discrepancy.measured_conditions }}.</li>
                <li>Spread: {{ fmtInt(data.discrepancy.mn) }}–{{ fmtInt(data.discrepancy.mx) }} cycles (middle half {{ fmtInt(data.discrepancy.q1) }}–{{ fmtInt(data.discrepancy.q3) }}).</li>
                <li>Stored rationale for the comparability flag: <em>{{ data.discrepancy.rationale }}</em></li>
                <li>The comparison is a stored graph fact (a claim-vs-measured record with full provenance — every number traces to its source), recomputed by <code>python -m src.kg.discrepancy</code>, not made up on this page.</li>
              </ul>
            </details>
          </template>
          <div v-else class="notice notice-info">
            Measurements exist but no claim-vs-measured comparison is stored in the graph for this cell yet.
          </div>
        </section>
      </div>

      <template v-else>
        <div class="notice notice-info">
          No independent cycling measurements for this cell in the graph yet — below is what the
          manufacturer promises, awaiting an independent check.
        </div>
        <section v-if="data.claims.length" class="card" :class="{ stale: loading }">
          <h2>What the datasheets claim</h2>
          <div class="table-wrap claims">
            <table>
              <thead>
                <tr><th>Property</th><th class="num">Value</th><th>Stated conditions</th><th>Document</th><th class="num">Page</th></tr>
              </thead>
              <tbody>
                <tr v-for="(c, i) in data.claims" :key="i">
                  <td>{{ propLabel(c.property) }}</td>
                  <td class="num">{{ fmtValue(c.property, c.value) }}</td>
                  <td>{{ c.stated_conditions ?? '—' }}</td>
                  <td class="mono">{{ c.document }}</td>
                  <td class="num">{{ c.page ?? '—' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <details>
            <summary>ℹ️ How this works</summary>
            <p class="small">
              Claims are hand-extracted from the official datasheet PDFs with page numbers and stated test
              conditions; “unspecified” means the datasheet itself omits the conditions — that omission is
              recorded, not filled in.
            </p>
          </details>
        </section>
      </template>
    </template>
  </div>
</template>

<style scoped>
.grid-promise { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 20px; align-items: start; }
.stale { opacity: .6; transition: opacity .15s; }
.caption { margin: 10px 0 0; }
.side { display: flex; flex-direction: column; gap: 14px; }
.side details { margin-top: 0; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.stat-delta { color: var(--text-muted); }
.claims { max-height: 520px; overflow: auto; margin-top: 12px; }
.segmented button { font-weight: 550; }
@media (max-width: 960px) { .grid-promise { grid-template-columns: minmax(0, 1fr); } }
</style>
