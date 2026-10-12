<script setup lang="ts">
// Who is lying? — spec sheets that disagree with each other (api: /api/specs).
// Headline as in the paper (Table 8): a comparison is a property present in BOTH documents.
import { computed, onMounted, ref } from 'vue'
import { api, type SpecRow, type Specs } from '../api'
import { fmtValue, parseConditions, propLabel } from '../format'

const data = ref<Specs | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    data.value = await api.specs()
  } catch (e) {
    error.value = `Knowledge graph unavailable — ${(e as Error).message}`
  }
})

const isHard = (r: SpecRow) => r.verdict === 'document_variant_conflict'
const sameValue = (r: SpecRow) => r.value_a === r.value_b && !isHard(r)

const cards = computed(() => (data.value?.problems ?? []).map((r) => ({
  r,
  hard: isHard(r),
  internal: r.provenance === 'manufacturer_internal',
  same: sameValue(r),
  cond: sameValue(r) ? parseConditions(r.detail) : null,
})))
</script>

<template>
  <div class="page">
    <header class="page-head">
      <h1>🔍 Who is lying?</h1>
      <p>We compared the same battery's numbers across different official documents — these are the places where the paperwork disagrees with itself.</p>
    </header>

    <div v-if="error" class="notice notice-error">{{ error }}</div>
    <div v-else-if="!data" class="loading">Loading…</div>

    <template v-else>
      <section class="headline">
        <h2><span class="big">{{ data.problems.length }}</span> of {{ data.n_comparisons }} spec comparisons don't add up</h2>
        <p class="muted">{{ data.n_single_document }} more specs appear in only one document — nothing to compare them against.</p>
        <div class="legend small">
          <span><i class="sw hard" />different numbers in the documents</span>
          <span><i class="sw soft" />same number, different test conditions</span>
        </div>
      </section>

      <section class="cards">
        <article v-for="({ r, hard, internal, same, cond }, i) in cards" :key="i" class="card spec" :class="hard ? 'hard' : 'soft'">
          <header class="spec-head">
            <h3>{{ r.cell }} — {{ propLabel(r.property) }}</h3>
            <span class="tag">{{ internal ? '🏭 same manufacturer, different documents' : '🌐 across the ecosystem' }}</span>
          </header>

          <p v-if="same" class="small">
            ⚠️ Same number, different test conditions — both documents say
            <strong>{{ fmtValue(r.property, r.value_a) }}</strong><template v-if="cond">, but disagree on the <em>{{ cond[0] }}</em>:</template><template v-else>:</template>
          </p>

          <div v-if="!same || cond" class="versus">
            <div class="side">
              <span class="src mono">{{ r.source_a }}</span>
              <span class="val">{{ same && cond ? cond[1] : fmtValue(r.property, r.value_a) }}</span>
            </div>
            <span class="vs">vs</span>
            <div class="side">
              <span class="src mono">{{ r.source_b }}</span>
              <span class="val">{{ same && cond ? cond[2] : fmtValue(r.property, r.value_b) }}</span>
            </div>
          </div>

          <details>
            <summary>ℹ️ Show me the exact source lines</summary>
            <p class="small"><strong class="mono">{{ r.source_a }}</strong></p>
            <blockquote>{{ r.fragment_a }}</blockquote>
            <p class="small"><strong class="mono">{{ r.source_b }}</strong></p>
            <blockquote>{{ r.fragment_b }}</blockquote>
            <p class="small"><em>Stored analysis:</em> {{ r.detail }}</p>
            <p v-if="!hard" class="small muted">
              (Same value, but stated under different test conditions — an apples-to-oranges risk, not necessarily a contradiction.)
            </p>
          </details>
        </article>
      </section>

      <details class="card all">
        <summary>
          ℹ️ All {{ data.rows.length }} stored rows: {{ data.n_comparisons }} comparisons (incl. {{ data.n_consistent }} consistent)
          + {{ data.n_single_document }} single-document specs
        </summary>
        <div class="table-wrap rows">
          <table>
            <thead>
              <tr><th>Cell</th><th>Property</th><th>Value A</th><th>Value B</th><th>Source A</th><th>Source B</th><th>Verdict</th><th>Provenance</th><th>Detail</th></tr>
            </thead>
            <tbody>
              <tr v-for="(r, i) in data.rows" :key="i">
                <td>{{ r.cell }}</td>
                <td>{{ propLabel(r.property) }}</td>
                <td class="nowrap">{{ fmtValue(r.property, r.value_a) }}</td>
                <td class="nowrap">{{ fmtValue(r.property, r.value_b) }}</td>
                <td class="mono">{{ r.source_a }}</td>
                <td class="mono">{{ r.source_b }}</td>
                <td class="mono">{{ r.verdict }}</td>
                <td class="mono">{{ r.provenance }}</td>
                <td>{{ r.detail }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="small muted">
          Values are compared after unit normalization (2% tolerance; bound satisfaction counts as consistent) by
          <code>src/kg/spec_consistency.py</code>; each row stores which documents said what, so every disagreement traces to its sources.
        </p>
      </details>
    </template>
  </div>
</template>

<style scoped>
.headline { display: flex; flex-direction: column; gap: 6px; }
.headline h2 { font-size: 22px; }
.headline p { margin: 0; }
.big { font-size: 34px; font-weight: 800; color: var(--bad); }
.legend { display: flex; gap: 18px; flex-wrap: wrap; color: var(--text-muted); margin-top: 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.sw.hard { background: var(--bad); }
.sw.soft { background: var(--warn); }

.cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: start; }
.spec { border-left: 4px solid var(--warn); display: flex; flex-direction: column; gap: 10px; }
.spec.hard { border-left-color: var(--bad); }
.spec p { margin: 0; }
.spec details { margin-top: 2px; }
.spec-head { display: flex; flex-direction: column; gap: 4px; }
.tag { font-size: 13px; color: var(--text-muted); }

.versus { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); gap: 12px; align-items: center; }
.side { display: flex; flex-direction: column; gap: 4px; background: var(--surface-2); border-radius: var(--radius-sm); padding: 10px 12px; min-width: 0; }
.src { font-size: 12px; color: var(--text-muted); overflow-wrap: anywhere; }
.val { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.vs { color: var(--text-muted); font-weight: 700; }

.all summary { font-size: 14px; }
.all { margin-top: 0; border-top: 1px solid var(--border); padding-top: 18px; }
.rows { max-height: 560px; overflow: auto; margin: 10px 0; }
.nowrap { white-space: nowrap; }

@media (max-width: 860px) { .cards { grid-template-columns: minmax(0, 1fr); } }
</style>
