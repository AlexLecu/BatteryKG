<script setup lang="ts">
// Will it last? — the paper's coverage-gated predictor (api: /api/predictor, /api/predict).
// The early-cycle profile is always a real tested cell from the neighbour bank.
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api, type Prediction, type Predictor, type Recipe } from '../api'
import RangeBar from '../components/RangeBar.vue'
import { fmtInt } from '../format'

const info = ref<Predictor | null>(null)
const recipe = reactive<Recipe>({ profile: '', c_rate_1: 0, c_rate_2: 0, soc_transition_pct: 0 })
const result = ref<Prediction | null>(null)
const error = ref('')
const pending = ref(false)

function apply(p: Recipe) {
  Object.assign(recipe, p)
}

onMounted(async () => {
  try {
    info.value = await api.predictor()
    apply(info.value.presets.pass)
  } catch (e) {
    error.value = `Model artifacts unavailable — ${(e as Error).message}`
  }
})

// one request per pause in slider movement; a newer request cancels the older one
let timer: ReturnType<typeof setTimeout> | undefined
let inflight: AbortController | undefined
watch(recipe, () => {
  if (!recipe.profile) return
  clearTimeout(timer)
  timer = setTimeout(async () => {
    inflight?.abort()
    inflight = new AbortController()
    pending.value = true
    try {
      result.value = await api.predict({ ...recipe }, inflight.signal)
      error.value = ''
    } catch (e) {
      if ((e as Error).name !== 'AbortError') error.value = (e as Error).message
    } finally {
      pending.value = false
    }
  }, 150)
}, { deep: true })

const speedNote = (rate: number) => {
  const ah = info.value?.nominal_ah ?? 1.1
  return `${rate.toFixed(1)}C ≈ ${(rate * ah).toFixed(1)} A on this ${ah.toFixed(1)} Ah cell ≈ full charge in ~${Math.round(60 / rate)} min at this speed`
}

const band = computed(() => {
  const r = result.value
  if (!r) return null
  // the three models are independent, so the quantile pair can sit slightly off
  // the point model — widen the drawn band so the marker stays inside it
  return { lo: Math.min(r.band_lo, r.prediction_cycles), hi: Math.max(r.band_hi, r.prediction_cycles) }
})

const supportPct = computed(() => result.value ? Math.min(result.value.coverage / (2 * result.value.threshold), 1) * 100 : 0)
</script>

<template>
  <div class="page">
    <header class="page-head">
      <h1>🔋 Will it last?</h1>
      <p>Pick how hard a battery is fast-charged and get a cycle-life prediction — or an honest refusal when the model hasn't seen anything similar.</p>
    </header>

    <div v-if="!info && error" class="notice notice-error">{{ error }}</div>
    <div v-else-if="!info" class="loading">Loading the model…</div>

    <template v-else>
      <div class="presets">
        <button class="btn" @click="apply(info.presets.pass)">▶ Typical fast-charged cell</button>
        <button class="btn" @click="apply(info.presets.refuse)">▶ Unusual usage pattern</button>
      </div>

      <div class="grid-2">
        <section class="card">
          <h2>Charging recipe</h2>
          <p class="muted">How hard the battery is fast-charged — drag and watch the answer change.</p>

          <div class="field">
            <label for="c1">First charging speed <output>{{ recipe.c_rate_1.toFixed(1) }}C</output></label>
            <input id="c1" v-model.number="recipe.c_rate_1" type="range" step="0.1"
                   :min="info.limits.c_rate_1[0]" :max="info.limits.c_rate_1[1]">
            <span class="muted small">{{ speedNote(recipe.c_rate_1) }} · 1C = full charge in 1 hour</span>
          </div>
          <div class="field">
            <label for="soc">Switch point <output>{{ recipe.soc_transition_pct.toFixed(0) }}%</output></label>
            <input id="soc" v-model.number="recipe.soc_transition_pct" type="range" step="1"
                   :min="info.limits.soc_transition_pct[0]" :max="info.limits.soc_transition_pct[1]">
            <span class="muted small">State of charge at which the charger switches to the second speed</span>
          </div>
          <div class="field">
            <label for="c2">Second charging speed <output>{{ recipe.c_rate_2.toFixed(1) }}C</output></label>
            <input id="c2" v-model.number="recipe.c_rate_2" type="range" step="0.1"
                   :min="info.limits.c_rate_2[0]" :max="info.limits.c_rate_2[1]">
            <span class="muted small">{{ speedNote(recipe.c_rate_2) }}</span>
          </div>

          <h2 class="section">Measured battery profile</h2>
          <p class="muted">Early-cycle measurement taken from a real tested cell — this part can't come from a slider, it has to be measured.</p>
          <div class="field">
            <label for="profile">Real tested cell</label>
            <select id="profile" v-model="recipe.profile">
              <option v-for="p in info.profiles" :key="p.id" :value="p.id">{{ p.id }} — tested with {{ p.policy }}</option>
            </select>
          </div>
        </section>

        <section class="card answer" :class="{ stale: pending }" aria-live="polite">
          <div v-if="error" class="notice notice-error">{{ error }}</div>

          <template v-if="result">
            <div v-if="result.abstain" class="refusal">
              <div class="verdict">❌ I won't guess</div>
              <p>
                Too few similar batteries in my data (support {{ result.coverage.toFixed(1) }} vs required
                {{ result.threshold.toFixed(1) }}) — a standard model would have answered anyway and been
                badly wrong on a case like this.
              </p>
            </div>
            <template v-else-if="band">
              <div class="verdict">≈ {{ fmtInt(result.prediction_cycles) }} cycles</div>
              <p class="muted">Best estimate of full charge–discharge cycles before the battery fades to 80% of its rated capacity.</p>
              <RangeBar :point="result.prediction_cycles" :lo="band.lo" :hi="band.hi" />
              <p class="muted small">
                Likely range — the model's middle-68% uncertainty band, stretched to include the best
                estimate when its separate quantile models disagree slightly.
              </p>
            </template>

            <div class="gauge">
              <div class="gauge-head">
                <span>📊 Data support: how many similar batteries I've seen</span>
                <strong>{{ result.coverage.toFixed(1) }}</strong>
              </div>
              <div class="gauge-track" role="meter" :aria-valuenow="result.coverage" aria-valuemin="0"
                   :aria-valuemax="2 * result.threshold" aria-label="Data support">
                <div class="gauge-fill" :class="{ low: result.abstain }" :style="{ width: supportPct + '%' }" />
                <div class="gauge-threshold" title="Refusal threshold" />
              </div>
              <div class="muted small">Refuses below {{ result.threshold.toFixed(1) }} (the mark in the middle)</div>
            </div>
          </template>
          <div v-else-if="!error" class="loading">Predicting…</div>

          <details>
            <summary>ℹ️ How this works</summary>
            <ul>
              <li>The predictor is the paper's graph-mediated XGBoost model, served unchanged through <code>predict_with_gate()</code>; nothing here is simulated.</li>
              <li><strong>Sliders</strong> set the charge-protocol features (<code>c_rate_1</code>, <code>c_rate_2</code>, <code>soc_transition_pct</code>), ranged to the {{ info.n_training_cells }} real Severson training cells.</li>
              <li>Harder charging generally shortens cycle life, but the Severson study showed the <em>combination</em> of the three settings matters, not just the peak rate — charging policy alone spanned 147–2236 cycles.</li>
              <li><strong>The profile</strong> supplies the measured early-cycle features (capacity-fade shape over the first 100 cycles) plus batch — these are physical measurements, so they come from a real tested cell, honestly labeled.</li>
              <li><strong>Data support</strong> is the model's <em>coverage</em>: the summed similarity of the {{ info.k_neighbors }} most similar training cells in early-fade behavior space. Below the threshold of {{ info.threshold.toFixed(2) }} (chosen on the cross-validated risk–coverage curve) the system <strong>abstains</strong> — it refuses to answer rather than extrapolate.</li>
              <li>The gauge reacts to the chosen profile (measured behavior), while the sliders steer the prediction itself — matching what the model actually consumes.</li>
              <li>Trained on {{ info.n_training_cells }} cells; grouped-CV error {{ info.cv_rmse_cycles.toFixed(0) }} cycles RMSE ({{ info.cv_mape_pct.toFixed(1) }}% MAPE). Both preset buttons replay real training-bank cells verified to produce their outcome.</li>
            </ul>
          </details>
        </section>
      </div>
    </template>
  </div>
</template>

<style scoped>
.presets { display: flex; gap: 10px; flex-wrap: wrap; }
.section { margin-top: 26px; }
.answer { display: flex; flex-direction: column; gap: 10px; transition: opacity .15s; }
.answer.stale { opacity: .7; }
.answer p { margin: 0; }
.verdict { font-size: 40px; font-weight: 750; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
.refusal { background: var(--bad-soft); border-radius: var(--radius-sm); padding: 14px 16px; }
.refusal .verdict { color: var(--bad); font-size: 32px; }
.refusal p { margin-top: 6px; }

.gauge { margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }
.gauge-head { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; }
.gauge-head strong { font-variant-numeric: tabular-nums; }
.gauge-track { position: relative; height: 10px; border-radius: 999px; background: var(--surface-2); overflow: hidden; }
.gauge-fill { height: 100%; background: var(--good); border-radius: 999px; transition: width .25s; }
.gauge-fill.low { background: var(--bad); }
.gauge-threshold { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: var(--text); opacity: .55; }
@media (max-width: 560px) { .verdict { font-size: 32px; } }
</style>
