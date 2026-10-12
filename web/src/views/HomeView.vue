<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type Health } from '../api'
import { PAGES } from '../pages'

const health = ref<Health | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    health.value = await api.health()
  } catch (e) {
    error.value = (e as Error).message
  }
})

const date = (iso: string) => new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })
</script>

<template>
  <div class="page">
    <section class="hero">
      <h1>🔋 BatteryKG</h1>
      <p class="lead">
        We check what battery makers promise against what independent tests measure — and our
        predictor refuses to guess when it shouldn't.
      </p>
    </section>

    <section class="cards">
      <RouterLink v-for="p in PAGES" :key="p.to" :to="p.to" class="card tile">
        <span class="tile-icon" aria-hidden="true">{{ p.icon }}</span>
        <h2>{{ p.title }}</h2>
        <p class="muted">{{ p.blurb }}</p>
        <span class="tile-cta">{{ p.cta }} →</span>
      </RouterLink>
    </section>

    <p class="muted source">
      <template v-if="error">Knowledge graph offline — {{ error }}</template>
      <template v-else-if="health?.kg === 'snapshot' && health.snapshot_generated_at">
        Read-only demo: knowledge-graph answers are served from a static snapshot recorded
        {{ date(health.snapshot_generated_at) }} — same data, no live database.
      </template>
      <template v-else-if="health?.kg === 'live'">Live knowledge graph connected.</template>
      <template v-else-if="health">Knowledge graph offline — {{ health.kg_error }}</template>
      Every number comes from the knowledge graph, the trained model artifacts, or an experiment
      report — nothing is mocked.
    </p>
  </div>
</template>

<style scoped>
.hero { padding: 20px 0 8px; display: flex; flex-direction: column; gap: 10px; }
.hero h1 { font-size: 34px; }
.lead { margin: 0; font-size: 19px; max-width: 760px; color: var(--text); }

.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
.tile { display: flex; flex-direction: column; gap: 8px; color: var(--text); transition: border-color .15s, transform .15s; }
.tile:hover { text-decoration: none; border-color: var(--primary); transform: translateY(-1px); }
.tile p { margin: 0; flex: 1; }
.tile-icon { font-size: 28px; }
.tile-cta { color: var(--primary); font-weight: 650; font-size: 14px; }
.source { border-top: 1px solid var(--border); padding-top: 16px; margin: 4px 0 0; }

@media (max-width: 1020px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .cards { grid-template-columns: minmax(0, 1fr); } }
</style>
