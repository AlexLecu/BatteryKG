<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { PAGES } from './pages'

const open = ref(false)
const route = useRoute()
watch(() => route.path, () => { open.value = false })
</script>

<template>
  <header class="topbar">
    <div class="container topbar-inner">
      <RouterLink to="/" class="brand">🔋 BatteryKG</RouterLink>
      <button class="menu-btn" :aria-expanded="open" aria-label="Menu" @click="open = !open">☰</button>
      <nav :class="{ open }">
        <RouterLink v-for="p in PAGES" :key="p.to" :to="p.to">{{ p.icon }} {{ p.short }}</RouterLink>
      </nav>
    </div>
  </header>

  <main class="container">
    <RouterView />
  </main>

  <footer class="container footer">
    <span>BatteryKG · Technical University of Cluj-Napoca</span>
    <span>
      <a href="https://github.com/AlexLecu/BatteryKG" target="_blank" rel="noopener">GitHub</a>
      · <a href="https://doi.org/10.3390/batteries12080314" target="_blank" rel="noopener">Paper</a>
    </span>
  </footer>
</template>

<style scoped>
.topbar {
  position: sticky; top: 0; z-index: 10;
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border);
}
.topbar-inner { display: flex; align-items: center; gap: 24px; height: 58px; }
.brand { font-weight: 750; font-size: 18px; color: var(--text); white-space: nowrap; }
.brand:hover { text-decoration: none; }
nav { display: flex; gap: 4px; margin-left: auto; }
nav a {
  padding: 7px 12px; border-radius: var(--radius-sm); color: var(--text-muted);
  font-weight: 550; font-size: 14.5px; white-space: nowrap;
}
nav a:hover { background: var(--surface-2); color: var(--text); text-decoration: none; }
nav a.router-link-active { color: var(--primary); background: var(--primary-soft); }
.menu-btn {
  display: none; margin-left: auto; background: none; border: 1px solid var(--border);
  border-radius: var(--radius-sm); color: var(--text); font-size: 18px; padding: 4px 10px; cursor: pointer;
}
.footer {
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
  padding-top: 18px; padding-bottom: 28px; border-top: 1px solid var(--border);
  color: var(--text-muted); font-size: 13px;
}

@media (max-width: 860px) {
  .menu-btn { display: block; }
  nav {
    display: none; position: absolute; top: 58px; left: 0; right: 0;
    flex-direction: column; padding: 8px 16px 14px; gap: 2px;
    background: var(--surface); border-bottom: 1px solid var(--border);
  }
  nav.open { display: flex; }
  nav a { padding: 10px 12px; }
}
</style>
