import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  scrollBehavior: () => ({ top: 0 }),
  routes: [
    { path: '/', name: 'home', component: HomeView, meta: { title: 'BatteryKG · Battery promises vs measurements' } },
    { path: '/will-it-last', name: 'predict', component: () => import('../views/PredictView.vue'), meta: { title: 'Will it last? · BatteryKG' } },
    { path: '/promise-vs-reality', name: 'promise', component: () => import('../views/PromiseView.vue'), meta: { title: 'Promise vs reality · BatteryKG' } },
    { path: '/who-is-lying', name: 'specs', component: () => import('../views/SpecsView.vue'), meta: { title: 'Who is lying? · BatteryKG' } },
    { path: '/graph', name: 'graph', component: () => import('../views/GraphView.vue'), meta: { title: 'The graph · BatteryKG' } },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

router.afterEach((to) => {
  if (typeof to.meta.title === 'string') document.title = to.meta.title
})

export default router
