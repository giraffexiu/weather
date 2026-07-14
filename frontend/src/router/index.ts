import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('../views/Dashboard.vue'),
  },
  {
    path: '/hourly',
    name: 'Hourly',
    component: () => import('../views/Hourly.vue'),
  },
  {
    path: '/daily',
    name: 'Daily',
    component: () => import('../views/Daily.vue'),
  },
  {
    path: '/explanation',
    name: 'Explanation',
    component: () => import('../views/ModelExplanation.vue'),
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router
