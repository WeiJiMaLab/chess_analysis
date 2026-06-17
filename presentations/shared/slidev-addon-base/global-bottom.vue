<template>
  <footer
    v-if="$nav.currentPage > 1 && $slidev.configs.showSlideNumbers !== false"
    class="absolute bottom-4 right-4 text-[10px] text-gray-400 font-mono opacity-50 z-50 pointer-events-none"
  >
    {{ $nav.currentPage }}
  </footer>
</template>
