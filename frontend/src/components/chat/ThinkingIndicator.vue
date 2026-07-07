<template>
  <div class="flex items-start gap-3 animate-fade-in">
    <img src="/images/logo.png" alt="OmniForge" class="w-12 h-12 flex-shrink-0" />
    <div class="flex items-center gap-1.5 py-2">
      <span
        v-for="i in 3"
        :key="i"
        class="thinking-dot w-2 h-2 rounded-full bg-primary/60 inline-block"
      ></span>
    </div>
  </div>
</template>
