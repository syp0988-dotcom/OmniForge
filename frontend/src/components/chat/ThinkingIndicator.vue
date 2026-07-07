<template>
  <div class="flex items-start animate-fade-in">
    <div class="flex items-center gap-1.5 py-2 ml-14">
      <span
        v-for="i in 3"
        :key="i"
        class="thinking-dot w-2 h-2 rounded-full bg-primary/60 inline-block"
      ></span>
    </div>
  </div>
</template>
