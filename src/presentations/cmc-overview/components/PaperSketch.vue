<template>
  <div class="sketch-container">
    <div class="timeline-spine"></div>
    
    <div class="sketch-step" v-for="(step, i) in steps" :key="i">
      <div class="step-marker" :class="'bg-' + step.color">
        {{ i + 1 }}
      </div>
      
      <div class="step-content">
        <h2 :class="'text-' + step.color">{{ step.title }}</h2>
        <p class="step-subtitle" v-if="step.subtitle">{{ step.subtitle }}</p>
        <ul class="step-list">
          <li v-for="(item, j) in step.items" :key="j" v-html="item"></li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
const steps = [
  {
    title: 'Part 1: Motivation',
    subtitle: 'The Stopping Problem',
    color: 'accent',
    items: [
      'Knowing when to think is hard but important (costs time and money)', 
      'People seem to be pretty good at it but we want to build it into ML',
      '<b>Goal:</b> Build a model that knows when to think vs act.'
    ]
  },
  {
    title: 'Part 2: Methods & Performance',
    subtitle: 'Economy of Thought',
    color: 'primary',
    items: [
      'Chess is a good test domain for this problem (data, models)',
      'Model: GNN + Halt Controller, we built it in a smart way',
      'Our model works and it does what we say it does'
    ]
  },
  {
    title: 'Part 3: Validation & Alignment',
    subtitle: 'Human Existence Proof',
    color: 'success',
    items: [
      'How do we know if the model is doing smart things? People do smart things.',
      'Things people do: Heavy-tailed thought, mid-game peak, clock-sensitivity.',
      '<b>Question:</b> How else can we show our model is doing something interesting?'
    ]
  }
]
</script>

<style scoped>
.sketch-container {
  position: relative;
  max-width: 650px;
  margin: 0 auto;
  padding-left: 2.5rem;
  padding-top: 0;
}

.timeline-spine {
  position: absolute;
  left: 31px;
  top: 15px;
  bottom: 25px;
  width: 2px;
  background: var(--border-subtle);
  opacity: 0.15;
}

.sketch-step {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1.25rem;
  position: relative;
  align-items: flex-start;
}

.step-marker {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 11px;
  font-weight: 900;
  flex-shrink: 0;
  z-index: 1;
  margin-top: 3px;
}

.step-content h2 {
  font-size: 1.15rem;
  margin: 0;
  font-weight: 800;
  line-height: 1.2;
}

.step-subtitle {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  opacity: 0.6;
  margin: 0.15rem 0 0.5rem 0;
}

.step-list {
  list-style: none;
  padding: 0;
  margin: 0;
  font-size: 0.88rem;
  opacity: 0.9;
}

.step-list li {
  position: relative;
  padding-left: 1.2rem;
  margin-bottom: 0.25rem;
  line-height: 1.35;
}

.step-list li::before {
  content: '→';
  position: absolute;
  left: 0;
  opacity: 0.4;
  font-weight: bold;
}

.bg-accent { background: var(--color-accent); }
.bg-primary { background: var(--color-primary); }
.bg-success { background: var(--color-success); }

.text-accent { color: var(--color-accent); }
.text-primary { color: var(--color-primary); }
.text-success { color: var(--color-success); }
</style>
