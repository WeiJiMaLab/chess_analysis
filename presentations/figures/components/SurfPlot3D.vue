<script setup>
import { onMounted, ref, onUnmounted } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const props = defineProps({
  csvPath: {
    type: String,
    required: true
  },
  title: {
    type: String,
    default: '3D Surface Plot'
  },
  zScale: {
    type: Number,
    default: 3.0
  }
})

const container = ref(null)
let scene, camera, renderer, controls, frameId

const init = async () => {
  try {
    // Parse CSV
    const response = await fetch(props.csvPath)
    if (!response.ok) throw new Error(`Failed to fetch CSV: ${response.statusText}`)
    const text = await response.text()
    const rows = text.trim().split('\n').map(r => r.split(','))
    
    const headers = rows[0].slice(1).map(Number)
    const data = rows.slice(1).map(r => ({
      ply: Number(r[0]),
      values: r.slice(1).map(Number)
    }))

    const width = headers.length
    const height = data.length
    
    // Scene Setup
    scene = new THREE.Scene()
    
    camera = new THREE.PerspectiveCamera(45, container.value.clientWidth / container.value.clientHeight, 0.1, 1000)
    camera.position.set(width * 1.5, height * 1.5, width * 1.5)
    
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(container.value.clientWidth, container.value.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.value.appendChild(renderer.domElement)
    
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    
    // Create Geometry
    const geometry = new THREE.BufferGeometry()
    const vertices = []
    const colors = []
    const indices = []
    
    // Find min/max for coloring
    let minZ = Infinity, maxZ = -Infinity
    data.forEach(row => row.values.forEach(v => {
      if (v < minZ) minZ = v
      if (v > maxZ) maxZ = v
    }))
    
    // Center the plot
    const xOffset = width / 2
    const yOffset = height / 2
    
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const z = data[i].values[j]
        // Three.js: X, Y, Z (where Y is up in standard Slidev scenes, but let's stick to XZ for plane and Y for height)
        vertices.push(j - xOffset, z * props.zScale, i - yOffset) 
        
        // Color based on Z (Blues)
        const t = (z - minZ) / (maxZ - minZ)
        const color = new THREE.Color().setHSL(0.6, 0.8, 0.4 + t * 0.4) 
        colors.push(color.r, color.g, color.b)
      }
    }
    
    for (let i = 0; i < height - 1; i++) {
      for (let j = 0; j < width - 1; j++) {
        const a = i * width + j
        const b = i * width + j + 1
        const c = (i + 1) * width + j
        const d = (i + 1) * width + j + 1
        
        indices.push(a, d, b)
        indices.push(a, c, d)
      }
    }
    
    geometry.setIndex(indices)
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    geometry.computeVertexNormals()
    
    const material = new THREE.MeshPhongMaterial({ 
      vertexColors: true, 
      side: THREE.DoubleSide,
      shininess: 60,
      specular: 0x444444,
      flatShading: false
    })
    
    const mesh = new THREE.Mesh(geometry, material)
    scene.add(mesh)
    
    // Grid Helper
    const grid = new THREE.GridHelper(Math.max(width, height) * 2, 20, 0xcccccc, 0xeeeeee)
    grid.position.y = minZ * props.zScale - 0.1
    scene.add(grid)
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7)
    scene.add(ambientLight)
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(10, 20, 10)
    scene.add(dirLight)
    
    // Center camera
    controls.target.set(0, (maxZ + minZ) / 2 * props.zScale, 0)
    controls.update()
    
    const animate = () => {
      frameId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()
    
    window.addEventListener('resize', onWindowResize)
  } catch (err) {
    console.error("Three.js Init Error:", err)
  }
}

const onWindowResize = () => {
  if (!container.value) return
  camera.aspect = container.value.clientWidth / container.value.clientHeight
  camera.updateProjectionMatrix()
  renderer.setSize(container.value.clientWidth, container.value.clientHeight)
}

onMounted(() => {
  init()
})

onUnmounted(() => {
  cancelAnimationFrame(frameId)
  window.removeEventListener('resize', onWindowResize)
  if (renderer) {
    renderer.dispose()
    if (container.value && renderer.domElement.parentElement === container.value) {
      container.value.removeChild(renderer.domElement)
    }
  }
})
</script>

<template>
  <div class="surf-plot-wrapper w-full h-full flex flex-col bg-slate-50/30 rounded-xl border border-slate-200/50 p-4 shadow-inner">
    <div ref="container" class="flex-grow cursor-move"></div>
  </div>
</template>

<style scoped>
.surf-plot-wrapper {
  min-height: 400px;
}
</style>
