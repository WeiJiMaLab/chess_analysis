<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';

const container = ref<HTMLElement | null>(null);
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let pieces: THREE.Object3D[] = [];
let frameId: number;

const MODELS = [
  'https://raw.githubusercontent.com/scenevr/chess/master/models/pawn.obj',
  'https://raw.githubusercontent.com/scenevr/chess/master/models/king.obj',
  'https://raw.githubusercontent.com/scenevr/chess/master/models/queen.obj',
  'https://raw.githubusercontent.com/scenevr/chess/master/models/knight.obj',
  'https://raw.githubusercontent.com/scenevr/chess/master/models/bishop.obj',
  'https://raw.githubusercontent.com/scenevr/chess/master/models/rook.obj'
];

onMounted(() => {
  if (!container.value) return;

  // 1. Scene Setup
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.set(0, 40, 180); // Lowered camera slightly for better angle
  camera.lookAt(0, 20, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  container.value.appendChild(renderer.domElement);

  // 2. Matte Lighting System
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  
  const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
  sunLight.position.set(100, 200, 100);
  scene.add(sunLight);

  // 3. Subtle Grid
  const grid = new THREE.GridHelper(1000, 50, 0xf1f5f9, 0xf8fafc);
  grid.position.y = -60;
  scene.add(grid);

  // 4. Load Models
  const loader = new OBJLoader();
  const material = new THREE.MeshStandardMaterial({ 
    color: 0xa5b4fc, // Indigo-300 (Light Purple Accent)
    roughness: 0.9,
    metalness: 0.05,
  });

  MODELS.forEach((url) => {
    loader.load(url, (obj) => {
      obj.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          mesh.material = material;
          mesh.geometry.computeVertexNormals();
        }
      });
      
      const box = new THREE.Box3().setFromObject(obj);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 22 / maxDim;
      obj.scale.set(scale, scale, scale);
      
      for (let i = 0; i < 4; i++) {
        const piece = obj.clone();
        resetPiece(piece, true); // initial randomize
        scene.add(piece);
        pieces.push(piece);
      }
    });
  });

  // 5. Animation Loop
  const animate = () => {
    pieces.forEach((p) => {
      p.position.y -= 0.15; // Slow, calm fall
      
      // Fixed tilt with a steady Y-spin
      p.rotation.y += 0.006;

      if (p.position.y < -180) {
        resetPiece(p);
      }
    });

    renderer.render(scene, camera);
    frameId = requestAnimationFrame(animate);
  };

  const resetPiece = (p: THREE.Object3D, initial = false) => {
    p.position.set(
      (Math.random() - 0.5) * 280,
      initial ? (Math.random() * 400 - 100) : 250,
      (Math.random() - 0.5) * 150
    );
    // Orient them mostly right-side up (0 in X, subtle tilt in Z)
    p.rotation.set(0, Math.random() * Math.PI * 2, (Math.random() - 0.5) * 0.4);
  };

  animate();

  window.addEventListener('resize', onResize);
});

onUnmounted(() => {
  cancelAnimationFrame(frameId);
  window.removeEventListener('resize', onResize);
  renderer.dispose();
});

function onResize() {
  if (!container.value) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
</script>

<template>
  <div ref="container" class="three-bg">
    <div class="overlay"></div>
  </div>
</template>

<style scoped>
.three-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

.overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(to bottom, transparent 0%, rgba(255,255,255,0.4) 40%, rgba(255,255,255,1) 90%);
  z-index: 1;
}
</style>
