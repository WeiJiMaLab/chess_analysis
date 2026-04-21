<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';

const container = ref<HTMLElement | null>(null);
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let pieces: { obj: THREE.Object3D, rotSpeed: THREE.Vector3, fallSpeed: number }[] = [];
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
  scene.background = new THREE.Color(0x020617);
  
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.set(0, 40, 180);
  camera.lookAt(0, 20, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.value.appendChild(renderer.domElement);

  // 2. EXTREMELY BRIGHT LIGHT SYSTEM
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.1); 
  scene.add(ambientLight);
  
  // Ultra Key (Indigo)
  const keyLight = new THREE.DirectionalLight(0x6366f1, 6.0);
  keyLight.position.set(100, 150, 50);
  scene.add(keyLight);

  // Ultra Fill (Red)
  const fillLight = new THREE.DirectionalLight(0xef4444, 5.0);
  fillLight.position.set(-150, -100, 50);
  scene.add(fillLight);

  // SUPER BACKLIGHT (Drives the SSS effect)
  const backLight = new THREE.PointLight(0xffffff, 10.0, 1000);
  backLight.position.set(0, 50, -250);
  scene.add(backLight);

  // 3. Subtle Dark Grid
  const grid = new THREE.GridHelper(1000, 50, 0x1e293b, 0x0f172a);
  grid.position.y = -60;
  scene.add(grid);

  // 4. Load Models
  const loader = new OBJLoader();
  
  const material = new THREE.MeshPhysicalMaterial({ 
    color: 0xffffff,
    metalness: 0.1,
    roughness: 0.25,
    transmission: 0.65,
    thickness: 8.0, // Increased thickness for more scattering volume
    ior: 1.5,
    attenuationColor: 0xa5b4fc,
    attenuationDistance: 0.8,
    transparent: true,
    opacity: 0.98
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
      
      for (let i = 0; i < 5; i++) {
        const piece = obj.clone();
        const rotSpeed = new THREE.Vector3(
          (Math.random() - 0.5) * 0.01,
          (Math.random() - 0.5) * 0.015,
          (Math.random() - 0.5) * 0.01
        );
        const fallSpeed = 0.12 + Math.random() * 0.1;
        resetPiece(piece, true);
        scene.add(piece);
        pieces.push({ obj: piece, rotSpeed, fallSpeed });
      }
    });
  });

  // 5. Animation Loop
  const animate = () => {
    pieces.forEach((p) => {
      p.obj.position.y -= p.fallSpeed;
      p.obj.rotation.x += p.rotSpeed.x;
      p.obj.rotation.y += p.rotSpeed.y;
      p.obj.rotation.z += p.rotSpeed.z;

      if (p.obj.position.y < -180) {
        resetPiece(p.obj);
      }
    });

    renderer.render(scene, camera);
    frameId = requestAnimationFrame(animate);
  };

  const resetPiece = (p: THREE.Object3D, initial = false) => {
    p.position.set(
      (Math.random() - 0.5) * 350,
      initial ? (Math.random() * 500 - 100) : 300,
      (Math.random() - 0.5) * 150
    );
    p.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
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
  /* Brighter atmosphere to match intense lights */
  background: 
    radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.25) 0%, transparent 60%),
    radial-gradient(circle at -10% 110%, rgba(239, 68, 68, 0.3) 0%, transparent 70%),
    linear-gradient(to bottom, transparent 0%, rgba(2, 6, 23, 0.6) 50%, rgba(2, 6, 23, 1) 95%);
  z-index: 1;
}
</style>
