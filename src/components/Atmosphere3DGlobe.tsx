import { useEffect, useRef, useState, useMemo } from "react";
import * as THREE from "three";
import { Station } from "@/data/skyguard";
import {
  Globe,
  Layers,
  Navigation,
  Play,
  Pause,
  RotateCcw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

interface Atmosphere3DGlobeProps {
  stations: Station[];
  selectedStationId: string;
  onSelectStation?: (stationId: string) => void;
  className?: string;
}

// Convert latitude and longitude to 3D Cartesian coordinates on sphere
function latLonToVector3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}

export function Atmosphere3DGlobe({
  stations,
  selectedStationId,
  onSelectStation,
  className = "",
}: Atmosphere3DGlobeProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [activeLayer, setActiveLayer] = useState<"atmosphere" | "jetstream" | "radar">(
    "atmosphere",
  );
  const [isHovered, setIsHovered] = useState(false);

  // References for Three.js objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const earthGroupRef = useRef<THREE.Group | null>(null);
  const cloudsRef = useRef<THREE.Mesh | null>(null);
  const markersGroupRef = useRef<THREE.Group | null>(null);
  const streamsGroupRef = useRef<THREE.Group | null>(null);
  const targetRotationRef = useRef<{ x: number; y: number } | null>(null);
  const autoRotateRef = useRef(autoRotate);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  const selectedStation = useMemo(() => {
    return stations.find((s) => s.station_id === selectedStationId) || stations[0];
  }, [stations, selectedStationId]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 340;

    // 1. Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
    camera.position.set(0, 5, 24);
    cameraRef.current = camera;

    // 3. Renderer with high performance & smooth anti-aliasing
    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xdbeafe, 1.2);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0x38bdf8, 2.6);
    sunLight.position.set(20, 15, 25);
    scene.add(sunLight);

    const rimLight = new THREE.DirectionalLight(0x818cf8, 1.8);
    rimLight.position.set(-20, -10, -15);
    scene.add(rimLight);

    // 5. Earth Group
    const earthGroup = new THREE.Group();
    scene.add(earthGroup);
    earthGroupRef.current = earthGroup;

    const earthRadius = 7.5;

    // Create procedural high-tech textured globe canvas
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      // Ocean deep gradient
      const oceanGrad = ctx.createLinearGradient(0, 0, 0, 512);
      oceanGrad.addColorStop(0, "#081022");
      oceanGrad.addColorStop(0.5, "#0d1b38");
      oceanGrad.addColorStop(1, "#060b18");
      ctx.fillStyle = oceanGrad;
      ctx.fillRect(0, 0, 1024, 512);

      // Lat-Long atmospheric grid lines
      ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
      ctx.lineWidth = 1;
      for (let lat = 0; lat <= 512; lat += 32) {
        ctx.beginPath();
        ctx.moveTo(0, lat);
        ctx.lineTo(1024, lat);
        ctx.stroke();
      }
      for (let lon = 0; lon <= 1024; lon += 32) {
        ctx.beginPath();
        ctx.moveTo(lon, 0);
        ctx.lineTo(lon, 512);
        ctx.stroke();
      }

      // Procedural continents representation (stylized dots & nodes)
      ctx.fillStyle = "rgba(56, 189, 248, 0.45)";
      for (let i = 0; i < 2800; i++) {
        const x = Math.random() * 1024;
        const y = Math.random() * 512;
        // Concentrate points around India subcontinent roughly: lon ~ 65-95E (690-780px), lat ~ 8-37N (200-280px)
        const inIndiaCluster = x > 670 && x < 800 && y > 180 && y < 310;
        const inEurasiaCluster = x > 500 && x < 860 && y > 100 && y < 340;
        if (inIndiaCluster || inEurasiaCluster || Math.random() < 0.25) {
          ctx.beginPath();
          ctx.arc(x, y, inIndiaCluster ? 1.8 : 1.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    const earthTexture = new THREE.CanvasTexture(canvas);
    earthTexture.wrapS = THREE.RepeatWrapping;
    earthTexture.wrapT = THREE.ClampToEdgeWrapping;

    // Earth Sphere
    const earthGeometry = new THREE.SphereGeometry(earthRadius, 64, 64);
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughness: 0.7,
      metalness: 0.1,
      bumpScale: 0.05,
    });
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    earthGroup.add(earthMesh);

    // Glowing Atmospheric Outer Halo (Fresnel Atmosphere)
    const atmosphereGeometry = new THREE.SphereGeometry(earthRadius * 1.045, 64, 64);
    const atmosphereMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.68 - dot(vNormal, vec3(0, 0, 1.0)), 2.6);
          gl_FragColor = vec4(0.22, 0.74, 0.97, 1.0) * intensity * 1.8;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
    const atmosphereMesh = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    earthGroup.add(atmosphereMesh);

    // Orbiting Atmospheric Cloud Layer
    const cloudCanvas = document.createElement("canvas");
    cloudCanvas.width = 512;
    cloudCanvas.height = 256;
    const cCtx = cloudCanvas.getContext("2d");
    if (cCtx) {
      cCtx.fillStyle = "rgba(0,0,0,0)";
      cCtx.fillRect(0, 0, 512, 256);
      cCtx.fillStyle = "rgba(255, 255, 255, 0.28)";
      for (let c = 0; c < 220; c++) {
        const cx = Math.random() * 512;
        const cy = 60 + Math.random() * 136;
        cCtx.beginPath();
        cCtx.arc(cx, cy, 14 + Math.random() * 24, 0, Math.PI * 2);
        cCtx.fill();
      }
    }
    const cloudTexture = new THREE.CanvasTexture(cloudCanvas);
    const cloudsGeometry = new THREE.SphereGeometry(earthRadius * 1.025, 48, 48);
    const cloudsMaterial = new THREE.MeshStandardMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
    });
    const cloudsMesh = new THREE.Mesh(cloudsGeometry, cloudsMaterial);
    earthGroup.add(cloudsMesh);
    cloudsRef.current = cloudsMesh;

    // 6. Station Pins Group
    const markersGroup = new THREE.Group();
    earthGroup.add(markersGroup);
    markersGroupRef.current = markersGroup;

    // 7. Synoptic Wind Streams Group
    const streamsGroup = new THREE.Group();
    earthGroup.add(streamsGroup);
    streamsGroupRef.current = streamsGroup;

    // Create 3 atmospheric curved jet streams
    const streamColors = [0x38bdf8, 0x34d399, 0x818cf8];
    for (let s = 0; s < 3; s++) {
      const curvePoints: THREE.Vector3[] = [];
      const baseLat = 12 + s * 10;
      for (let lon = 50; lon <= 110; lon += 5) {
        const lat = baseLat + Math.sin(lon * 0.15 + s) * 4;
        curvePoints.push(latLonToVector3(lat, lon, earthRadius * (1.035 + s * 0.01)));
      }
      const curve = new THREE.CatmullRomCurve3(curvePoints);
      const tubeGeom = new THREE.TubeGeometry(curve, 32, 0.04, 6, false);
      const tubeMat = new THREE.MeshBasicMaterial({
        color: streamColors[s] ?? 0x38bdf8,
        transparent: true,
        opacity: 0.6,
      });
      const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
      streamsGroup.add(tubeMesh);
    }

    // Default orientation: point towards India (Lat ~ 20N, Lon ~ 80E)
    earthGroup.rotation.y = -Math.PI * 0.42;
    earthGroup.rotation.x = 0.28;

    // Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Mouse drag controls
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging || !earthGroup) return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      earthGroup.rotation.y += deltaX * 0.005;
      earthGroup.rotation.x = Math.max(-0.8, Math.min(0.8, earthGroup.rotation.x + deltaY * 0.005));
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const domEl = renderer.domElement;
    domEl.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    // Wheel zoom
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.position.z = Math.max(14, Math.min(36, camera.position.z + e.deltaY * 0.015));
    };
    domEl.addEventListener("wheel", onWheel, { passive: false });

    // Animation Loop
    let animId: number;
    let clock = 0;
    const animate = () => {
      clock += 0.016;

      // Rotate clouds slightly faster than earth
      if (cloudsRef.current) {
        cloudsRef.current.rotation.y += 0.0007;
      }

      // Smooth target rotation if aiming for a station
      if (targetRotationRef.current && earthGroup) {
        const { x: tX, y: tY } = targetRotationRef.current;
        earthGroup.rotation.x += (tX - earthGroup.rotation.x) * 0.05;
        earthGroup.rotation.y += (tY - earthGroup.rotation.y) * 0.05;
        if (
          Math.abs(tX - earthGroup.rotation.x) < 0.001 &&
          Math.abs(tY - earthGroup.rotation.y) < 0.001
        ) {
          targetRotationRef.current = null;
        }
      } else if (autoRotateRef.current && !isDragging && earthGroup) {
        earthGroup.rotation.y += 0.0012;
      }

      // Animate stream glow pulse
      if (streamsGroupRef.current) {
        streamsGroupRef.current.children.forEach((c, idx) => {
          const mesh = c as { material?: { opacity?: number } };
          if (mesh.material && typeof mesh.material.opacity === "number") {
            mesh.material.opacity = 0.4 + 0.3 * Math.sin(clock * 2 + idx);
          }
        });
      }

      renderer.render(scene, camera);
      animId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      domEl.removeEventListener("mousedown", onMouseDown);
      domEl.removeEventListener("wheel", onWheel);
      if (container.contains(domEl)) {
        container.removeChild(domEl);
      }
      renderer.dispose();
    };
  }, []);

  // Update Station Marker pins whenever stations or selectedStationId change
  useEffect(() => {
    const markersGroup = markersGroupRef.current;
    if (!markersGroup) return;

    // Clear previous markers
    while (markersGroup.children.length > 0) {
      const obj = markersGroup.children[0];
      if (obj) markersGroup.remove(obj);
      else break;
    }

    const earthRadius = 7.5;

    stations.forEach((stn) => {
      const isSelected = stn.station_id === selectedStationId;
      const isCritical = stn.status === "CRITICAL" || stn.status === "ANOMALY";
      const markerColor = isCritical ? 0xf43f5e : isSelected ? 0x38bdf8 : 0x34d399;

      const pos = latLonToVector3(stn.latitude, stn.longitude, earthRadius * 1.01);

      // Marker Dot
      const dotGeom = new THREE.SphereGeometry(isSelected ? 0.22 : 0.12, 16, 16);
      const dotMat = new THREE.MeshBasicMaterial({ color: markerColor });
      const dotMesh = new THREE.Mesh(dotGeom, dotMat);
      dotMesh.position.copy(pos);
      markersGroup.add(dotMesh);

      // For selected station, add a beacon antenna beam and pulsing halo ring
      if (isSelected) {
        const normal = pos.clone().normalize();
        const beamLength = 1.4;
        const beamEnd = pos.clone().add(normal.clone().multiplyScalar(beamLength));

        const lineGeom = new THREE.BufferGeometry().setFromPoints([pos, beamEnd]);
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.9,
          linewidth: 2,
        });
        const line = new THREE.Line(lineGeom, lineMat);
        markersGroup.add(line);

        // Halo Ring
        const ringGeom = new THREE.RingGeometry(0.28, 0.42, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0x38bdf8,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.75,
        });
        const ringMesh = new THREE.Mesh(ringGeom, ringMat);
        ringMesh.position.copy(beamEnd);
        ringMesh.lookAt(beamEnd.clone().add(normal));
        markersGroup.add(ringMesh);
      }
    });

    // Smoothly focus on selected station coordinates
    if (selectedStation && earthGroupRef.current) {
      const latRad = selectedStation.latitude * (Math.PI / 180);
      const lonRad = (selectedStation.longitude + 180) * (Math.PI / 180);

      // Target rotation angles to face the camera
      const targetY = -lonRad + Math.PI * 0.5;
      const targetX = latRad * 0.5;
      targetRotationRef.current = { x: targetX, y: targetY };
    }
  }, [stations, selectedStationId, selectedStation]);

  return (
    <div
      className={`relative rounded-2xl overflow-hidden glass-panel border border-white/10 ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* 3D WebGL Canvas Container */}
      <div ref={containerRef} className="w-full h-80 cursor-grab active:cursor-grabbing" />

      {/* Atmospheric Vignette & Lighting Rim */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_45%,rgba(8,13,25,0.75)_100%)]" />

      {/* Top Floating Telemetry Overlay */}
      <div className="absolute top-3.5 left-3.5 right-3.5 flex items-center justify-between pointer-events-none z-10">
        <div className="flex items-center gap-2 pointer-events-auto bg-slate-950/70 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 shadow-lg">
          <Globe className="h-3.5 w-3.5 text-sky-400 animate-spin-slow" />
          <span className="text-xs font-semibold text-white tracking-wide">
            Atmospheric 3D Node
          </span>
          <span className="h-2.5 w-px bg-white/20" />
          <span className="text-[11px] font-mono text-sky-300">
            {selectedStation?.station_name || "Vijayawada"} ({selectedStation?.latitude.toFixed(1)}
            °N, {selectedStation?.longitude.toFixed(1)}°E)
          </span>
        </div>

        {/* 3D Orbit Control Pills */}
        <div className="flex items-center gap-1.5 pointer-events-auto">
          <button
            type="button"
            onClick={() => setAutoRotate(!autoRotate)}
            className={`p-1.5 rounded-lg border text-xs transition-all cursor-pointer backdrop-blur-md ${
              autoRotate
                ? "bg-sky-500/20 text-sky-300 border-sky-400/35"
                : "bg-slate-900/70 text-slate-300 border-white/10 hover:bg-white/10"
            }`}
            title={autoRotate ? "Pause Auto Orbit" : "Resume Auto Orbit"}
          >
            {autoRotate ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>

          <button
            type="button"
            onClick={() => {
              if (earthGroupRef.current) {
                targetRotationRef.current = { x: 0.28, y: -Math.PI * 0.42 };
              }
            }}
            className="p-1.5 rounded-lg border border-white/10 bg-slate-900/70 text-slate-300 hover:text-white hover:bg-white/10 text-xs transition-all cursor-pointer backdrop-blur-md"
            title="Reset India Centered View"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Bottom Floating Station Status Pill */}
      <div className="absolute bottom-3 left-3.5 right-3.5 flex items-center justify-between text-xs pointer-events-none z-10">
        <div className="flex items-center gap-2 bg-slate-950/75 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10">
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
            </span>
            <span className="text-[11px] font-mono font-bold text-slate-200 uppercase">
              {selectedStation?.station_id}
            </span>
          </div>
          <span className="text-white/20">•</span>
          <span className="text-[11px] font-mono text-emerald-300">
            {selectedStation?.temperature !== null
              ? `${selectedStation?.temperature?.toFixed(1)}°C`
              : "—"}
          </span>
          <span className="text-white/20">•</span>
          <span className="text-[11px] font-mono text-sky-300">
            {selectedStation?.pressure !== null
              ? `${selectedStation?.pressure?.toFixed(1)} hPa`
              : "—"}
          </span>
        </div>

        <div className="text-[10px] font-mono text-slate-400 bg-slate-950/65 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/8 hidden sm:block">
          Interactive 3D • Drag to Orbit • Scroll to Zoom
        </div>
      </div>
    </div>
  );
}
