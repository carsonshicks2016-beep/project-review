import React, { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows, Text } from '@react-three/drei'
import * as THREE from 'three'

const ROOM_WIDTH = 12
const ROOM_HEIGHT = 10
const ROOM_CEILING = 9

// Wall component with realistic material
const Wall = ({ position, rotation, dimensions, color = '#f5f5f5' }) => {
  const [width, height, depth] = dimensions
  return (
    <mesh position={position} rotation={rotation}>
      <boxGeometry args={[width, height, depth]} />
      <meshStandardMaterial 
        color={color}
        roughness={0.8}
        metalness={0.1}
      />
    </mesh>
  )
}

// Floor with realistic wood texture
const Floor = () => {
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 512
    canvas.height = 512
    const ctx = canvas.getContext('2d')
    
    // Create wood plank pattern
    ctx.fillStyle = '#8B7355'
    ctx.fillRect(0, 0, 512, 512)
    
    // Add wood grain lines
    ctx.strokeStyle = '#6B5344'
    ctx.lineWidth = 2
    for (let i = 0; i < 50; i++) {
      ctx.beginPath()
      ctx.moveTo(0, Math.random() * 512)
      ctx.bezierCurveTo(
        128, Math.random() * 512,
        384, Math.random() * 512,
        512, Math.random() * 512
      )
      ctx.stroke()
    }
    
    // Add plank lines
    ctx.strokeStyle = '#5B4334'
    ctx.lineWidth = 3
    for (let i = 0; i < 8; i++) {
      ctx.beginPath()
      ctx.moveTo(0, i * 64)
      ctx.lineTo(512, i * 64)
      ctx.stroke()
    }
    
    const texture = new THREE.CanvasTexture(canvas)
    texture.wrapS = THREE.RepeatWrapping
    texture.wrapT = THREE.RepeatWrapping
    texture.repeat.set(4, 4)
    return texture
  }, [])

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[ROOM_WIDTH / 2, 0, ROOM_HEIGHT / 2]}>
      <planeGeometry args={[ROOM_WIDTH, ROOM_HEIGHT]} />
      <meshStandardMaterial 
        map={texture}
        roughness={0.6}
        metalness={0.05}
      />
    </mesh>
  )
}

// Window component with glass effect
const Window = ({ position }) => {
  return (
    <group position={position}>
      {/* Window frame */}
      <mesh position={[0, 0, -0.1]}>
        <boxGeometry args={[4, 5, 0.2]} />
        <meshStandardMaterial color="#4a4a4a" roughness={0.3} metalness={0.7} />
      </mesh>
      {/* Glass */}
      <mesh>
        <planeGeometry args={[3.5, 4.5]} />
        <meshPhysicalMaterial 
          color="#87CEEB"
          transparent
          opacity={0.3}
          roughness={0.1}
          metalness={0.1}
          transmission={0.9}
        />
      </mesh>
      {/* Window dividers */}
      <mesh position={[0, 0, 0.1]}>
        <boxGeometry args={[0.1, 4.5, 0.1]} />
        <meshStandardMaterial color="#4a4a4a" />
      </mesh>
      <mesh position={[0, 0, 0.1]}>
        <boxGeometry args={[3.5, 0.1, 0.1]} />
        <meshStandardMaterial color="#4a4a4a" />
      </mesh>
    </group>
  )
}

// Door component
const Door = ({ position }) => {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[3, 7, 0.2]} />
        <meshStandardMaterial color="#8B4513" roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Door handle */}
      <mesh position={[1, 0, 0.2]}>
        <sphereGeometry args={[0.15]} />
        <meshStandardMaterial color="#FFD700" roughness={0.2} metalness={0.9} />
      </mesh>
    </group>
  )
}

// Closet component
const Closet = ({ position }) => {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[5, 7, 2]} />
        <meshStandardMaterial color="#d4d4d4" roughness={0.6} metalness={0.1} />
      </mesh>
      {/* Closet doors */}
      <mesh position={[0, 0, 1.1]}>
        <boxGeometry args={[4.8, 6.8, 0.1]} />
        <meshStandardMaterial color="#e8e8e8" roughness={0.5} metalness={0.1} />
      </mesh>
      {/* Door handles */}
      <mesh position={[-1, 0, 1.2]}>
        <cylinderGeometry args={[0.1, 0.1, 0.3]} />
        <meshStandardMaterial color="#C0C0C0" roughness={0.2} metalness={0.8} />
      </mesh>
      <mesh position={[1, 0, 1.2]}>
        <cylinderGeometry args={[0.1, 0.1, 0.3]} />
        <meshStandardMaterial color="#C0C0C0" roughness={0.2} metalness={0.8} />
      </mesh>
    </group>
  )
}

// 3D Furniture components
const Bed3D = ({ position, color, size, rotation }) => {
  const sizes = {
    twin: { width: 4, height: 2, length: 7 },
    full: { width: 5, height: 2, length: 7 },
    queen: { width: 6, height: 2, length: 7 },
    king: { width: 7, height: 2, length: 7 }
  }
  const { width, height, length } = sizes[size] || sizes.queen

  return (
    <group position={position} rotation={[0, rotation, 0]}>
      {/* Bed frame */}
      <mesh position={[0, height / 2, 0]}>
        <boxGeometry args={[width, height, length]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Mattress */}
      <mesh position={[0, height + 0.5, 0]}>
        <boxGeometry args={[width - 0.2, 0.8, length - 0.2]} />
        <meshStandardMaterial color="#ffffff" roughness={0.9} metalness={0} />
      </mesh>
      {/* Pillow */}
      <mesh position={[0, height + 1, -length / 3]}>
        <boxGeometry args={[width - 1, 0.4, 1.5]} />
        <meshStandardMaterial color="#f0f0f0" roughness={0.9} metalness={0} />
      </mesh>
      {/* Headboard */}
      <mesh position={[0, height + 1, -length / 2]}>
        <boxGeometry args={[width, 3, 0.2]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.1} />
      </mesh>
    </group>
  )
}

const Desk3D = ({ position, color, size, rotation }) => {
  const sizes = {
    small: { width: 3, height: 2.5, depth: 1.5 },
    standard: { width: 4, height: 2.5, depth: 2 },
    large: { width: 5, height: 2.5, depth: 2.5 }
  }
  const { width, height, depth } = sizes[size] || sizes.standard

  return (
    <group position={position} rotation={[0, rotation, 0]}>
      {/* Desk top */}
      <mesh position={[0, height, 0]}>
        <boxGeometry args={[width, 0.2, depth]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.1} />
      </mesh>
      {/* Legs */}
      {[[-width/2 + 0.2, -depth/2 + 0.2], [width/2 - 0.2, -depth/2 + 0.2], 
        [-width/2 + 0.2, depth/2 - 0.2], [width/2 - 0.2, depth/2 - 0.2]].map(([x, z], i) => (
        <mesh key={i} position={[x, height / 2, z]}>
          <boxGeometry args={[0.15, height, 0.15]} />
          <meshStandardMaterial color="#4a4a4a" roughness={0.5} metalness={0.3} />
        </mesh>
      ))}
    </group>
  )
}

const Chair3D = ({ position, color, rotation }) => {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      {/* Seat */}
      <mesh position={[0, 1.5, 0]}>
        <boxGeometry args={[1.5, 0.2, 1.5]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.1} />
      </mesh>
      {/* Back */}
      <mesh position={[0, 2.5, -0.65]}>
        <boxGeometry args={[1.5, 2, 0.2]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.1} />
      </mesh>
      {/* Legs */}
      {[[-0.6, -0.6], [0.6, -0.6], [-0.6, 0.6], [0.6, 0.6]].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.75, z]}>
          <cylinderGeometry args={[0.08, 0.08, 1.5]} />
          <meshStandardMaterial color="#4a4a4a" roughness={0.5} metalness={0.3} />
        </mesh>
      ))}
    </group>
  )
}

const Nightstand3D = ({ position, color, size, rotation }) => {
  const sizes = {
    small: { width: 1.5, height: 2, depth: 1.5 },
    standard: { width: 2, height: 2.5, depth: 1.5 }
  }
  const { width, height, depth } = sizes[size] || sizes.standard

  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Drawer */}
      <mesh position={[0, 0.5, depth / 2 + 0.05]}>
        <boxGeometry args={[width - 0.2, 0.8, 0.1]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.1} />
      </mesh>
      {/* Handle */}
      <mesh position={[0, 0.5, depth / 2 + 0.15]}>
        <cylinderGeometry args={[0.08, 0.08, 0.3]} rotation={[Math.PI / 2, 0, 0]} />
        <meshStandardMaterial color="#C0C0C0" roughness={0.2} metalness={0.8} />
      </mesh>
    </group>
  )
}

const Dresser3D = ({ position, color, size, rotation }) => {
  const sizes = {
    small: { width: 3, height: 3, depth: 1.5 },
    standard: { width: 4, height: 3.5, depth: 1.8 },
    large: { width: 5, height: 4, depth: 2 }
  }
  const { width, height, depth } = sizes[size] || sizes.standard

  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Drawers */}
      {[0, 1, 2].map((i) => (
        <group key={i}>
          <mesh position={[0, -height / 2 + 0.8 + i * 1.1, depth / 2 + 0.05]}>
            <boxGeometry args={[width - 0.2, 0.9, 0.1]} />
            <meshStandardMaterial color={color} roughness={0.6} metalness={0.1} />
          </mesh>
          <mesh position={[0, -height / 2 + 0.8 + i * 1.1, depth / 2 + 0.15]}>
            <cylinderGeometry args={[0.08, 0.08, 0.3]} rotation={[Math.PI / 2, 0, 0]} />
            <meshStandardMaterial color="#C0C0C0" roughness={0.2} metalness={0.8} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

const Wardrobe3D = ({ position, color, rotation }) => {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh>
        <boxGeometry args={[3, 7, 2]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Doors */}
      <mesh position={[0, 0, 1.1]}>
        <boxGeometry args={[2.8, 6.8, 0.1]} />
        <meshStandardMaterial color="#e8e8e8" roughness={0.5} metalness={0.1} />
      </mesh>
      {/* Handles */}
      <mesh position={[-0.7, 0, 1.2]}>
        <cylinderGeometry args={[0.08, 0.08, 0.3]} rotation={[Math.PI / 2, 0, 0]} />
        <meshStandardMaterial color="#C0C0C0" roughness={0.2} metalness={0.8} />
      </mesh>
      <mesh position={[0.7, 0, 1.2]}>
        <cylinderGeometry args={[0.08, 0.08, 0.3]} rotation={[Math.PI / 2, 0, 0]} />
        <meshStandardMaterial color="#C0C0C0" roughness={0.2} metalness={0.8} />
      </mesh>
    </group>
  )
}

const Rug3D = ({ position, color, size, rotation }) => {
  const sizes = {
    small: { width: 5, length: 7 },
    standard: { width: 7, length: 9 },
    large: { width: 9, length: 11 }
  }
  const { width, length } = sizes[size] || sizes.standard

  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh position={[0, 0.05, 0]}>
        <boxGeometry args={[width, 0.1, length]} />
        <meshStandardMaterial color={color} roughness={0.9} metalness={0} />
      </mesh>
    </group>
  )
}

const Lamp3D = ({ position, color, rotation }) => {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      {/* Base */}
      <mesh position={[0, 0.2, 0]}>
        <cylinderGeometry args={[0.3, 0.4, 0.4]} />
        <meshStandardMaterial color="#4a4a4a" roughness={0.3} metalness={0.7} />
      </mesh>
      {/* Stem */}
      <mesh position={[0, 1, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 1.6]} />
        <meshStandardMaterial color="#4a4a4a" roughness={0.3} metalness={0.7} />
      </mesh>
      {/* Shade */}
      <mesh position={[0, 2, 0]}>
        <coneGeometry args={[0.5, 0.8, 32, 1, true]} />
        <meshStandardMaterial color={color} roughness={0.8} metalness={0.1} side={THREE.DoubleSide} />
      </mesh>
      {/* Light bulb */}
      <pointLight position={[0, 1.8, 0]} intensity={0.5} distance={5} color="#fff5e6" />
    </group>
  )
}

const Plant3D = ({ position, color, size, rotation }) => {
  const sizes = {
    small: { height: 1.5, pot: 0.8 },
    standard: { height: 2.5, pot: 1 },
    large: { height: 3.5, pot: 1.2 }
  }
  const { height, pot } = sizes[size] || sizes.standard

  return (
    <group position={position} rotation={[0, rotation, 0]}>
      {/* Pot */}
      <mesh position={[0, pot / 2, 0]}>
        <cylinderGeometry args={[pot, pot * 0.8, pot, 32]} />
        <meshStandardMaterial color="#8B4513" roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Plant stems */}
      {[0, 1, 2, 3, 4].map((i) => (
        <mesh key={i} position={[
          Math.sin(i * 1.26) * 0.3,
          pot + height / 2,
          Math.cos(i * 1.26) * 0.3
        ]} rotation={[0.2, i * 0.5, 0]}>
          <cylinderGeometry args={[0.05, 0.08, height]} />
          <meshStandardMaterial color={color} roughness={0.6} metalness={0} />
        </mesh>
      ))}
      {/* Leaves */}
      {[0, 1, 2, 3, 4].map((i) => (
        <mesh key={`leaf-${i}`} position={[
          Math.sin(i * 1.26) * 0.5,
          pot + height,
          Math.cos(i * 1.26) * 0.5
        ]} rotation={[0, i * 0.5, 0.5]}>
          <sphereGeometry args={[0.3, 8, 8]} />
          <meshStandardMaterial color={color} roughness={0.6} metalness={0} />
        </mesh>
      ))}
    </group>
  )
}

// Main room component
const Room = ({ placedItems, selectedItem, onSelectItem }) => {
  const getFurnitureComponent = (item) => {
    const props = {
      key: item.id,
      position: [item.x / 50, 0, item.y / 50],
      color: item.color,
      size: item.size,
      rotation: (item.rotation * Math.PI) / 180,
      onClick: (e) => {
        e.stopPropagation()
        onSelectItem(item)
      }
    }

    switch (item.type) {
      case 'bed': return <Bed3D {...props} />
      case 'desk': return <Desk3D {...props} />
      case 'chair': return <Chair3D {...props} />
      case 'nightstand': return <Nightstand3D {...props} />
      case 'dresser': return <Dresser3D {...props} />
      case 'wardrobe': return <Wardrobe3D {...props} />
      case 'rug': return <Rug3D {...props} />
      case 'lamp': return <Lamp3D {...props} />
      case 'plant': return <Plant3D {...props} />
      default: return null
    }
  }

  return (
    <group>
      {/* Floor */}
      <Floor />
      
      {/* Walls */}
      <Wall position={[0, ROOM_CEILING / 2, ROOM_HEIGHT / 2]} rotation={[0, 0, 0]} dimensions={[0.2, ROOM_CEILING, ROOM_HEIGHT]} />
      <Wall position={[ROOM_WIDTH, ROOM_CEILING / 2, ROOM_HEIGHT / 2]} rotation={[0, 0, 0]} dimensions={[0.2, ROOM_CEILING, ROOM_HEIGHT]} />
      <Wall position={[ROOM_WIDTH / 2, ROOM_CEILING / 2, 0]} rotation={[0, Math.PI / 2, 0]} dimensions={[0.2, ROOM_CEILING, ROOM_WIDTH]} />
      <Wall position={[ROOM_WIDTH / 2, ROOM_CEILING / 2, ROOM_HEIGHT]} rotation={[0, Math.PI / 2, 0]} dimensions={[0.2, ROOM_CEILING, ROOM_WIDTH]} />
      
      {/* Ceiling */}
      <mesh position={[ROOM_WIDTH / 2, ROOM_CEILING, ROOM_HEIGHT / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[ROOM_WIDTH, ROOM_HEIGHT]} />
        <meshStandardMaterial color="#f5f5f5" roughness={0.9} metalness={0} />
      </mesh>
      
      {/* Window on right wall */}
      <Window position={[ROOM_WIDTH, 5, ROOM_HEIGHT / 2]} rotation={[0, -Math.PI / 2, 0]} />
      
      {/* Door on left wall */}
      <Door position={[0, 3.5, ROOM_HEIGHT - 3]} rotation={[0, Math.PI / 2, 0]} />
      
      {/* Closet */}
      <Closet position={[2.5, 3.5, ROOM_HEIGHT - 1]} />
      
      {/* Furniture */}
      {placedItems.map(item => getFurnitureComponent(item))}
    </group>
  )
}

const Room3D = ({ placedItems, selectedItem, onSelectItem }) => {
  return (
    <div className="room-3d-container">
      <div className="room-info">
        <h3>🏠 3D Bedroom View</h3>
        <p>Drag to rotate • Scroll to zoom • Click items to select</p>
      </div>
      <div className="canvas-wrapper">
        <Canvas
          camera={{ position: [15, 12, 15], fov: 50 }}
          shadows
          gl={{ antialias: true, alpha: true }}
        >
          <color attach="background" args={['#1a1a2e']} />
          
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={1}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <pointLight position={[ROOM_WIDTH - 1, 5, ROOM_HEIGHT / 2]} intensity={0.5} color="#fff5e6" />
          
          {/* Room */}
          <Room placedItems={placedItems} selectedItem={selectedItem} onSelectItem={onSelectItem} />
          
          {/* Controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={5}
            maxDistance={30}
            maxPolarAngle={Math.PI / 2}
          />
          
          {/* Environment for realistic reflections */}
          <Environment preset="apartment" />
          
          {/* Contact shadows for realism */}
          <ContactShadows
            position={[0, 0, 0]}
            opacity={0.3}
            scale={20}
            blur={2}
            far={10}
            resolution={256}
            color="#000000"
          />
        </Canvas>
      </div>
    </div>
  )
}

export default Room3D
