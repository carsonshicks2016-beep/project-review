import React, { useRef, useEffect, useState } from 'react'
import './RoomCanvas.css'

const ROOM_WIDTH = 500
const ROOM_HEIGHT = 400
const WALL_THICKNESS = 10

const RoomCanvas = ({ placedItems, selectedItem, onSelectItem, onUpdateItem }) => {
  const canvasRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [draggedItem, setDraggedItem] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    drawRoom(ctx)
    drawItems(ctx)
  }, [placedItems, selectedItem])

  const drawRoom = (ctx) => {
    // Clear canvas
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)

    // Draw floor
    ctx.fillStyle = '#f5f5dc'
    ctx.fillRect(WALL_THICKNESS, WALL_THICKNESS, ROOM_WIDTH, ROOM_HEIGHT)

    // Draw grid pattern
    ctx.strokeStyle = '#e0e0d0'
    ctx.lineWidth = 1
    for (let x = WALL_THICKNESS; x <= ROOM_WIDTH + WALL_THICKNESS; x += 25) {
      ctx.beginPath()
      ctx.moveTo(x, WALL_THICKNESS)
      ctx.lineTo(x, ROOM_HEIGHT + WALL_THICKNESS)
      ctx.stroke()
    }
    for (let y = WALL_THICKNESS; y <= ROOM_HEIGHT + WALL_THICKNESS; y += 25) {
      ctx.beginPath()
      ctx.moveTo(WALL_THICKNESS, y)
      ctx.lineTo(ROOM_WIDTH + WALL_THICKNESS, y)
      ctx.stroke()
    }

    // Draw walls
    ctx.fillStyle = '#4a4a4a'
    // Top wall
    ctx.fillRect(0, 0, ROOM_WIDTH + WALL_THICKNESS * 2, WALL_THICKNESS)
    // Bottom wall
    ctx.fillRect(0, ROOM_HEIGHT + WALL_THICKNESS, ROOM_WIDTH + WALL_THICKNESS * 2, WALL_THICKNESS)
    // Left wall
    ctx.fillRect(0, 0, WALL_THICKNESS, ROOM_HEIGHT + WALL_THICKNESS * 2)
    // Right wall (exterior wall with window)
    ctx.fillRect(ROOM_WIDTH + WALL_THICKNESS, 0, WALL_THICKNESS, ROOM_HEIGHT + WALL_THICKNESS * 2)

    // Draw window on right wall
    ctx.fillStyle = '#87CEEB'
    ctx.fillRect(ROOM_WIDTH + WALL_THICKNESS - 5, 150, 15, 100)
    ctx.strokeStyle = '#4a4a4a'
    ctx.lineWidth = 2
    ctx.strokeRect(ROOM_WIDTH + WALL_THICKNESS - 5, 150, 15, 100)

    // Draw door on left wall
    ctx.fillStyle = '#8B4513'
    ctx.fillRect(WALL_THICKNESS - 5, 300, 15, 60)
    ctx.strokeStyle = '#4a4a4a'
    ctx.strokeRect(WALL_THICKNESS - 5, 300, 15, 60)

    // Draw closet area
    ctx.fillStyle = '#d4d4d4'
    ctx.fillRect(WALL_THICKNESS, ROOM_HEIGHT + WALL_THICKNESS - 80, 150, 80)
    ctx.strokeStyle = '#999'
    ctx.lineWidth = 2
    ctx.strokeRect(WALL_THICKNESS, ROOM_HEIGHT + WALL_THICKNESS - 80, 150, 80)
    
    // Closet door line
    ctx.beginPath()
    ctx.moveTo(WALL_THICKNESS, ROOM_HEIGHT + WALL_THICKNESS - 80)
    ctx.lineTo(WALL_THICKNESS + 150, ROOM_HEIGHT + WALL_THICKNESS - 80)
    ctx.stroke()
  }

  const drawItems = (ctx) => {
    placedItems.forEach(item => {
      ctx.save()
      ctx.translate(item.x + getItemWidth(item) / 2, item.y + getItemHeight(item) / 2)
      ctx.rotate((item.rotation * Math.PI) / 180)
      ctx.translate(-getItemWidth(item) / 2, -getItemHeight(item) / 2)

      // Draw item
      ctx.fillStyle = item.color
      ctx.fillRect(0, 0, getItemWidth(item), getItemHeight(item))

      // Draw item border
      ctx.strokeStyle = selectedItem?.id === item.id ? '#667eea' : '#333'
      ctx.lineWidth = selectedItem?.id === item.id ? 3 : 1
      ctx.strokeRect(0, 0, getItemWidth(item), getItemHeight(item))

      // Draw item label
      ctx.fillStyle = '#fff'
      ctx.font = '12px Arial'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(item.type, getItemWidth(item) / 2, getItemHeight(item) / 2)

      ctx.restore()
    })
  }

  const getItemWidth = (item) => {
    const sizes = {
      bed: { twin: 60, full: 75, queen: 90, king: 100 },
      desk: { small: 80, standard: 100, large: 120 },
      chair: { standard: 40 },
      nightstand: { small: 30, standard: 40 },
      dresser: { small: 60, standard: 80, large: 100 },
      wardrobe: { standard: 70 },
      rug: { small: 80, standard: 120, large: 160 },
      lamp: { standard: 25 },
      plant: { small: 20, standard: 30, large: 40 }
    }
    return sizes[item.type]?.[item.size] || 50
  }

  const getItemHeight = (item) => {
    const sizes = {
      bed: { twin: 80, full: 80, queen: 85, king: 90 },
      desk: { small: 40, standard: 50, large: 60 },
      chair: { standard: 40 },
      nightstand: { small: 30, standard: 40 },
      dresser: { small: 30, standard: 35, large: 40 },
      wardrobe: { standard: 80 },
      rug: { small: 60, standard: 80, large: 100 },
      lamp: { standard: 25 },
      plant: { small: 20, standard: 30, large: 40 }
    }
    return sizes[item.type]?.[item.size] || 50
  }

  const handleMouseDown = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // Check if clicking on an item (reverse order to check top items first)
    for (let i = placedItems.length - 1; i >= 0; i--) {
      const item = placedItems[i]
      const itemWidth = getItemWidth(item)
      const itemHeight = getItemHeight(item)
      
      if (x >= item.x && x <= item.x + itemWidth &&
          y >= item.y && y <= item.y + itemHeight) {
        setIsDragging(true)
        setDraggedItem(item)
        setDragOffset({ x: x - item.x, y: y - item.y })
        onSelectItem(item)
        return
      }
    }

    // If not clicking on an item, deselect
    onSelectItem(null)
  }

  const handleMouseMove = (e) => {
    if (!isDragging || !draggedItem) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left - dragOffset.x
    const y = e.clientY - rect.top - dragOffset.y

    // Constrain to room bounds
    const itemWidth = getItemWidth(draggedItem)
    const itemHeight = getItemHeight(draggedItem)
    const constrainedX = Math.max(WALL_THICKNESS, Math.min(x, ROOM_WIDTH - itemWidth))
    const constrainedY = Math.max(WALL_THICKNESS, Math.min(y, ROOM_HEIGHT - itemHeight))

    onUpdateItem(draggedItem.id, { x: constrainedX, y: constrainedY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    setDraggedItem(null)
  }

  return (
    <div className="room-canvas-container">
      <div className="room-info">
        <h3>Bedroom Layout (12' x 10')</h3>
        <p>Drag items to reposition • Click to select • Use panel to customize</p>
      </div>
      <canvas
        ref={canvasRef}
        width={ROOM_WIDTH + WALL_THICKNESS * 2}
        height={ROOM_HEIGHT + WALL_THICKNESS * 2}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="room-canvas"
      />
      <div className="room-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#87CEEB' }}></div>
          <span>Window</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#8B4513' }}></div>
          <span>Door</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#d4d4d4' }}></div>
          <span>Closet</span>
        </div>
      </div>
    </div>
  )
}

export default RoomCanvas
