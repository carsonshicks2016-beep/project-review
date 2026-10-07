import React from 'react'
import { Trash2, RotateCw } from 'lucide-react'
import './CustomizationPanel.css'

const CustomizationPanel = ({ item, onUpdateItem, onDeleteItem }) => {
  const colorOptions = [
    { name: 'Oak', value: '#8B4513' },
    { name: 'Walnut', value: '#654321' },
    { name: 'Cherry', value: '#A0522D' },
    { name: 'Maple', value: '#D2691E' },
    { name: 'White', value: '#FFFFFF' },
    { name: 'Black', value: '#2F4F4F' },
    { name: 'Gray', value: '#808080' },
    { name: 'Blue', value: '#4169E1' },
    { name: 'Green', value: '#228B22' },
    { name: 'Gold', value: '#FFD700' },
  ]

  const getSizeOptions = (type) => {
    const sizeMap = {
      bed: [
        { name: 'Twin', value: 'twin' },
        { name: 'Full', value: 'full' },
        { name: 'Queen', value: 'queen' },
        { name: 'King', value: 'king' }
      ],
      desk: [
        { name: 'Small', value: 'small' },
        { name: 'Standard', value: 'standard' },
        { name: 'Large', value: 'large' }
      ],
      dresser: [
        { name: 'Small', value: 'small' },
        { name: 'Standard', value: 'standard' },
        { name: 'Large', value: 'large' }
      ],
      nightstand: [
        { name: 'Small', value: 'small' },
        { name: 'Standard', value: 'standard' }
      ],
      rug: [
        { name: 'Small (5x7)', value: 'small' },
        { name: 'Standard (8x10)', value: 'standard' },
        { name: 'Large (9x12)', value: 'large' }
      ],
      plant: [
        { name: 'Small', value: 'small' },
        { name: 'Standard', value: 'standard' },
        { name: 'Large', value: 'large' }
      ]
    }
    return sizeMap[type] || [{ name: 'Standard', value: 'standard' }]
  }

  const handleRotate = () => {
    const newRotation = (item.rotation + 90) % 360
    onUpdateItem(item.id, { rotation: newRotation })
  }

  const handleDelete = () => {
    if (confirm('Are you sure you want to remove this item?')) {
      onDeleteItem(item.id)
    }
  }

  return (
    <div className="customization-panel">
      <h2>🎨 Customize {item.type}</h2>
      
      <div className="customization-section">
        <h3>Color / Material</h3>
        <div className="color-grid">
          {colorOptions.map(color => (
            <button
              key={color.value}
              onClick={() => onUpdateItem(item.id, { color: color.value })}
              className={`color-btn ${item.color === color.value ? 'active' : ''}`}
              style={{ backgroundColor: color.value }}
              title={color.name}
            />
          ))}
        </div>
      </div>

      <div className="customization-section">
        <h3>Size</h3>
        <div className="size-options">
          {getSizeOptions(item.type).map(size => (
            <button
              key={size.value}
              onClick={() => onUpdateItem(item.id, { size: size.value })}
              className={`size-btn ${item.size === size.value ? 'active' : ''}`}
            >
              {size.name}
            </button>
          ))}
        </div>
      </div>

      <div className="customization-section">
        <h3>Rotation</h3>
        <div className="rotation-controls">
          <button onClick={handleRotate} className="rotate-btn">
            <RotateCw size={18} />
            Rotate 90°
          </button>
          <span className="rotation-value">{item.rotation}°</span>
        </div>
      </div>

      <div className="customization-section">
        <h3>Position</h3>
        <div className="position-controls">
          <div className="position-input">
            <label>X:</label>
            <input
              type="number"
              value={Math.round(item.x)}
              onChange={(e) => onUpdateItem(item.id, { x: parseInt(e.target.value) || 0 })}
              min="10"
              max="490"
            />
          </div>
          <div className="position-input">
            <label>Y:</label>
            <input
              type="number"
              value={Math.round(item.y)}
              onChange={(e) => onUpdateItem(item.id, { y: parseInt(e.target.value) || 0 })}
              min="10"
              max="390"
            />
          </div>
        </div>
      </div>

      <button onClick={handleDelete} className="delete-btn">
        <Trash2 size={18} />
        Remove Item
      </button>
    </div>
  )
}

export default CustomizationPanel
