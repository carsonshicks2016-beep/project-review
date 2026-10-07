import React, { useState, useCallback } from 'react'
import { Save, RotateCcw, Trash2, Palette, Maximize2, Box } from 'lucide-react'
import Room3D from './components/Room3D'
import ProductCatalog from './components/ProductCatalog'
import CustomizationPanel from './components/CustomizationPanel'
import './App.css'

function App() {
  const [placedItems, setPlacedItems] = useState([
    { id: 1, type: 'bed', x: 100, y: 100, rotation: 0, color: '#8B4513', size: 'queen' },
    { id: 2, type: 'desk', x: 350, y: 50, rotation: 0, color: '#D2691E', size: 'standard' },
  ])
  const [selectedItem, setSelectedItem] = useState(null)
  const [catalogOpen, setCatalogOpen] = useState(true)

  const handleAddItem = useCallback((item) => {
    const newItem = {
      id: Date.now(),
      type: item.type,
      x: 200,
      y: 150,
      rotation: 0,
      color: item.defaultColor,
      size: item.defaultSize,
      ...item
    }
    setPlacedItems(prev => [...prev, newItem])
    setSelectedItem(newItem)
  }, [])

  const handleUpdateItem = useCallback((id, updates) => {
    setPlacedItems(prev => prev.map(item => 
      item.id === id ? { ...item, ...updates } : item
    ))
    if (selectedItem?.id === id) {
      setSelectedItem(prev => ({ ...prev, ...updates }))
    }
  }, [selectedItem])

  const handleDeleteItem = useCallback((id) => {
    setPlacedItems(prev => prev.filter(item => item.id !== id))
    if (selectedItem?.id === id) {
      setSelectedItem(null)
    }
  }, [selectedItem])

  const handleClearAll = useCallback(() => {
    setPlacedItems([])
    setSelectedItem(null)
  }, [])

  const handleSave = useCallback(() => {
    const design = { placedItems }
    localStorage.setItem('roomDesign', JSON.stringify(design))
    alert('Design saved successfully!')
  }, [placedItems])

  const handleLoad = useCallback(() => {
    const saved = localStorage.getItem('roomDesign')
    if (saved) {
      const design = JSON.parse(saved)
      setPlacedItems(design.placedItems || [])
      setSelectedItem(null)
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏠 Interactive Room Planner</h1>
        <p className="subtitle">Customize your bedroom with real-world products</p>
      </header>

      <div className="app-content">
        <div className="main-panel">
          <Room3D 
            placedItems={placedItems}
            selectedItem={selectedItem}
            onSelectItem={setSelectedItem}
          />
          
          <div className="toolbar">
            <button onClick={handleSave} className="btn btn-primary">
              <Save size={18} />
              Save Design
            </button>
            <button onClick={handleLoad} className="btn btn-secondary">
              <RotateCcw size={18} />
              Load Design
            </button>
            <button onClick={handleClearAll} className="btn btn-danger">
              <Trash2 size={18} />
              Clear All
            </button>
            <button 
              onClick={() => setCatalogOpen(!catalogOpen)} 
              className="btn btn-secondary"
            >
              <Maximize2 size={18} />
              {catalogOpen ? 'Hide Catalog' : 'Show Catalog'}
            </button>
          </div>
        </div>

        {catalogOpen && (
          <div className="side-panel">
            <ProductCatalog onAddItem={handleAddItem} />
            
            {selectedItem && (
              <CustomizationPanel 
                item={selectedItem}
                onUpdateItem={handleUpdateItem}
                onDeleteItem={handleDeleteItem}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
