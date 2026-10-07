import React, { useState } from 'react'
import { Plus } from 'lucide-react'
import './ProductCatalog.css'

const products = [
  {
    type: 'bed',
    name: 'Platform Bed',
    brand: 'West Elm',
    price: 899,
    defaultColor: '#8B4513',
    defaultSize: 'queen',
    description: 'Modern platform bed with storage',
    image: '🛏️'
  },
  {
    type: 'desk',
    name: 'Writing Desk',
    brand: 'IKEA',
    price: 179,
    defaultColor: '#D2691E',
    defaultSize: 'standard',
    description: 'Minimalist writing desk',
    image: '🪑'
  },
  {
    type: 'chair',
    name: 'Office Chair',
    brand: 'Herman Miller',
    price: 649,
    defaultColor: '#2F4F4F',
    defaultSize: 'standard',
    description: 'Ergonomic office chair',
    image: '💺'
  },
  {
    type: 'nightstand',
    name: 'Nightstand',
    brand: 'CB2',
    price: 199,
    defaultColor: '#8B4513',
    defaultSize: 'standard',
    description: 'Modern nightstand with drawer',
    image: '🗄️'
  },
  {
    type: 'dresser',
    name: '6-Drawer Dresser',
    brand: 'Pottery Barn',
    price: 599,
    defaultColor: '#A0522D',
    defaultSize: 'standard',
    description: 'Spacious dresser with 6 drawers',
    image: '🗃️'
  },
  {
    type: 'wardrobe',
    name: 'Wardrobe',
    brand: 'Wayfair',
    price: 449,
    defaultColor: '#654321',
    defaultSize: 'standard',
    description: 'Freestanding wardrobe with doors',
    image: '🚪'
  },
  {
    type: 'rug',
    name: 'Area Rug',
    brand: 'Ruggable',
    price: 299,
    defaultColor: '#4169E1',
    defaultSize: 'standard',
    description: 'Soft washable area rug',
    image: '🟦'
  },
  {
    type: 'lamp',
    name: 'Table Lamp',
    brand: 'Target',
    price: 49,
    defaultColor: '#FFD700',
    defaultSize: 'standard',
    description: 'Modern table lamp',
    image: '💡'
  },
  {
    type: 'plant',
    name: 'Potted Plant',
    brand: 'The Sill',
    price: 79,
    defaultColor: '#228B22',
    defaultSize: 'standard',
    description: 'Low-maintenance indoor plant',
    image: '🪴'
  }
]

const ProductCatalog = ({ onAddItem }) => {
  const [filter, setFilter] = useState('all')

  const categories = ['all', 'furniture', 'decor', 'lighting']

  const filteredProducts = products.filter(product => {
    if (filter === 'all') return true
    if (filter === 'furniture') return ['bed', 'desk', 'chair', 'nightstand', 'dresser', 'wardrobe'].includes(product.type)
    if (filter === 'decor') return ['rug', 'plant'].includes(product.type)
    if (filter === 'lighting') return ['lamp'].includes(product.type)
    return true
  })

  return (
    <div className="product-catalog">
      <h2>🛍️ Product Catalog</h2>
      
      <div className="category-filters">
        {categories.map(category => (
          <button
            key={category}
            onClick={() => setFilter(category)}
            className={`filter-btn ${filter === category ? 'active' : ''}`}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </button>
        ))}
      </div>

      <div className="product-grid">
        {filteredProducts.map(product => (
          <div key={product.type} className="product-card">
            <div className="product-image">{product.image}</div>
            <div className="product-info">
              <h4>{product.name}</h4>
              <p className="product-brand">{product.brand}</p>
              <p className="product-price">${product.price}</p>
              <p className="product-description">{product.description}</p>
            </div>
            <button 
              onClick={() => onAddItem(product)}
              className="add-btn"
            >
              <Plus size={16} />
              Add to Room
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ProductCatalog
