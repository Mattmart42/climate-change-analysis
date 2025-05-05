import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Homepage from './Homepage.jsx';
import References from './References.jsx';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
        <div>
            <div className='header'>
                <div className='header-logo'>
                  <Link to="/" className='title'>Climate Change Analysis</Link>
                  <img src="src/assets/cloud.png" alt="Logo" className='logo' />
                </div>
                <Link to="/references" className='header-item'>References</Link>
            </div>
            <Routes>
                 <Route path="/" element={<Homepage />} />
                 <Route path="/references" element={<References />} />
            </Routes>
        </div>
    </BrowserRouter>
  );
}