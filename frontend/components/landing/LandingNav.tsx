'use client'

import React from 'react'

interface LandingNavProps {
  onTryFree: () => void
}

export const LandingNav: React.FC<LandingNavProps> = ({ onTryFree }) => {
  return (
    <nav id="navbar" className="flex justify-between items-center p-4 md:p-6 transition-opacity duration-500 ease-out fade-in">
      <div className="text-xl md:text-2xl font-semibold">智译视界 ::</div>
      <div className="nav-container">
        <a href="#" className="nav-button-inner">项目案例</a>
        <a href="#" className="nav-button-inner">功能特性</a>
        <a href="#" className="nav-button-inner">关于我们</a>
        <a 
          href="#" 
          onClick={(e) => {
            e.preventDefault();
            onTryFree();
          }}
          id="free-trial-button-top" 
          className="nav-button-primary-inner"
        >
          <span>免费试用</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M7 7h10v10"/>
            <path d="M7 17 17 7"/>
          </svg>
        </a>
      </div>
    </nav>
  )
} 