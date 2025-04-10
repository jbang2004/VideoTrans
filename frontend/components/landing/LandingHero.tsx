'use client'

import React from 'react'

interface LandingHeroProps {
  onTryFree: () => void
}

export const LandingHero: React.FC<LandingHeroProps> = ({ onTryFree }) => {
  return (
    <main id="main-content" className="flex-grow flex flex-col justify-center items-center text-center px-4 -mt-16 transition-opacity duration-500 ease-out fade-in">
      <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-medium tracking-tight mb-4 md:mb-6">
        AI 智能视频翻译
      </h1>
      <p className="text-xl md:text-2xl font-semibold">
        智译视界 ::
      </p>
      <a 
        href="#" 
        id="free-trial-button-bottom"
        onClick={(e) => {
          e.preventDefault();
          onTryFree();
        }}
        className="mt-12 text-base md:text-lg px-8 py-3 rounded-full bg-blue-600/90 text-white font-medium flex items-center space-x-2 hover:bg-blue-700 transition-all duration-300 shadow-lg"
      >
        <span>免费试用</span>
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 7h10v10"/>
          <path d="M7 17 17 7"/>
        </svg>
      </a>
    </main>
  )
} 