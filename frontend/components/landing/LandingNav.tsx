'use client'

import React, { useState, useRef, useEffect } from 'react'

interface LandingNavProps {
  onTryFree: () => void
}

export const LandingNav: React.FC<LandingNavProps> = ({ onTryFree }) => {
  const [activeIndex, setActiveIndex] = useState(3); // 默认"免费试用"按钮高亮
  const navItemsRef = useRef<(HTMLAnchorElement | null)[]>([]);
  const [highlightStyle, setHighlightStyle] = useState({
    left: 0,
    width: 0,
    height: 0,
    opacity: 0,
  });

  // 计算并设置高亮块的位置和尺寸
  useEffect(() => {
    const activeElement = navItemsRef.current[activeIndex];
    if (activeElement) {
      setHighlightStyle({
        left: activeElement.offsetLeft,
        width: activeElement.offsetWidth,
        height: activeElement.offsetHeight,
        opacity: 1,
      });
    }
  }, [activeIndex]);

  // 初始化高亮块位置
  useEffect(() => {
    // 组件挂载后延迟执行一次，确保DOM已经完全渲染
    const timer = setTimeout(() => {
      const activeElement = navItemsRef.current[activeIndex];
      if (activeElement) {
        setHighlightStyle({
          left: activeElement.offsetLeft,
          width: activeElement.offsetWidth,
          height: activeElement.offsetHeight,
          opacity: 1,
        });
      }
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <nav id="navbar" className="flex justify-between items-center p-4 md:p-6 transition-opacity duration-500 ease-out fade-in">
      <div className="text-xl md:text-2xl font-semibold">智译视界 ::</div>
      <div className="nav-container relative">
        {/* 高亮背景块 */}
        <div 
          className="absolute transition-all duration-300 ease-in-out rounded-md bg-[#1f2937] pointer-events-none"
          style={{
            left: `${highlightStyle.left}px`,
            width: `${highlightStyle.width}px`,
            height: `${highlightStyle.height}px`,
            opacity: highlightStyle.opacity,
            zIndex: 0
          }}
        />
        
        <a 
          ref={(el) => { navItemsRef.current[0] = el; }}
          href="#" 
          className="relative z-10 flex items-center px-3 py-1 rounded-md"
          onMouseEnter={() => setActiveIndex(0)}
        >
          <span className={activeIndex === 0 ? "text-white" : "text-gray-700"}>
            项目案例
          </span>
        </a>
        <a 
          ref={(el) => { navItemsRef.current[1] = el; }}
          href="#" 
          className="relative z-10 flex items-center px-3 py-1 rounded-md"
          onMouseEnter={() => setActiveIndex(1)}
        >
          <span className={activeIndex === 1 ? "text-white" : "text-gray-700"}>
            功能特性
          </span>
        </a>
        <a 
          ref={(el) => { navItemsRef.current[2] = el; }}
          href="#" 
          className="relative z-10 flex items-center px-3 py-1 rounded-md"
          onMouseEnter={() => setActiveIndex(2)}
        >
          <span className={activeIndex === 2 ? "text-white" : "text-gray-700"}>
            关于我们
          </span>
        </a>
        <a 
          ref={(el) => { navItemsRef.current[3] = el; }}
          href="#" 
          onClick={(e) => {
            e.preventDefault();
            onTryFree();
          }}
          id="free-trial-button-top" 
          className="relative z-10 flex items-center font-semibold px-4 py-1 rounded-md"
          onMouseEnter={() => setActiveIndex(3)}
        >
          <span className={activeIndex === 3 ? "text-white" : "text-gray-700"}>
            免费试用
          </span>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-1">
            <path d="M7 7h10v10"/>
            <path d="M7 17 17 7"/>
          </svg>
        </a>
      </div>
    </nav>
  )
} 