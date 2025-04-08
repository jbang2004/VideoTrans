'use client'

import React, { useState, useRef, useEffect } from 'react'
import { cn } from "@/lib/utils"; // 引入 cn 工具函数

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
 const calculateHighlight = () => {
    const activeElement = navItemsRef.current[activeIndex];
    if (activeElement) {
        setHighlightStyle({
            left: activeElement.offsetLeft,
            width: activeElement.offsetWidth,
            height: activeElement.offsetHeight,
            opacity: 1,
        });
    }
  };

  // 初始计算和窗口大小变化时重新计算
  useEffect(() => {
    calculateHighlight(); // 初始计算

    window.addEventListener('resize', calculateHighlight);
    return () => {
      window.removeEventListener('resize', calculateHighlight);
    };
  }, [activeIndex]); // 依赖 activeIndex，当它变化时重新计算


 return (
  <nav id="navbar" className="flex justify-between items-center p-4 md:p-6 transition-opacity duration-500 ease-out fade-in">
   <div className="text-xl md:text-2xl font-semibold text-white">智译视界 ::</div> {/* 确保文字颜色为白色 */}

   {/* 使用 Tailwind 类替代 .nav-container */}
   <div className="relative inline-flex items-center gap-1 bg-gray-100 rounded-lg p-1">
    {/* 高亮背景块 */}
    <div
     className="absolute transition-all duration-300 ease-in-out rounded-md bg-gray-800 pointer-events-none" // 使用 Tailwind 颜色
     style={{
      left: `${highlightStyle.left}px`,
      width: `${highlightStyle.width}px`,
      height: `${highlightStyle.height}px`,
      opacity: highlightStyle.opacity,
      zIndex: 0 // 确保在文字下方
     }}
    />

    {/* 导航项 - 使用 Tailwind 类替代 .nav-button-inner */}
    <a
     ref={(el) => { navItemsRef.current[0] = el; }}
     href="#"
     className={cn(
            "relative z-10 flex items-center px-3 py-1 rounded-md text-sm whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out",
            activeIndex === 0 ? "text-white" : "text-gray-700 hover:bg-black/5" // 高亮时文字白色，否则灰色并有悬停效果
          )}
     onMouseEnter={() => setActiveIndex(0)}
    >
     项目案例
    </a>
    <a
     ref={(el) => { navItemsRef.current[1] = el; }}
     href="#"
     className={cn(
            "relative z-10 flex items-center px-3 py-1 rounded-md text-sm whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out",
            activeIndex === 1 ? "text-white" : "text-gray-700 hover:bg-black/5"
          )}
     onMouseEnter={() => setActiveIndex(1)}
    >
     功能特性
    </a>
    <a
     ref={(el) => { navItemsRef.current[2] = el; }}
     href="#"
     className={cn(
            "relative z-10 flex items-center px-3 py-1 rounded-md text-sm whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out",
            activeIndex === 2 ? "text-white" : "text-gray-700 hover:bg-black/5"
          )}
     onMouseEnter={() => setActiveIndex(2)}
    >
     关于我们
    </a>

    {/* 免费试用按钮 - 使用 Tailwind 类替代 .nav-button-primary-inner */}
    <a
     ref={(el) => { navItemsRef.current[3] = el; }}
     href="#"
     onClick={(e) => {
      e.preventDefault();
      onTryFree();
     }}
     id="free-trial-button-top"
     className={cn(
            "relative z-10 flex items-center px-4 py-1 rounded-md text-sm font-semibold whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out gap-1",
             activeIndex === 3 ? "text-white" : "text-gray-700 hover:bg-black/5" // 同样应用高亮和悬停
           )}
     onMouseEnter={() => setActiveIndex(3)}
    >
     免费试用
     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M7 7h10v10"/>
      <path d="M7 17 17 7"/>
     </svg>
    </a>
   </div>
  </nav>
 )
}